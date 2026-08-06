"""Gemini Live DuplexSession implementation.

The Aimer brief picks Gemini Live for the pragmatic v1 duplex model. This adapter
uses google-genai to connect to Gemini's multimodal Live API.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from aimer_core import ContextPacket, FullFrame
from google import genai
from google.genai import types

from duplex_bridge.audio_metrics import compute_rms_int16
from duplex_bridge.deixis import PointerContext, PointerReferentResolver
from duplex_bridge.session import (
    AudioOutCallback,
    DuplexSession,
    InterruptCallback,
    TextOutCallback,
    ToolCallCallback,
    ToolCancellationCallback,
    TurnCompleteCallback,
)

logger = logging.getLogger(__name__)

# System instruction explaining Aimer's role
_SYSTEM_INSTRUCTION = (
    "You are Aimer, a pointer-grounded assistant. The user points at things on screen "
    "and speaks. You receive cursor position, window context, selected text, and screen tiles. "
    "Respond naturally and concisely. "
    "Always respond in the language the user speaks. The on-screen text, [context] "
    "annotations, and screen tiles may be in any language; treat them only as reference for "
    "what the user is pointing at, and never let their language change the language you "
    "reply in. "
    "When the user uses a deictic reference ('this', 'that', 'these', 'here'), resolve the "
    "referent from the marked cursor tile coordinates (tile_cursor) and the "
    "accessibility label or selected-text context provided in [context] annotations, then "
    "respond or act directly; only ask for clarification when the referent is genuinely ambiguous. "
    "The cursor marks a point inside a larger element. Resolve the reference to the whole element "
    "the cursor sits within — the full table cell, link, heading, list item, or paragraph — not "
    "the single character or sub-word at the exact pixel (unless the user explicitly asks "
    "about one "
    "word). Answer only about that pointed-at element; do not describe the whole page or a "
    "neighboring element. "
    "If a [context] annotation carries a resume= field, the connection was interrupted and "
    "restored: it lists background tasks still running or PAUSED awaiting the user's "
    "confirmation. Act on it immediately — for a paused task, ask the user the pending "
    "question out loud and relay their answer via confirm_task; never ignore a resume= note. "
    "App identity comes from two [context] fields: app= is the FOCUSED app (where keyboard "
    "input goes) and pointer_app= is the app under the cursor. When pointer_app= is present "
    "the user is hovering a different app than the focused one, and any deictic or ambiguous "
    "app-identity question — 'what app is this', 'what app am I pointing at', 'what app am I "
    "on', 'where am I' — means the pointed-at app: answer with pointer_app=. Answer with "
    "app= only when the user explicitly asks about the focused or active app or where their "
    "typing goes. When pointer_app= is absent, pointer and focus agree and app= answers "
    "every form. pointer= describes only the pointed-at element and may be imprecise about "
    "app identity — never answer app-identity questions from pointer= wording. "
    "Questions asking you to identify or describe what is on screen or under the pointer "
    "('what is this', 'what am I pointing at', 'describe that') must be answered directly "
    "from the [context] fields (pointer=, pointer_app=, app=, ax=, selected text) and the "
    "visible tile — never by calling delegate_task or computer_use just to look at the "
    "screen; delegated tools are for performing actions, not for identifying what you can "
    "already see. "
    "Before answering ANY question about the status or progress of a delegated or "
    "background task, you MUST call check_tasks first — never answer from memory, and "
    "never say you are 'working on' or have finished a task without having just called "
    "check_tasks, since a task may have already completed or changed status since you "
    "last heard about it."
)

_INITIAL_CONNECT_TIMEOUT_S = 5.0
_RECONNECT_INITIAL_S = 0.5
_RECONNECT_CAP_S = 4.0
_DEFAULT_AUDIO_ACTIVITY_RMS_THRESHOLD = 300.0
_RECV_DIAGNOSTIC_LIMIT = 5
# Rolling record of the last N sends (kind, truncated summary, timestamp offset), logged at
# WARNING on disconnect so a 1007-style close is diagnosable from logs alone (2026-07-14
# live-smoke finding: a barge-in disconnect had zero visibility into what was last sent).
_SEND_RECORD_LIMIT = 5
# Visual context streams on the realtime channels, but the Live API caps video at <=1 FPS,
# so we throttle streaming (tile, plus the text annotation during a turn) to this interval,
# keyed on packet capture time.
_VISUAL_STREAM_MIN_INTERVAL_S = 1.0

# Deixis resolve-on-settle: cursor positions are bucketed into cells of this size (logical
# points); entering a new cell (or a new AX label) re-arms the settle timer, so the resolver
# fires once per distinct target, not per 10 Hz packet, and never mid-mouse-travel.
_DEIXIS_BUCKET_PT = 64


@dataclass
class _Stats:
    """Internal session statistics for debugging."""

    reconnects: int = 0
    dropped_during_reconnect: int = 0
    first_any_response_after_any_send_ms: float | None = None
    first_audio_out_after_any_send_ms: float | None = None
    first_audio_out_after_first_audio_chunk_send_ms: float | None = None
    first_audio_out_after_first_audio_activity_send_ms: float | None = None
    first_audio_out_after_last_audio_activity_ms: float | None = None


class GeminiLiveSession(DuplexSession):
    """Adapter for the Gemini Live API.

    Uses google-genai to establish a bidirectional session with visual context and
    audio streams. Audio output is 24 kHz PCM by default.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str = "GEMINI_API_KEY",
        response_modalities: list[str] | None = None,
        audio_activity_rms_threshold: float = _DEFAULT_AUDIO_ACTIVITY_RMS_THRESHOLD,
        vad_silence_ms: int | None = None,
        vad_start_sensitivity: str | None = None,
        vad_end_sensitivity: str | None = None,
        turn_coverage: str | None = None,
        manual_vad: bool = False,
        thinking_level: str | None = None,
        escalate_with_full_frame: bool = False,
        output_audio_transcription: bool = False,
        tools: list[dict[str, Any]] | None = None,
        deixis_resolver: PointerReferentResolver | None = None,
        deixis_settle_s: float = 0.2,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.response_modalities = response_modalities or ["AUDIO"]
        # The native-audio model (gemini-3.1-flash-live-preview) rejects TEXT-only
        # output (close 1007). To read the model's words — e.g. in the deictic eval —
        # request AUDIO and enable server-side transcription; the transcript text is
        # surfaced through on_text_out alongside any direct text parts.
        self.output_audio_transcription = output_audio_transcription
        self.audio_activity_rms_threshold = audio_activity_rms_threshold
        self.vad_silence_ms = vad_silence_ms
        self.vad_start_sensitivity = vad_start_sensitivity
        self.vad_end_sensitivity = vad_end_sensitivity
        self.turn_coverage = turn_coverage
        # Manual VAD disables Gemini's automatic end-of-turn detection so the caller
        # signals end-of-speech explicitly via send_activity_end(). This removes the
        # server-side silence wait, which dominates end-of-speech→response latency.
        self.manual_vad = manual_vad
        self.thinking_level = thinking_level
        # When True, the latest cached FullFrame is force-sent at turn start (video channel,
        # never interrupts) so the model has full-display layout context for relational deixis
        # ("compare these two windows").
        self.escalate_with_full_frame = escalate_with_full_frame
        # Provider-neutral tool/function declarations (name/description/parameters dicts).
        # Converted to types.Tool in _build_live_config so the model can emit these tool calls;
        # the bridge dispatches them off the hot path via the BackgroundWorker (Week 6).
        self.tools = tools
        # Decoupled deixis (Week 9): a small vision model resolves the pointed-at referent
        # off the hot path, on cursor settle; the result is injected as the pointer= field
        # of the [context] annotation and snapshotted per turn for delegated tasks.
        self._deixis_resolver = deixis_resolver
        self.deixis_settle_s = deixis_settle_s

        self._client: genai.Client | None = None
        self._session: Any = None
        self._session_ctx: Any = None
        self._session_task: asyncio.Task[None] | None = None
        self._open = False
        self._connected = False
        self._close_event = asyncio.Event()
        self._connected_event = asyncio.Event()
        self._api_key: str | None = None
        self._stats = _Stats()
        self._session_open_at: float | None = None
        self._first_visual_send_at: float | None = None
        self._first_audio_chunk_send_at: float | None = None
        self._first_audio_activity_send_at: float | None = None
        self._first_any_send_at: float | None = None
        self._last_audio_activity_send_at: float | None = None
        self._first_any_response_at: float | None = None
        self._first_audio_out_at: float | None = None
        self._recv_diagnostic_count = 0
        self._send_log: deque[tuple[str, str, float]] = deque(maxlen=_SEND_RECORD_LIMIT)
        self._audio_send_count = 0

        # Latest visual context, cached so it can be streamed during a turn / at turn start.
        # The tile rides video (never interrupts); the text annotation rides realtime-text,
        # which only stays safe while the model is idle — i.e. during a manual-VAD turn
        # (activity_start..activity_end). _in_turn gates that; _last_stream_t throttles
        # streaming to <=1 FPS by capture time.
        self._latest_packet: ContextPacket | None = None
        self._last_stream_t: float | None = None
        self._in_turn = False
        # Latest full-frame snapshot, cached for escalation at turn start. Updated whenever
        # a packet carries a FullFrame; None between captures and after reset_timing.
        self._latest_full_frame: FullFrame | None = None

        # Deixis resolve-on-settle state: pending settle task, current target key, latest
        # resolved referent (latest-wins for the annotation), and the referent history —
        # accumulated since the last turn end, snapshotted at activity_end so delegated
        # tasks can ground on everything the user pointed at while (and just before) speaking.
        self._deixis_task: asyncio.Task[None] | None = None
        self._deixis_key: str | None = None
        self._latest_pointer_referent: str | None = None
        self._latest_pointer_referent_app: str | None = None
        self._turn_pointer_history: list[str] = []
        self._last_turn_pointer_history: list[str] = []

        # Reconnect resume context: a reconnect builds a brand-new Live session with no
        # conversational memory, so state the old session promised to act on (e.g. a task
        # paused on a voice confirmation) is silently orphaned. The provider (wired by the
        # bridge, typically TaskManager.pending_note) is consulted after each RE-connect;
        # a non-empty note rides the next turn's [context] annotation as resume= (the only
        # channel that is always safe on the native-audio model) and is sent once.
        self._resume_context_provider: Callable[[], str] | None = None
        self._resume_note: str = ""

        self._audio_callbacks: list[AudioOutCallback] = []
        self._text_callbacks: list[TextOutCallback] = []
        self._tool_callbacks: list[ToolCallCallback] = []
        self._interrupt_callbacks: list[InterruptCallback] = []
        self._tool_cancellation_callbacks: list[ToolCancellationCallback] = []
        self._turn_complete_callbacks: list[TurnCompleteCallback] = []

    @property
    def stats(self) -> dict[str, int | float | None]:
        """Return current session statistics for debugging."""
        return {
            "reconnects": self._stats.reconnects,
            "dropped_during_reconnect": self._stats.dropped_during_reconnect,
            "first_any_response_after_any_send_ms": (
                self._stats.first_any_response_after_any_send_ms
            ),
            "first_audio_out_after_any_send_ms": self._stats.first_audio_out_after_any_send_ms,
            "first_audio_out_after_first_audio_chunk_send_ms": (
                self._stats.first_audio_out_after_first_audio_chunk_send_ms
            ),
            "first_audio_out_after_first_audio_activity_send_ms": (
                self._stats.first_audio_out_after_first_audio_activity_send_ms
            ),
            "first_audio_out_after_last_audio_activity_ms": (
                self._stats.first_audio_out_after_last_audio_activity_ms
            ),
            # Backward-compatible aliases for previous diagnostics.
            "first_any_response_ms": self._stats.first_any_response_after_any_send_ms,
            "first_audio_out_ms": self._stats.first_audio_out_after_any_send_ms,
        }

    async def open(self) -> None:
        """Open the Gemini Live session and start receiving."""
        if self._open:
            raise RuntimeError("GeminiLiveSession is already open")

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        self._api_key = api_key

        self._close_event = asyncio.Event()
        self._connected_event = asyncio.Event()
        self._connected = False
        self._open = True
        self._session_task = asyncio.create_task(self._session_loop())

        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=_INITIAL_CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise RuntimeError("failed to connect to Gemini within 5s") from exc

    def set_resume_context_provider(self, provider: Callable[[], str]) -> None:
        """Register the callable consulted after every reconnect for a resume= note.

        Return the outstanding cross-session state as one line (or "" when none) —
        e.g. tasks still running / paused awaiting a spoken confirmation. The note is
        injected once into the next turn's [context] annotation so the fresh session
        can pick up what the dead one left hanging.
        """
        self._resume_context_provider = provider

    async def _session_loop(self) -> None:
        """Connect to Gemini Live, receive messages, and reconnect on failure.

        Reconnects create a fresh Live session. The system instruction is resent via
        LiveConnectConfig, but in-flight visual context is intentionally lost; pending
        cross-session state re-enters via the resume-context provider (see
        set_resume_context_provider).
        """
        backoff = _RECONNECT_INITIAL_S

        while not self._close_event.is_set():
            try:
                self._client = genai.Client(
                    api_key=self._api_key,
                    http_options=types.HttpOptions(api_version="v1beta"),
                )
                config = self._build_live_config()

                self._session_ctx = self._client.aio.live.connect(model=self.model, config=config)
                self._session = await self._session_ctx.__aenter__()
                self._reset_ttfb_tracking()
                self._session_open_at = time.perf_counter()
                self._connected = True
                self._connected_event.set()
                # Yield once so a caller unblocked by the event above (e.g. open()) gets a
                # chance to run before this task proceeds into the receive loop — mirrors
                # real network behavior (there is always a gap before the first frame) and
                # avoids a same-tick race where a mocked/instant first-receive error would
                # otherwise log the disconnect before the caller's first send is observed.
                await asyncio.sleep(0)
                backoff = _RECONNECT_INITIAL_S

                if self._stats.reconnects and self._resume_context_provider is not None:
                    # Fresh session, amnesiac by construction: queue what the old one
                    # left unfinished for the next turn's annotation.
                    try:
                        self._resume_note = self._resume_context_provider() or ""
                    except Exception:  # noqa: BLE001 — resume aid must never block connect
                        logger.exception("[gemini] resume context provider failed")
                        self._resume_note = ""
                    if self._resume_note:
                        logger.info(
                            "[gemini] reconnected with pending state: %s",
                            self._resume_note[:200],
                        )

                logger.info("[gemini] connected to %s", self.model)
                await self._recv_loop()

                if not self._close_event.is_set():
                    raise RuntimeError("Gemini receive loop ended")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._connected = False
                if self._close_event.is_set():
                    break

                logger.warning("[gemini] session disconnected: %s", e)
                self._log_send_log_on_disconnect()
                self._stats.reconnects += 1

                logger.info("[gemini] reconnecting in %.1fs", backoff)
                try:
                    await asyncio.wait_for(self._close_event.wait(), timeout=backoff)
                    break
                except asyncio.TimeoutError:
                    pass

                backoff = min(backoff * 2, _RECONNECT_CAP_S)
            finally:
                self._connected = False
                if self._session_ctx is not None:
                    try:
                        await self._session_ctx.__aexit__(None, None, None)
                    except Exception as e:
                        logger.warning("[gemini] error closing session: %s", e)
                    self._session_ctx = None
                    self._session = None

                if self._client is not None:
                    with contextlib.suppress(Exception):
                        self._client.close()
                    self._client = None

        self._connected = False

    def _drop_if_reconnecting(self) -> bool:
        """Return True when a send should be dropped during reconnect."""
        if self._connected and self._session is not None:
            return False
        self._stats.dropped_during_reconnect += 1
        return True

    async def send_audio(self, frames: bytes) -> None:
        """Send raw PCM audio frames to Gemini Live."""
        if not self._open:
            raise RuntimeError("Session is not open")
        if self._drop_if_reconnecting():
            return

        now = time.perf_counter()
        self._mark_first_audio_chunk_send(now)
        audio_rms = compute_rms_int16(frames)
        if audio_rms > self.audio_activity_rms_threshold:
            self._mark_audio_activity_send(now, audio_rms)
        self._record_audio_send()
        await self._session.send_realtime_input(
            audio=types.Blob(mime_type="audio/pcm;rate=16000", data=frames)
        )

    async def send_visual_context(self, packet: ContextPacket) -> None:
        """Cache the latest visual context and stream it on the appropriate channel(s).

        Streaming a text annotation per packet on the realtime-text channel was the original
        barge-in flood: realtime text cancels in-progress generation. It is safe only while
        the model is idle, which in manual VAD is the active turn (activity_start ->
        activity_end). So:
          - manual-VAD active turn: stream the tile (video) AND the text annotation — both
            safe (the model is listening, not generating) — so it tracks what the user points
            at / selects mid-sentence;
          - automatic VAD: stream the tile only (no idle-window guarantee — text could
            interrupt a response);
          - manual VAD between turns: cache only (turn start force-sends the freshest context).
        All streaming is throttled to <=1 FPS by capture time.
        """
        if not self._open:
            raise RuntimeError("Session is not open")
        if self._drop_if_reconnecting():
            return

        self._latest_packet = packet
        if packet.full_frame is not None:
            self._latest_full_frame = packet.full_frame
        self._maybe_schedule_deixis(packet)
        if self._in_turn:
            await self._maybe_stream_context(packet, with_text=True)
        elif not self.manual_vad:
            await self._maybe_stream_context(packet, with_text=False)

    async def send_activity_start(self) -> None:
        """Signal the start of a user turn (manual VAD only) and inject visual context.

        After the activity_start marker, the latest cached ContextPacket is injected as
        this turn's visual context (tile + text annotation) via _inject_turn_context.
        No-op unless manual_vad is enabled. With automatic VAD, Gemini detects speech
        boundaries itself and sending activity markers raises an error.
        """
        if not self.manual_vad:
            return
        if not self._open:
            raise RuntimeError("Session is not open")
        # Record the attempt before the connectivity gate: a dropped-during-reconnect
        # activity signal is still diagnostically useful around a disconnect.
        self._record_send("activity_start", "activity_start")
        if self._drop_if_reconnecting():
            return
        await self._session.send_realtime_input(activity_start=types.ActivityStart())
        self._in_turn = True
        if self._latest_packet is not None:
            await self._maybe_stream_context(self._latest_packet, with_text=True, force=True)
        if (
            self.escalate_with_full_frame
            and self._latest_full_frame is not None
            and self._session is not None
        ):
            frame_bytes = base64.b64decode(self._latest_full_frame.frame_b64)
            await self._session.send_realtime_input(
                video=types.Blob(mime_type="image/jpeg", data=frame_bytes)
            )

    async def send_activity_end(self) -> None:
        """Signal the end of a user turn (manual VAD only).

        The server responds immediately with no silence-detection wait, which is
        the primary lever for cutting end-of-speech→response latency. No-op unless
        manual_vad is enabled.
        """
        if not self.manual_vad:
            return
        if not self._open:
            raise RuntimeError("Session is not open")
        # Record the attempt before the connectivity gate — see send_activity_start.
        self._record_send("activity_end", "activity_end")
        if self._drop_if_reconnecting():
            return
        await self._session.send_realtime_input(activity_end=types.ActivityEnd())
        self._in_turn = False
        # Snapshot the deictic referents this turn saw (including any resolved just before
        # activity_start — pointing usually precedes speech) for delegated tasks, then start
        # accumulating for the next turn.
        self._last_turn_pointer_history = list(self._turn_pointer_history)
        self._turn_pointer_history.clear()

    async def _maybe_stream_context(
        self, packet: ContextPacket, *, with_text: bool, force: bool = False
    ) -> None:
        """Stream the cached visual context, throttled to <=1 FPS by capture time (packet.t).

        The tile rides realtime-video (never interrupts). The text annotation rides realtime-
        text and is sent only when with_text=True — safe solely while the model is idle (a
        manual-VAD turn or its start). force=True bypasses the throttle for the turn-start
        injection, where freshness matters more than the cap.
        """
        if (
            not force
            and self._last_stream_t is not None
            and (packet.t - self._last_stream_t) < _VISUAL_STREAM_MIN_INTERVAL_S
        ):
            return
        sent = False
        if packet.hover_region and packet.hover_region.tile_b64:
            self._mark_first_visual_send()
            tile_bytes = base64.b64decode(packet.hover_region.tile_b64)
            await self._session.send_realtime_input(
                video=types.Blob(mime_type="image/jpeg", data=tile_bytes)
            )
            self._record_send("video", f"tile bytes={len(tile_bytes)}")
            sent = True
        if with_text:
            self._mark_first_visual_send()
            annotation = self._build_text_annotation(packet, self._current_pointer_referent(packet))
            if self._resume_note:
                # One-shot: deliver the post-reconnect pending state with this turn's
                # context, then clear so it never repeats.
                annotation = f"{annotation}\nresume={self._resume_note}"
                self._resume_note = ""
            await self._session.send_realtime_input(text=annotation)
            self._record_send("text", annotation)
            sent = True
        if sent:
            self._last_stream_t = packet.t

    @staticmethod
    def _pointer_app(packet: ContextPacket) -> str | None:
        """App that owns the window under the cursor, falling back to the focused app.

        Pointer and focus can be different apps (e.g. cursor resting over Notion while a
        terminal is focused); prefer ``app_under_cursor`` wherever deixis reasons about
        "the app the pointer is in", and only fall back to ``focus_window.app`` when the
        capture provider didn't populate it (older packets, or lookup failure).
        """
        return packet.app_under_cursor or packet.focus_window.app

    def _maybe_schedule_deixis(self, packet: ContextPacket) -> None:
        """Resolve-on-settle: fire the pointer resolver when the cursor settles on a NEW target.

        The target key is a bucketed cursor cell + AX label; a key change cancels any pending
        settle task and arms a new one, so mid-travel positions never resolve and the same
        target never re-resolves. Latest referent wins for the live annotation; every referent
        joins the turn history handed to delegated tasks.
        """
        if self._deixis_resolver is None:
            return
        if packet.hover_region is None or not packet.hover_region.tile_b64:
            return
        ax_label = packet.semantic.accessibility_label or ""
        app = self._pointer_app(packet) or ""
        title = packet.focus_window.title or ""
        key = (
            f"{round(packet.cursor.x / _DEIXIS_BUCKET_PT)}:"
            f"{round(packet.cursor.y / _DEIXIS_BUCKET_PT)}:{ax_label[:80]}:{app}:{title[:80]}"
        )
        if key == self._deixis_key:
            return
        self._deixis_key = key
        if self._deixis_task is not None and not self._deixis_task.done():
            self._deixis_task.cancel()
        self._deixis_task = asyncio.create_task(self._settle_and_resolve(packet, key))

    async def _settle_and_resolve(self, packet: ContextPacket, key: str) -> None:
        await asyncio.sleep(self.deixis_settle_s)
        if key != self._deixis_key or self._deixis_resolver is None:
            return
        if packet.hover_region is None or not packet.hover_region.tile_b64:
            return
        # The focused window's title only describes the tile when the cursor is over the
        # focused app; under a different app it would mislabel the tile (e.g. app='Notion'
        # (window: 'zsh')), so drop it there.
        title_matches_pointer = packet.app_under_cursor in (None, packet.focus_window.app)
        context = PointerContext(
            app=self._pointer_app(packet),
            window_title=packet.focus_window.title if title_matches_pointer else None,
            accessibility_label=packet.semantic.accessibility_label,
            selected_text=packet.semantic.selected_text,
            cursor_tile_x=packet.hover_region.cursor_tile_x,
            cursor_tile_y=packet.hover_region.cursor_tile_y,
        )
        tile_bytes = base64.b64decode(packet.hover_region.tile_b64)
        referent = await self._deixis_resolver.resolve(tile_bytes, context)
        if not referent:
            return
        self._latest_pointer_referent = referent
        self._latest_pointer_referent_app = self._pointer_app(packet)
        if not self._turn_pointer_history or self._turn_pointer_history[-1] != referent:
            self._turn_pointer_history.append(referent)
        logger.info("[deixis] pointer referent: %s", referent[:120])

    def pointer_history_for_delegate(self) -> list[str]:
        """Referents from the last completed turn — the grounding for a delegated task.

        Falls back to the single latest referent when the turn saw none (e.g. the user
        pointed, waited, then spoke a turn with no cursor movement at all).
        """
        if self._last_turn_pointer_history:
            return list(self._last_turn_pointer_history)
        return [self._latest_pointer_referent] if self._latest_pointer_referent else []

    def pointer_click_target(self) -> tuple[str, float, float] | None:
        """Fresh pointer referent + settled cursor coordinates for the ``click_pointer``
        fast path (live-fix 4c): a resolved referent paired with the cursor position from
        the latest cached context packet, in the same logical-point space the ``Computer``
        seam clicks in. Returns ``None`` when no referent has resolved yet (deixis resolver
        disabled, or the cursor hasn't settled on anything) or no packet has been cached.
        """
        if not self._latest_pointer_referent or self._latest_packet is None:
            return None
        cursor = self._latest_packet.cursor
        return self._latest_pointer_referent, cursor.x, cursor.y

    def _current_pointer_referent(self, packet: ContextPacket) -> str | None:
        """Return the latest resolved referent, gated by app staleness and app-claim scrub.

        A referent resolved while focused on app A must not leak into an annotation for
        app B (2026-07-14 live-smoke bug: a stale terminal referent answered "what app am
        I in" after a cmd-tab to Notion). Missing app info on either side (the referent was
        resolved with no app known, or the current packet has none) is treated as a match —
        over-dropping a still-valid referent is worse than keeping it.

        The staleness gate alone is not enough: the resolver's free-text sentence can
        itself assert a wrong app name even when the app-under-cursor bookkeeping is correct
        and fresh (2026-07-23 live run: the same Notion target described as "Notion" and,
        10s later, as "Google Chrome document editor interface" — VLM app-naming flakiness
        in the sentence, not a staleness bug). ``_scrub_contradicting_app_claim`` strips any
        well-known app name from the referent that contradicts the authoritative app, so the
        wrong claim never reaches the live model, while keeping the rest of the description.
        """
        if not self._latest_pointer_referent:
            return None
        current_app = self._pointer_app(packet)
        if current_app is None or self._latest_pointer_referent_app is None:
            return _scrub_contradicting_app_claim(self._latest_pointer_referent, current_app)
        if current_app != self._latest_pointer_referent_app:
            return None
        return _scrub_contradicting_app_claim(self._latest_pointer_referent, current_app)

    @staticmethod
    def _build_text_annotation(packet: ContextPacket, pointer_referent: str | None = None) -> str:
        """Build a concise text annotation (window, cursor, selected text) for a turn.

        When hover_region carries cursor_tile_x/y offsets, a tile_cursor=(x,y) field is
        appended as the deictic anchor for the model. If an accessibility label is available
        it is appended as ax=. When the decoupled resolver has read the pointed-at element,
        its one-line referent is appended as pointer= (the injection point the Week-4
        decoupling eval validated). All fields are omitted when absent.
        """
        parts = ["[context]"]
        if packet.focus_window:
            if packet.focus_window.app:
                parts.append(f"app={packet.focus_window.app}")
            if packet.focus_window.title:
                parts.append(f"title={packet.focus_window.title}")
        if packet.app_under_cursor and packet.app_under_cursor != packet.focus_window.app:
            parts.append(f"pointer_app={packet.app_under_cursor}")
        parts.append(f"cursor=({packet.cursor.x:.0f},{packet.cursor.y:.0f})")
        if (
            packet.hover_region is not None
            and packet.hover_region.cursor_tile_x is not None
            and packet.hover_region.cursor_tile_y is not None
        ):
            tx = round(packet.hover_region.cursor_tile_x)
            ty = round(packet.hover_region.cursor_tile_y)
            parts.append(f"tile_cursor=({tx},{ty})")
        if packet.semantic and packet.semantic.selected_text:
            parts.append(f"selected={packet.semantic.selected_text[:80]}")
        if packet.semantic and packet.semantic.accessibility_label:
            parts.append(f"ax={packet.semantic.accessibility_label[:80]}")
        if pointer_referent:
            parts.append(f"pointer={pointer_referent[:200]}")
        return " ".join(parts)

    def on_audio_out(self, callback: AudioOutCallback) -> None:
        """Register a callback for streaming audio output (24 kHz PCM)."""
        self._audio_callbacks.append(callback)

    def on_text_out(self, callback: TextOutCallback) -> None:
        """Register a callback for text output from the model (text-modality sessions)."""
        self._text_callbacks.append(callback)

    def on_interrupt(self, callback: InterruptCallback) -> None:
        """Register a callback fired when Gemini reports a barge-in interruption."""
        self._interrupt_callbacks.append(callback)

    def on_tool_call(self, callback: ToolCallCallback) -> None:
        """Register a callback for model-emitted tool calls."""
        self._tool_callbacks.append(callback)

    def on_turn_complete(self, callback: TurnCompleteCallback) -> None:
        """Register a callback fired when Gemini's current turn completes."""
        self._turn_complete_callbacks.append(callback)

    def on_tool_call_cancellation(self, callback: ToolCancellationCallback) -> None:
        """Register a callback fired with the ids of tool calls the server cancels."""
        self._tool_cancellation_callbacks.append(callback)

    async def send_tool_response(
        self,
        *,
        name: str,
        call_id: str,
        response: Mapping[str, Any],
        is_error: bool = False,
        final: bool = True,
    ) -> None:
        """Send a FunctionResponse for ``call_id``.

        ``final=True`` uses WHEN_IDLE scheduling: the model speaks the result at the next
        idle moment instead of barging into ongoing speech (for a blocking call the model
        is already idle-waiting, so it fires immediately). ``final=False`` is the silent
        progress ack for NON_BLOCKING tools (``will_continue=True``) — validated end-to-end
        by scripts/diag/probe_tool_response_scheduling.py.
        """
        if self._session is None or not self._connected:
            logger.warning("[gemini] send_tool_response with no live session; dropping %s", name)
            return
        function_response = types.FunctionResponse(
            id=call_id,
            name=name,
            response=dict(response) if not is_error else {"error": dict(response).get("error")},
            scheduling=(
                types.FunctionResponseScheduling.WHEN_IDLE
                if final
                else types.FunctionResponseScheduling.SILENT
            ),
            will_continue=not final,
        )
        await self._session.send_tool_response(function_responses=function_response)
        self._record_send("tool_response", f"name={name} id={call_id} final={final}")
        logger.info(
            "[gemini] tool response sent name=%s id=%s final=%s error=%s",
            name,
            call_id,
            final,
            is_error,
        )

    async def close(self) -> None:
        """Close the Gemini Live session."""
        if not self._open:
            return  # Idempotent

        self._open = False
        self._connected = False
        self._close_event.set()

        if self._deixis_task is not None:
            self._deixis_task.cancel()
            self._deixis_task = None

        if self._session_task is not None:
            self._session_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._session_task
            self._session_task = None

        logger.info("[gemini] closed")

    async def _recv_loop(self) -> None:
        """Receive messages from Gemini Live and dispatch to callbacks.

        ``session.receive()`` yields messages for ONE turn and then ends — that is
        normal turn completion, NOT a disconnect. Re-enter it for each subsequent
        turn on the SAME session so a conversation continues without reconnecting
        (which would lose context). A real disconnect raises inside ``receive()``,
        which propagates to the session loop to reconnect; a pass that yields no
        messages means the stream is closed, so we return and let it reconnect.
        """
        while not self._close_event.is_set():
            received_any = False
            turn_complete_dispatched = False
            async for message in self._session.receive():
                received_any = True
                self._log_recv_diagnostic(message)

                # Barge-in: the user talked over the assistant. Gemini stops
                # generating and flags the turn interrupted; we must drop the audio
                # already buffered downstream or it keeps playing for a beat. Handle
                # before dispatching this message's audio (the interrupted flag can
                # ride along with a trailing chunk).
                if _interrupted_from_message(message):
                    await self._dispatch_interrupt()

                audio_data = _audio_data_from_message(message)
                if audio_data is not None or _tool_call_from_message(message) is not None:
                    self._mark_first_any_response()

                # Dispatch audio output
                if audio_data is not None:
                    self._mark_first_audio_out()
                    for callback in self._audio_callbacks:
                        audio_result = callback(audio_data)
                        if asyncio.iscoroutine(audio_result):
                            await audio_result

                # Dispatch text output (text-modality or hybrid responses)
                text_data = _text_data_from_message(message)
                if text_data:
                    for text_callback in self._text_callbacks:
                        text_result = text_callback(text_data)
                        if asyncio.iscoroutine(text_result):
                            await text_result

                # Dispatch tool calls
                tool_call = _tool_call_from_message(message)
                if tool_call is not None:
                    for tool_callback in self._tool_callbacks:
                        tool_result = tool_callback(tool_call)
                        if asyncio.iscoroutine(tool_result):
                            await tool_result

                # Dispatch server-side tool-call cancellations (e.g. after a barge-in the
                # server may withdraw calls it no longer wants answered).
                cancelled_ids = _tool_cancellation_from_message(message)
                if cancelled_ids:
                    for cancel_callback in self._tool_cancellation_callbacks:
                        cancel_result = cancel_callback(cancelled_ids)
                        if asyncio.iscoroutine(cancel_result):
                            await cancel_result

                # Turn completion: fire promptly on the explicit signal so callers (e.g.
                # the deictic eval harness) don't have to wait out a fixed settle window.
                # Guard against double-firing when the message that carries the flag is
                # also the last message the stream yields (the async-for exhausting below
                # is itself a turn-complete signal per this method's docstring).
                if not turn_complete_dispatched and _turn_complete_from_message(message):
                    turn_complete_dispatched = True
                    await self._dispatch_turn_complete()

            if not received_any:
                return  # stream closed with no data → let the session loop reconnect

            if not turn_complete_dispatched:
                await self._dispatch_turn_complete()

    async def _dispatch_interrupt(self) -> None:
        """Notify interrupt subscribers so backends flush buffered playback."""
        logger.info("[gemini] interruption (barge-in); flushing buffered playback")
        for callback in self._interrupt_callbacks:
            result = callback()
            if asyncio.iscoroutine(result):
                await result

    async def _dispatch_turn_complete(self) -> None:
        """Notify turn-complete subscribers exactly once per finished turn."""
        for callback in self._turn_complete_callbacks:
            result = callback()
            if asyncio.iscoroutine(result):
                await result

    def reset_timing(self) -> None:
        """Reset per-turn TTFB anchors to measure a later turn on the same session.

        The cold first turn after connect is inflated by model warm-up; calling this
        between turns lets a caller measure steady-state (warm) turn latency, which is
        what a user experiences mid-conversation.
        """
        self._reset_ttfb_tracking()

    def _reset_ttfb_tracking(self) -> None:
        self._session_open_at = None
        self._first_visual_send_at = None
        self._last_stream_t = None
        self._latest_full_frame = None
        self._in_turn = False
        self._first_audio_chunk_send_at = None
        self._first_audio_activity_send_at = None
        self._first_any_send_at = None
        self._last_audio_activity_send_at = None
        self._first_any_response_at = None
        self._first_audio_out_at = None
        self._recv_diagnostic_count = 0
        self._stats.first_any_response_after_any_send_ms = None
        self._stats.first_audio_out_after_any_send_ms = None
        self._stats.first_audio_out_after_first_audio_chunk_send_ms = None
        self._stats.first_audio_out_after_first_audio_activity_send_ms = None
        self._stats.first_audio_out_after_last_audio_activity_ms = None

    def _mark_first_visual_send(self) -> None:
        now = time.perf_counter()
        if self._first_visual_send_at is None:
            self._first_visual_send_at = now
        self._mark_first_any_send(now)

    def _mark_first_audio_chunk_send(self, now: float) -> None:
        if self._first_audio_chunk_send_at is None:
            self._first_audio_chunk_send_at = now
        self._mark_first_any_send(now)

    def _mark_audio_activity_send(self, now: float, rms: float) -> None:
        if self._first_audio_activity_send_at is None:
            self._first_audio_activity_send_at = now
            logger.info(
                "[gemini] first_audio_activity rms=%.1f threshold=%.1f",
                rms,
                self.audio_activity_rms_threshold,
            )
        self._last_audio_activity_send_at = now

    def _mark_first_any_send(self, now: float) -> None:
        if self._first_any_send_at is None:
            self._first_any_send_at = now

    def _mark_first_any_response(self) -> None:
        if self._first_any_send_at is None or self._first_any_response_at is not None:
            return
        self._first_any_response_at = time.perf_counter()
        self._stats.first_any_response_after_any_send_ms = (
            self._first_any_response_at - self._first_any_send_at
        ) * 1000.0

    def _mark_first_audio_out(self) -> None:
        if self._first_any_send_at is None or self._first_audio_out_at is not None:
            return
        self._first_audio_out_at = time.perf_counter()
        self._stats.first_audio_out_after_any_send_ms = (
            self._first_audio_out_at - self._first_any_send_at
        ) * 1000.0
        self._stats.first_audio_out_after_first_audio_chunk_send_ms = _elapsed_ms(
            self._first_audio_out_at,
            self._first_audio_chunk_send_at,
        )
        self._stats.first_audio_out_after_first_audio_activity_send_ms = _elapsed_ms(
            self._first_audio_out_at,
            self._first_audio_activity_send_at,
        )
        self._stats.first_audio_out_after_last_audio_activity_ms = _elapsed_ms(
            self._first_audio_out_at,
            self._last_audio_activity_send_at,
        )
        logger.info(
            "[gemini] ttfb first_any_response_after_any_send_ms=%s "
            "first_audio_out_after_any_send_ms=%.1f "
            "first_audio_out_after_first_audio_chunk_send_ms=%s "
            "first_audio_out_after_first_audio_activity_send_ms=%s "
            "first_audio_out_after_last_audio_activity_ms=%s",
            _format_optional_ms(self._stats.first_any_response_after_any_send_ms),
            self._stats.first_audio_out_after_any_send_ms,
            _format_optional_ms(self._stats.first_audio_out_after_first_audio_chunk_send_ms),
            _format_optional_ms(self._stats.first_audio_out_after_first_audio_activity_send_ms),
            _format_optional_ms(self._stats.first_audio_out_after_last_audio_activity_ms),
        )

    def _build_live_config(self) -> types.LiveConnectConfig:
        kwargs: dict[str, Any] = {
            "response_modalities": self.response_modalities,
            "system_instruction": types.Content(parts=[types.Part(text=_SYSTEM_INSTRUCTION)]),
        }
        realtime_input_config = self._build_realtime_input_config()
        if realtime_input_config is not None:
            kwargs["realtime_input_config"] = realtime_input_config
        if self.thinking_level is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=_thinking_level(self.thinking_level)
            )
        if self.output_audio_transcription:
            kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()
        if self.tools:
            kwargs["tools"] = self._build_tools()
        return types.LiveConnectConfig(**kwargs)

    def _build_tools(self) -> list[types.Tool]:
        """Convert provider-neutral tool dicts to a single Gemini types.Tool.

        An optional ``"behavior": "NON_BLOCKING"`` key marks long-running tools: the model
        keeps conversing after emitting the call and receives the result later via a
        will_continue=False FunctionResponse (see send_tool_response).
        """
        declarations = []
        for decl in self.tools or []:
            kwargs: dict[str, Any] = {
                "name": decl["name"],
                "description": decl.get("description", ""),
                "parameters": _schema_from_dict(decl.get("parameters")),
            }
            behavior = decl.get("behavior")
            if behavior:
                kwargs["behavior"] = types.Behavior[str(behavior).upper()]
            declarations.append(types.FunctionDeclaration(**kwargs))
        return [types.Tool(function_declarations=declarations)]

    def _build_realtime_input_config(self) -> types.RealtimeInputConfig | None:
        if (
            not self.manual_vad
            and self.vad_silence_ms is None
            and self.vad_start_sensitivity is None
            and self.vad_end_sensitivity is None
            and self.turn_coverage is None
        ):
            return None

        automatic_activity_detection_kwargs: dict[str, Any] = {}
        if self.manual_vad:
            # Disable server VAD entirely; the caller drives turns via
            # send_activity_start/send_activity_end. Other VAD knobs are ignored
            # by the server in this mode, so we do not set them.
            automatic_activity_detection_kwargs["disabled"] = True
        else:
            if self.vad_silence_ms is not None:
                automatic_activity_detection_kwargs["silence_duration_ms"] = self.vad_silence_ms
            if self.vad_start_sensitivity is not None:
                automatic_activity_detection_kwargs["start_of_speech_sensitivity"] = (
                    _start_sensitivity(self.vad_start_sensitivity)
                )
            if self.vad_end_sensitivity is not None:
                automatic_activity_detection_kwargs["end_of_speech_sensitivity"] = _end_sensitivity(
                    self.vad_end_sensitivity
                )

        kwargs: dict[str, Any] = {}
        if automatic_activity_detection_kwargs:
            kwargs["automatic_activity_detection"] = types.AutomaticActivityDetection(
                **automatic_activity_detection_kwargs
            )
        if self.turn_coverage is not None:
            kwargs["turn_coverage"] = _turn_coverage(self.turn_coverage)
        return types.RealtimeInputConfig(**kwargs)

    def _record_send(self, kind: str, summary: str) -> None:
        """Append a cheap (kind, truncated summary, offset-since-open) entry to the rolling
        send log. Formatting stays trivial — no json dumps — to keep this negligible on the
        hot path; audio call sites must coalesce to a single entry, never one per chunk.
        """
        now = time.perf_counter()
        offset_s = now - self._session_open_at if self._session_open_at is not None else 0.0
        self._send_log.append((kind, summary[:120], offset_s))
        if kind != "audio":
            self._audio_send_count = 0

    def _record_audio_send(self) -> None:
        """Coalesce audio sends into a single 'audio xN chunks' entry — never one per
        chunk, which would flood the rolling send log at 10s of packets/sec.
        """
        now = time.perf_counter()
        offset_s = now - self._session_open_at if self._session_open_at is not None else 0.0
        self._audio_send_count += 1
        summary = f"audio x{self._audio_send_count} chunks"
        if self._send_log and self._send_log[-1][0] == "audio":
            self._send_log[-1] = ("audio", summary, offset_s)
        else:
            self._send_log.append(("audio", summary, offset_s))

    def _log_send_log_on_disconnect(self) -> None:
        if not self._send_log:
            return
        entries = "; ".join(
            f"{kind}@+{offset_s:.2f}s {summary}" for kind, summary, offset_s in self._send_log
        )
        logger.warning("[gemini] recent sends before disconnect: %s", entries)

    def _log_recv_diagnostic(self, message: Any) -> None:
        if self._recv_diagnostic_count >= _RECV_DIAGNOSTIC_LIMIT:
            return

        now = time.perf_counter()
        offsets = [
            _format_offset("open", now, self._session_open_at),
            _format_offset("first_visual", now, self._first_visual_send_at),
            _format_offset("first_audio_chunk", now, self._first_audio_chunk_send_at),
            _format_offset("first_audio_activity", now, self._first_audio_activity_send_at),
        ]
        logger.info(
            "[gemini] recv #%d %s type=%s has_data=%s has_tool_call=%s",
            self._recv_diagnostic_count + 1,
            " ".join(offset for offset in offsets if offset is not None) or "no timing",
            type(message).__name__,
            getattr(message, "data", None) is not None,
            getattr(message, "tool_call", None) is not None,
        )
        self._recv_diagnostic_count += 1


# Common desktop/browser apps the resolver's free-text sentence might wrongly name (2026-07-23
# live-run flakiness: "Google Chrome document editor interface" asserted verbatim for a Notion
# target). This is a heuristic safety net, not the primary fix — the primary fix is the
# resolver prompt no longer inviting the model to name the app (deixis/resolver.py) — but a
# VLM can still slip an app name into free text, so scrub known names that contradict the
# authoritative app before the annotation reaches the live model. Longest names first so
# multi-word names (e.g. "Google Chrome") are matched before their single-word substrings
# ("Chrome") — matches are removed independent of order via the sort in the scrub function.
_KNOWN_APP_NAMES: tuple[str, ...] = (
    "Google Chrome",
    "Microsoft Edge",
    "Chrome",
    "Safari",
    "Firefox",
    "Edge",
    "Notion",
    "Slack",
    "Discord",
    "Zoom",
    "Spotify",
    "Finder",
    "Mail",
    "Messages",
    "Calendar",
    "Notes",
    "Preview",
    "TextEdit",
    "Visual Studio Code",
    "VS Code",
    "Xcode",
    "Microsoft Word",
    "Microsoft Excel",
    "Microsoft PowerPoint",
    "Word",
    "Excel",
    "PowerPoint",
    "Keynote",
    "Pages",
    "Numbers",
    "Figma",
    "Linear",
    "Asana",
    "Trello",
    "Photoshop",
    "Illustrator",
    "WhatsApp",
    "Telegram",
    "Signal",
    "Obsidian",
    "Todoist",
    "Things",
    "Fantastical",
    "Terminal",
    "iTerm2",
    "iTerm",
    "Ghostty",
    "Warp",
    "Alacritty",
    "Google Docs",
    "Google Sheets",
    "Google Slides",
    "Dropbox",
    "OneDrive",
)


def _scrub_contradicting_app_claim(referent: str, authoritative_app: str | None) -> str | None:
    """Strip any well-known app name from ``referent`` that contradicts ``authoritative_app``.

    Returns the (possibly unchanged) referent, or None when nothing descriptive survives the
    scrub (the whole sentence was the wrong app claim). When ``authoritative_app`` is unknown,
    or matches/relates to the referent's app claim, the referent passes through untouched —
    scrubbing is only ever applied against a known-correct signal.
    """
    if not referent or not authoritative_app:
        return referent

    auth_lower = authoritative_app.strip().lower()
    scrubbed = referent
    for name in sorted(_KNOWN_APP_NAMES, key=len, reverse=True):
        name_lower = name.lower()
        if name_lower == auth_lower or name_lower in auth_lower or auth_lower in name_lower:
            continue  # matches (or relates to) the authoritative app — not a contradiction
        # Match whole words, case-sensitively: the resolver writes real app names capitalised
        # ("Google Chrome", "Edge browser"), while common-word uses are lowercase ("the edge of",
        # "password", "email"). Word boundaries + case together keep those descriptions intact and
        # only strip a genuine standalone app-name token.
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        if pattern.search(scrubbed):
            scrubbed = pattern.sub("", scrubbed)

    if scrubbed == referent:
        return referent
    scrubbed = re.sub(r"\s+", " ", scrubbed).strip(" ,.")
    return scrubbed or None


def _audio_data_from_message(message: Any) -> bytes | None:
    """Return PCM bytes from a Gemini message when it is an audio payload."""
    data = getattr(message, "data", None)
    if not data:
        return None

    mime_type = getattr(message, "mime_type", None)
    if not isinstance(mime_type, str):
        mime_type = None
    blob = getattr(message, "blob", None)
    if mime_type is None and blob is not None:
        mime_type = getattr(blob, "mime_type", None)
    if not isinstance(mime_type, str):
        mime_type = None
    if mime_type is not None and "audio" not in str(mime_type).lower():
        return None

    return data if isinstance(data, bytes) else bytes(data)


def _text_data_from_message(message: Any) -> str | None:
    """Return the model's words from a Gemini message.

    Covers both true TEXT-modality parts (``message.text``) and — for the
    AUDIO-modality + ``output_audio_transcription`` path used by the deictic eval —
    the server-side output transcription (``server_content.output_transcription.text``).
    """
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return text
    server_content = getattr(message, "server_content", None)
    transcription = getattr(server_content, "output_transcription", None)
    transcript_text = getattr(transcription, "text", None)
    if isinstance(transcript_text, str) and transcript_text:
        return transcript_text
    return None


def _tool_call_from_message(message: Any) -> Any | None:
    tool_call = getattr(message, "tool_call", None)
    return tool_call if tool_call else None


def _tool_cancellation_from_message(message: Any) -> list[str] | None:
    """Return the cancelled function-call ids, if this message carries a cancellation."""
    cancellation = getattr(message, "tool_call_cancellation", None)
    ids = getattr(cancellation, "ids", None)
    # Require a concrete sequence: server messages carry a list; anything else
    # (absent field, mock artifacts) is not a cancellation.
    if not isinstance(ids, (list, tuple)) or not ids:
        return None
    return [str(i) for i in ids]


def _interrupted_from_message(message: Any) -> bool:
    """Return True when Gemini flags the current turn as interrupted (barge-in)."""
    server_content = getattr(message, "server_content", None)
    return bool(getattr(server_content, "interrupted", False))


def _turn_complete_from_message(message: Any) -> bool:
    """Return True when Gemini flags the current turn as complete.

    Only ``turn_complete`` counts. ``generation_complete`` is deliberately NOT
    treated as equivalent: with output_audio_transcription, trailing transcription
    chunks routinely arrive after ``generation_complete`` but before
    ``turn_complete`` — ending capture on the earlier signal would truncate them
    (the exact artifact the turn-complete hook exists to eliminate).
    """
    server_content = getattr(message, "server_content", None)
    return bool(getattr(server_content, "turn_complete", False))


def _elapsed_ms(end: float, start: float | None) -> float | None:
    if start is None:
        return None
    return (end - start) * 1000.0


def _format_optional_ms(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


def _format_offset(name: str, now: float, start: float | None) -> str | None:
    if start is None:
        return None
    return f"+{(now - start) * 1000.0:.0f}ms since {name}"


def _start_sensitivity(value: str) -> types.StartSensitivity:
    match value:
        case "high":
            return types.StartSensitivity.START_SENSITIVITY_HIGH
        case "low":
            return types.StartSensitivity.START_SENSITIVITY_LOW
        case _:
            raise ValueError(f"unsupported VAD start sensitivity: {value}")


def _end_sensitivity(value: str) -> types.EndSensitivity:
    match value:
        case "high":
            return types.EndSensitivity.END_SENSITIVITY_HIGH
        case "low":
            return types.EndSensitivity.END_SENSITIVITY_LOW
        case _:
            raise ValueError(f"unsupported VAD end sensitivity: {value}")


def _turn_coverage(value: str) -> types.TurnCoverage:
    match value:
        case "all":
            return types.TurnCoverage.TURN_INCLUDES_ALL_INPUT
        case "activity_only":
            return types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY
        case _:
            raise ValueError(f"unsupported turn coverage: {value}")


def _schema_from_dict(schema: dict[str, Any] | None) -> types.Schema | None:
    """Convert a JSON-schema-style dict to a Gemini types.Schema (recursive)."""
    if not schema:
        return None
    kwargs: dict[str, Any] = {}
    if "type" in schema:
        kwargs["type"] = str(schema["type"]).upper()
    if "description" in schema:
        kwargs["description"] = schema["description"]
    if "properties" in schema:
        kwargs["properties"] = {
            key: _schema_from_dict(val) for key, val in schema["properties"].items()
        }
    if "items" in schema:
        kwargs["items"] = _schema_from_dict(schema["items"])
    if "required" in schema:
        kwargs["required"] = schema["required"]
    return types.Schema(**kwargs)


def _thinking_level(value: str) -> types.ThinkingLevel:
    match value:
        case "minimal":
            return types.ThinkingLevel.MINIMAL
        case "low":
            return types.ThinkingLevel.LOW
        case "medium":
            return types.ThinkingLevel.MEDIUM
        case "high":
            return types.ThinkingLevel.HIGH
        case _:
            raise ValueError(f"unsupported thinking level: {value}")
