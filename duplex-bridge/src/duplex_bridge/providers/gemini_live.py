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
import time
from dataclasses import dataclass
from typing import Any

from aimer_core import ContextPacket, FullFrame
from google import genai
from google.genai import types

from duplex_bridge.audio_metrics import compute_rms_int16
from duplex_bridge.session import (
    AudioOutCallback,
    DuplexSession,
    InterruptCallback,
    TextOutCallback,
    ToolCallCallback,
)

logger = logging.getLogger(__name__)

# System instruction explaining Aimer's role
_SYSTEM_INSTRUCTION = (
    "You are Aimer, a pointer-grounded assistant. The user points at things on screen "
    "and speaks. You receive cursor position, window context, selected text, and screen tiles. "
    "Respond naturally and concisely. "
    "Always respond in the language the user speaks. The on-screen text, [context] "
    "annotations, and screen tiles may be in any language; treat them only as reference for "
    "what the user is pointing at, and never let their language change the language you reply in. "
    "When the user uses a deictic reference ('this', 'that', 'these', 'here'), resolve the "
    "referent from the marked cursor tile coordinates (tile_cursor) and the "
    "accessibility label or selected-text context provided in [context] annotations, then "
    "respond or act directly; only ask for clarification when the referent is genuinely ambiguous. "
    "The cursor marks a point inside a larger element. Resolve the reference to the whole element "
    "the cursor sits within — the full table cell, link, heading, list item, or paragraph — not "
    "the single character or sub-word at the exact pixel (unless the user explicitly asks about one "
    "word). Answer only about that pointed-at element; do not describe the whole page or a "
    "neighboring element."
)

_INITIAL_CONNECT_TIMEOUT_S = 5.0
_RECONNECT_INITIAL_S = 0.5
_RECONNECT_CAP_S = 4.0
_DEFAULT_AUDIO_ACTIVITY_RMS_THRESHOLD = 300.0
_RECV_DIAGNOSTIC_LIMIT = 5
# Visual context streams on the realtime channels, but the Live API caps video at <=1 FPS,
# so we throttle streaming (tile, plus the text annotation during a turn) to this interval,
# keyed on packet capture time.
_VISUAL_STREAM_MIN_INTERVAL_S = 1.0


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

        self._audio_callbacks: list[AudioOutCallback] = []
        self._text_callbacks: list[TextOutCallback] = []
        self._tool_callbacks: list[ToolCallCallback] = []
        self._interrupt_callbacks: list[InterruptCallback] = []

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

    async def _session_loop(self) -> None:
        """Connect to Gemini Live, receive messages, and reconnect on failure.

        Reconnects create a fresh Live session. The system instruction is resent via
        LiveConnectConfig, but in-flight visual context is intentionally lost.
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
                backoff = _RECONNECT_INITIAL_S

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
        if self._drop_if_reconnecting():
            return
        await self._session.send_realtime_input(activity_end=types.ActivityEnd())
        self._in_turn = False

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
            sent = True
        if with_text:
            self._mark_first_visual_send()
            await self._session.send_realtime_input(text=self._build_text_annotation(packet))
            sent = True
        if sent:
            self._last_stream_t = packet.t

    @staticmethod
    def _build_text_annotation(packet: ContextPacket) -> str:
        """Build a concise text annotation (window, cursor, selected text) for a turn.

        When hover_region carries cursor_tile_x/y offsets, a tile_cursor=(x,y) field is
        appended as the deictic anchor for the model. If an accessibility label is available
        it is appended as ax=. Both fields are omitted when absent.
        """
        parts = ["[context]"]
        if packet.focus_window:
            if packet.focus_window.app:
                parts.append(f"app={packet.focus_window.app}")
            if packet.focus_window.title:
                parts.append(f"title={packet.focus_window.title}")
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

    async def close(self) -> None:
        """Close the Gemini Live session."""
        if not self._open:
            return  # Idempotent

        self._open = False
        self._connected = False
        self._close_event.set()

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

            if not received_any:
                return  # stream closed with no data → let the session loop reconnect

    async def _dispatch_interrupt(self) -> None:
        """Notify interrupt subscribers so backends flush buffered playback."""
        logger.info("[gemini] interruption (barge-in); flushing buffered playback")
        for callback in self._interrupt_callbacks:
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
        return types.LiveConnectConfig(**kwargs)

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


def _interrupted_from_message(message: Any) -> bool:
    """Return True when Gemini flags the current turn as interrupted (barge-in)."""
    server_content = getattr(message, "server_content", None)
    return bool(getattr(server_content, "interrupted", False))


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
