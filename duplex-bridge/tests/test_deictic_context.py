"""Tests for Week 4 deictic context features.

Covers:
  - _build_text_annotation: tile_cursor= field when hover_region has cursor offsets.
  - escalate_with_full_frame: extra video send at turn start when flag is True.
  - on_text_out callback: fires when a server message carries .text.
  - reset_timing: clears _latest_full_frame.
"""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aimer_core import (
    ContextPacket,
    CursorPosition,
    FocusWindow,
    FullFrame,
    HoverRegion,
    SemanticContext,
)
from duplex_bridge.providers.gemini_live import GeminiLiveSession

# ---------------------------------------------------------------------------
# Async iterator helpers (mirrored from test_gemini_live.py)
# ---------------------------------------------------------------------------


class _AsyncIter:
    """Yields a fixed sequence of messages."""

    def __init__(self, items):
        self._iter = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


class _PendingIter:
    """Stays open until the task is cancelled."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()
        raise StopAsyncIteration


class _QueueIter:
    """Yields messages pushed by the test at runtime."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.queue.get()
        if item is StopAsyncIteration:
            raise StopAsyncIteration
        return item


# ---------------------------------------------------------------------------
# Shared mock fixture (same pattern as test_gemini_live.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_genai_client():
    """Mock the google.genai.Client."""
    with patch("duplex_bridge.providers.gemini_live.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_session = MagicMock()
        mock_session.send_realtime_input = AsyncMock()
        mock_session.receive = MagicMock(return_value=_PendingIter())

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock()

        mock_client.aio.live.connect.return_value = mock_session_ctx

        yield mock_client, mock_session, mock_session_ctx


# ---------------------------------------------------------------------------
# Helper: build a minimal JPEG-ish tile bytes (just a header marker)
# ---------------------------------------------------------------------------

_TILE_BYTES = b"\xff\xd8\xff\xe0fake-tile"
_TILE_B64 = base64.b64encode(_TILE_BYTES).decode()

_FRAME_BYTES = b"\xff\xd8\xff\xe0fake-frame"
_FRAME_B64 = base64.b64encode(_FRAME_BYTES).decode()


# ---------------------------------------------------------------------------
# A. _build_text_annotation — tile_cursor= field
# ---------------------------------------------------------------------------


def test_build_text_annotation_includes_tile_cursor_when_offsets_set():
    """tile_cursor=(x,y) appears in the annotation when hover_region has cursor_tile offsets."""
    packet = ContextPacket(
        cursor=CursorPosition(x=500, y=300),
        hover_region=HoverRegion(
            tile_b64=_TILE_B64,
            cursor_tile_x=37.6,
            cursor_tile_y=22.1,
        ),
    )
    annotation = GeminiLiveSession._build_text_annotation(packet)
    assert "tile_cursor=(38,22)" in annotation
    # The base cursor= field must still be present.
    assert "cursor=(500,300)" in annotation


def test_build_text_annotation_omits_tile_cursor_when_offsets_absent():
    """tile_cursor= is absent when cursor_tile_x/y are None."""
    packet = ContextPacket(
        cursor=CursorPosition(x=100, y=200),
        hover_region=HoverRegion(tile_b64=_TILE_B64),
    )
    annotation = GeminiLiveSession._build_text_annotation(packet)
    assert "tile_cursor" not in annotation


def test_build_text_annotation_omits_tile_cursor_when_no_hover_region():
    """tile_cursor= is absent when there is no hover_region at all."""
    packet = ContextPacket(cursor=CursorPosition(x=10, y=20))
    annotation = GeminiLiveSession._build_text_annotation(packet)
    assert "tile_cursor" not in annotation


def test_build_text_annotation_includes_ax_when_accessibility_label_present():
    """ax= is included when semantic.accessibility_label is set."""
    packet = ContextPacket(
        cursor=CursorPosition(x=0, y=0),
        semantic=SemanticContext(accessibility_label="Close button"),
    )
    annotation = GeminiLiveSession._build_text_annotation(packet)
    assert "ax=Close button" in annotation


def test_build_text_annotation_omits_ax_when_no_accessibility_label():
    """ax= is absent when accessibility_label is None."""
    packet = ContextPacket(
        cursor=CursorPosition(x=0, y=0),
        semantic=SemanticContext(selected_text="some text"),
    )
    annotation = GeminiLiveSession._build_text_annotation(packet)
    assert "ax=" not in annotation


def test_build_text_annotation_all_fields():
    """All deictic fields appear together when fully populated."""
    packet = ContextPacket(
        cursor=CursorPosition(x=400, y=600),
        focus_window=FocusWindow(app="Xcode", title="main.swift"),
        hover_region=HoverRegion(
            tile_b64=_TILE_B64,
            cursor_tile_x=10.0,
            cursor_tile_y=5.0,
        ),
        semantic=SemanticContext(
            selected_text="func viewDidLoad()",
            accessibility_label="Source Editor",
        ),
    )
    annotation = GeminiLiveSession._build_text_annotation(packet)
    assert "app=Xcode" in annotation
    assert "title=main.swift" in annotation
    assert "cursor=(400,600)" in annotation
    assert "tile_cursor=(10,5)" in annotation
    assert "selected=func viewDidLoad()" in annotation
    assert "ax=Source Editor" in annotation


# ---------------------------------------------------------------------------
# B. escalate_with_full_frame — extra video send at turn start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalate_full_frame_sends_frame_at_turn_start(mock_genai_client, monkeypatch):
    """With escalate_with_full_frame=True and a cached FullFrame, send_activity_start
    results in an extra send_realtime_input(video=...) carrying the full frame bytes."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(
        model="gemini-3.1-flash-live-preview",
        manual_vad=True,
        escalate_with_full_frame=True,
    )
    await session.open()

    try:
        full_frame = FullFrame(
            frame_b64=_FRAME_B64,
            width_px=320,
            height_px=200,
        )
        # Send a packet that carries the full frame — should cache it.
        packet = ContextPacket(
            cursor=CursorPosition(x=100, y=100),
            full_frame=full_frame,
        )
        await session.send_visual_context(packet)
        assert session._latest_full_frame is full_frame

        # Now start a turn — should send activity_start, then the full frame.
        mock_session.send_realtime_input.reset_mock()
        await session.send_activity_start()

        call_kwarg_keys = [
            next(iter(c[1])) for c in mock_session.send_realtime_input.call_args_list
        ]
        # Expect: activity_start, then (optionally tile/text from packet if present),
        # then the full-frame video.
        assert "activity_start" in call_kwarg_keys
        assert call_kwarg_keys.count("video") >= 1, "Expected at least one video send"

        # Verify the last video call carries the full-frame bytes.
        video_calls = [
            c for c in mock_session.send_realtime_input.call_args_list if "video" in c[1]
        ]
        last_video_blob = video_calls[-1][1]["video"]
        assert last_video_blob.data == _FRAME_BYTES
        assert last_video_blob.mime_type == "image/jpeg"

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_resume_note_rides_next_turn_annotation_exactly_once(mock_genai_client, monkeypatch):
    """A queued post-reconnect resume note is appended to the next turn's text annotation
    as resume= and never repeats (live finding 2026-07-03: a task paused on confirmation
    was orphaned by an amnesiac reconnect)."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview", manual_vad=True)
    await session.open()
    try:
        packet = ContextPacket(
            cursor=CursorPosition(x=1, y=2),
            hover_region=HoverRegion(tile_b64=_TILE_B64),
        )
        await session.send_visual_context(packet)  # between turns: cache only
        session._resume_note = "task-2 is PAUSED awaiting the user's confirmation"

        mock_session.send_realtime_input.reset_mock()
        await session.send_activity_start()
        texts = [
            c[1]["text"] for c in mock_session.send_realtime_input.call_args_list if "text" in c[1]
        ]
        assert any("resume=task-2 is PAUSED" in t for t in texts)
        assert session._resume_note == ""

        # Next turn: no repeat.
        await session.send_activity_end()
        mock_session.send_realtime_input.reset_mock()
        await session.send_activity_start()
        texts = [
            c[1]["text"] for c in mock_session.send_realtime_input.call_args_list if "text" in c[1]
        ]
        assert texts and not any("resume=" in t for t in texts)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_escalate_full_frame_not_sent_when_flag_false(mock_genai_client, monkeypatch):
    """With escalate_with_full_frame=False, no full-frame video is sent even when cached."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(
        model="gemini-3.1-flash-live-preview",
        manual_vad=True,
        escalate_with_full_frame=False,
    )
    await session.open()

    try:
        full_frame = FullFrame(frame_b64=_FRAME_B64, width_px=320, height_px=200)
        packet = ContextPacket(cursor=CursorPosition(x=100, y=100), full_frame=full_frame)
        await session.send_visual_context(packet)

        mock_session.send_realtime_input.reset_mock()
        await session.send_activity_start()

        call_kwarg_keys = [
            next(iter(c[1])) for c in mock_session.send_realtime_input.call_args_list
        ]
        # Only activity_start should be sent (no tile/text since no hover_region.tile_b64
        # on the cached packet, and no full-frame escalation).
        assert "activity_start" in call_kwarg_keys
        # No video call should carry _FRAME_BYTES — the full frame must be absent.
        video_calls = [
            c for c in mock_session.send_realtime_input.call_args_list if "video" in c[1]
        ]
        for vc in video_calls:
            assert vc[1]["video"].data != _FRAME_BYTES, (
                "Full frame bytes must not be sent when escalate_with_full_frame=False"
            )

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_escalate_full_frame_not_sent_when_no_cached_frame(mock_genai_client, monkeypatch):
    """When escalate_with_full_frame=True but no FullFrame has been cached, no extra send."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(
        model="gemini-3.1-flash-live-preview",
        manual_vad=True,
        escalate_with_full_frame=True,
    )
    await session.open()

    try:
        # Send a packet WITHOUT full_frame — nothing cached.
        packet = ContextPacket(cursor=CursorPosition(x=50, y=50))
        await session.send_visual_context(packet)
        assert session._latest_full_frame is None

        mock_session.send_realtime_input.reset_mock()
        await session.send_activity_start()

        # Only activity_start; no video frame escalation.
        video_calls_with_frame = [
            c
            for c in mock_session.send_realtime_input.call_args_list
            if "video" in c[1] and c[1]["video"].data == _FRAME_BYTES
        ]
        assert len(video_calls_with_frame) == 0

    finally:
        await session.close()


# ---------------------------------------------------------------------------
# C. on_text_out callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_text_out_callback_fires_when_message_has_text(mock_genai_client, monkeypatch):
    """When the server sends a message with .text, registered text callbacks are invoked."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    receive_iter = _QueueIter()
    mock_session.receive = MagicMock(return_value=receive_iter)

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")

    texts_received: list[str] = []

    def text_cb(text: str) -> None:
        texts_received.append(text)

    session.on_text_out(text_cb)
    await session.open()

    try:
        text_message = MagicMock()
        text_message.text = "Here is the answer."
        text_message.data = None
        text_message.tool_call = None
        text_message.server_content.interrupted = False

        await receive_iter.queue.put(text_message)
        await asyncio.sleep(0.1)

        assert texts_received == ["Here is the answer."]

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_on_text_out_callback_not_fired_for_audio_only_message(
    mock_genai_client, monkeypatch
):
    """A message that has audio data but no .text must NOT fire text callbacks."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    receive_iter = _QueueIter()
    mock_session.receive = MagicMock(return_value=receive_iter)

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")

    texts_received: list[str] = []
    session.on_text_out(lambda t: texts_received.append(t))
    await session.open()

    try:
        audio_message = MagicMock()
        audio_message.text = None  # no text
        audio_message.data = b"pcm-audio"
        audio_message.mime_type = "audio/pcm;rate=24000"
        audio_message.tool_call = None
        audio_message.server_content.interrupted = False

        await receive_iter.queue.put(audio_message)
        await asyncio.sleep(0.1)

        assert texts_received == []

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_on_text_out_multiple_callbacks(mock_genai_client, monkeypatch):
    """Multiple registered text callbacks all receive the text."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    receive_iter = _QueueIter()
    mock_session.receive = MagicMock(return_value=receive_iter)

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")

    bucket_a: list[str] = []
    bucket_b: list[str] = []
    session.on_text_out(lambda t: bucket_a.append(t))
    session.on_text_out(lambda t: bucket_b.append(t))
    await session.open()

    try:
        msg = MagicMock()
        msg.text = "hello"
        msg.data = None
        msg.tool_call = None
        msg.server_content.interrupted = False

        await receive_iter.queue.put(msg)
        await asyncio.sleep(0.1)

        assert bucket_a == ["hello"]
        assert bucket_b == ["hello"]

    finally:
        await session.close()


# ---------------------------------------------------------------------------
# D. reset_timing clears _latest_full_frame
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_timing_clears_latest_full_frame(mock_genai_client, monkeypatch):
    """reset_timing() sets _latest_full_frame back to None."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")
    await session.open()

    try:
        full_frame = FullFrame(frame_b64=_FRAME_B64, width_px=640, height_px=400)
        packet = ContextPacket(cursor=CursorPosition(x=0, y=0), full_frame=full_frame)
        await session.send_visual_context(packet)
        assert session._latest_full_frame is full_frame

        session.reset_timing()
        assert session._latest_full_frame is None

    finally:
        await session.close()


# ---------------------------------------------------------------------------
# E. output_audio_transcription — eval reads the model's words as a transcript
# ---------------------------------------------------------------------------


def test_live_config_omits_transcription_by_default():
    """Without the flag, the live config carries no output_audio_transcription."""
    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")
    config = session._build_live_config()
    assert getattr(config, "output_audio_transcription", None) is None


def test_live_config_includes_transcription_when_enabled():
    """With output_audio_transcription=True, the config requests server-side transcription."""
    session = GeminiLiveSession(
        model="gemini-3.1-flash-live-preview",
        output_audio_transcription=True,
    )
    config = session._build_live_config()
    assert config.output_audio_transcription is not None


@pytest.mark.asyncio
async def test_on_text_out_fires_from_output_transcription(mock_genai_client, monkeypatch):
    """When response is AUDIO, the transcript text (server_content.output_transcription)
    is surfaced through on_text_out even though message.text is empty."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    receive_iter = _QueueIter()
    mock_session.receive = MagicMock(return_value=receive_iter)

    session = GeminiLiveSession(
        model="gemini-3.1-flash-live-preview",
        output_audio_transcription=True,
    )
    collected: list[str] = []
    session.on_text_out(collected.append)
    await session.open()

    try:
        msg = MagicMock()
        msg.text = None  # AUDIO modality: no direct text part
        msg.data = None
        msg.tool_call = None
        msg.server_content.interrupted = False
        msg.server_content.output_transcription.text = "a red error line"

        await receive_iter.queue.put(msg)
        await asyncio.sleep(0.1)

        assert collected == ["a red error line"]

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_visual_context_caches_full_frame_always(mock_genai_client, monkeypatch):
    """send_visual_context always caches full_frame regardless of turn state or VAD mode."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")  # auto-VAD
    await session.open()

    try:
        full_frame = FullFrame(frame_b64=_FRAME_B64, width_px=320, height_px=200)
        # In auto-VAD between turns, the packet is cached; full_frame must also be cached.
        packet = ContextPacket(cursor=CursorPosition(x=50, y=50), full_frame=full_frame)
        await session.send_visual_context(packet)
        assert session._latest_full_frame is full_frame

    finally:
        await session.close()


# ---------------------------------------------------------------------------
# F. Week 9: decoupled deixis — pointer= annotation + resolve-on-settle
# ---------------------------------------------------------------------------


def _packet(x: float = 500, y: float = 300, ax: str | None = None) -> ContextPacket:
    return ContextPacket(
        cursor=CursorPosition(x=x, y=y),
        hover_region=HoverRegion(tile_b64=_TILE_B64, cursor_tile_x=10.0, cursor_tile_y=20.0),
        semantic=SemanticContext(accessibility_label=ax),
    )


class _FakeResolver:
    """Records resolve calls; returns canned referents in order (cycling the last)."""

    def __init__(self, referents: list[str]) -> None:
        self.referents = referents
        self.calls: list[tuple[bytes, object]] = []

    async def resolve(self, tile_jpeg: bytes, context=None) -> str:
        self.calls.append((tile_jpeg, context))
        index = min(len(self.calls) - 1, len(self.referents) - 1)
        return self.referents[index]


def test_build_text_annotation_includes_pointer_when_provided():
    """The resolved referent appears as pointer= (the eval-validated injection point)."""
    annotation = GeminiLiveSession._build_text_annotation(
        _packet(), "the 'Submit' button of the checkout form"
    )
    assert "pointer=the 'Submit' button of the checkout form" in annotation


def test_build_text_annotation_omits_pointer_when_none():
    annotation = GeminiLiveSession._build_text_annotation(_packet())
    assert "pointer=" not in annotation


async def test_settle_resolves_once_per_target_and_caches_latest():
    """Packets in the same cursor bucket resolve once; the referent and history update."""
    resolver = _FakeResolver(["the Coffee heading"])
    session = GeminiLiveSession(model="m", deixis_resolver=resolver, deixis_settle_s=0.01)
    session._maybe_schedule_deixis(_packet(x=500, y=300))
    session._maybe_schedule_deixis(_packet(x=505, y=302))  # same 64pt bucket — no re-arm
    await asyncio.sleep(0.08)

    assert len(resolver.calls) == 1
    assert resolver.calls[0][0] == _TILE_BYTES
    assert session._latest_pointer_referent == "the Coffee heading"
    assert session._turn_pointer_history == ["the Coffee heading"]


async def test_settle_debounce_restarts_on_new_target():
    """A new target inside the settle window cancels the pending resolve — only the last fires."""
    resolver = _FakeResolver(["only-one"])
    session = GeminiLiveSession(model="m", deixis_resolver=resolver, deixis_settle_s=0.05)
    session._maybe_schedule_deixis(_packet(x=0, y=0))
    await asyncio.sleep(0.01)  # still inside the settle window
    session._maybe_schedule_deixis(_packet(x=1000, y=1000))
    await asyncio.sleep(0.15)

    assert len(resolver.calls) == 1  # first target never resolved
    assert session._latest_pointer_referent == "only-one"


async def test_settle_accumulates_history_across_targets():
    """Distinct targets resolved in sequence all join the turn history (multi-deixis)."""
    resolver = _FakeResolver(["the red mug", "the blue mug"])
    session = GeminiLiveSession(model="m", deixis_resolver=resolver, deixis_settle_s=0.01)
    session._maybe_schedule_deixis(_packet(x=0, y=0))
    await asyncio.sleep(0.08)
    session._maybe_schedule_deixis(_packet(x=1000, y=1000))
    await asyncio.sleep(0.08)

    assert session._turn_pointer_history == ["the red mug", "the blue mug"]


async def test_turn_end_snapshots_history_for_delegate(mock_genai_client, monkeypatch):
    """activity_end snapshots the referent history for delegated tasks and clears it."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    session = GeminiLiveSession(model="m", manual_vad=True)
    await session.open()
    try:
        session._turn_pointer_history = ["the red mug", "the blue mug"]
        await session.send_activity_end()
        assert session.pointer_history_for_delegate() == ["the red mug", "the blue mug"]
        assert session._turn_pointer_history == []
    finally:
        await session.close()


def test_pointer_history_falls_back_to_latest_referent():
    """With no completed-turn history, the single latest referent is the grounding."""
    session = GeminiLiveSession(model="m")
    assert session.pointer_history_for_delegate() == []
    session._latest_pointer_referent = "the Coffee heading"
    assert session.pointer_history_for_delegate() == ["the Coffee heading"]


async def test_resolver_returns_empty_string_on_failure(monkeypatch):
    """The resolver never raises into the caller — missing key/API failure yields ''."""
    from duplex_bridge.deixis import PointerReferentResolver

    monkeypatch.delenv("AIMER_TEST_MISSING_KEY", raising=False)
    resolver = PointerReferentResolver(api_key_env="AIMER_TEST_MISSING_KEY")
    assert await resolver.resolve(b"\xff\xd8tile") == ""


async def test_app_switch_rearms_settle_task_even_with_same_cursor_and_ax():
    """Switching the focused app with a stationary cursor and identical AX label must
    re-arm the settle task and re-resolve the referent (2026-07-14 live-smoke bug: a
    cmd-tab from Ghostty to Notion left the stale Ghostty referent injected because the
    settle key — cursor bucket + ax_label only — was unchanged)."""
    resolver = _FakeResolver(["a terminal prompt", "a Notion page titled Roadmap"])
    session = GeminiLiveSession(model="m", deixis_resolver=resolver, deixis_settle_s=0.01)

    packet_ghostty = ContextPacket(
        cursor=CursorPosition(x=500, y=300),
        focus_window=FocusWindow(app="Ghostty", title="zsh"),
        hover_region=HoverRegion(tile_b64=_TILE_B64, cursor_tile_x=10.0, cursor_tile_y=20.0),
        semantic=SemanticContext(accessibility_label="terminal output"),
    )
    session._maybe_schedule_deixis(packet_ghostty)
    await asyncio.sleep(0.05)
    assert len(resolver.calls) == 1
    assert session._latest_pointer_referent == "a terminal prompt"

    key_after_ghostty = session._deixis_key

    packet_notion = ContextPacket(
        cursor=CursorPosition(x=500, y=300),  # identical bucket
        focus_window=FocusWindow(app="Notion", title="zsh"),  # only the app changed
        hover_region=HoverRegion(tile_b64=_TILE_B64, cursor_tile_x=10.0, cursor_tile_y=20.0),
        semantic=SemanticContext(accessibility_label="terminal output"),  # identical AX
    )
    session._maybe_schedule_deixis(packet_notion)

    # The key must change purely from the app switch, or the resolver never re-arms.
    assert session._deixis_key != key_after_ghostty

    await asyncio.sleep(0.05)

    assert len(resolver.calls) == 2, "app switch with stationary cursor must re-resolve"
    assert session._latest_pointer_referent == "a Notion page titled Roadmap"


def _packet_with_app(app: str | None, referent_present: bool = True) -> ContextPacket:
    return ContextPacket(
        cursor=CursorPosition(x=1, y=2),
        focus_window=FocusWindow(app=app, title="w" if app is not None else None),
        hover_region=HoverRegion(tile_b64=_TILE_B64),
    )


def test_current_pointer_referent_omitted_when_focused_app_differs():
    """A referent resolved from app A must not leak into an annotation for app B — the
    exact 21:12 live-smoke bug (Ghostty referent answered while focused on Notion). A
    missing app on either side is treated as a match so we never over-drop."""
    session = GeminiLiveSession(model="m")
    session._latest_pointer_referent = "the Coffee heading"
    session._latest_pointer_referent_app = "Ghostty"

    other_app_packet = _packet_with_app("Notion")
    same_app_packet = _packet_with_app("Ghostty")
    no_app_packet = _packet_with_app(None)

    assert session._current_pointer_referent(other_app_packet) is None
    assert session._current_pointer_referent(same_app_packet) == "the Coffee heading"
    # Current packet carries no app at all — treat as a match, don't drop.
    assert session._current_pointer_referent(no_app_packet) == "the Coffee heading"

    # Referent resolved with no app info at all — always included regardless of current app.
    session._latest_pointer_referent_app = None
    assert session._current_pointer_referent(other_app_packet) == "the Coffee heading"

    # The referent itself (and delegate-facing state) must not be cleared by staleness.
    assert session.pointer_history_for_delegate() == ["the Coffee heading"]
    session._latest_packet = other_app_packet
    assert session.pointer_click_target() is not None


async def test_stream_context_annotation_gated_by_current_focused_app(
    mock_genai_client, monkeypatch
):
    """The realtime-text annotation actually sent must reflect the app-gated referent,
    not the raw latest one — this is the live call site, not just the helper."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview", manual_vad=True)
    await session.open()
    try:
        session._latest_pointer_referent = "the Submit button"
        session._latest_pointer_referent_app = "Ghostty"

        stale_packet = ContextPacket(
            cursor=CursorPosition(x=1, y=2),
            focus_window=FocusWindow(app="Notion", title="Notes"),
            hover_region=HoverRegion(tile_b64=_TILE_B64),
        )
        mock_session.send_realtime_input.reset_mock()
        await session._maybe_stream_context(stale_packet, with_text=True, force=True)
        texts = [
            c[1]["text"] for c in mock_session.send_realtime_input.call_args_list if "text" in c[1]
        ]
        assert texts and "pointer=" not in texts[0]

        fresh_packet = ContextPacket(
            cursor=CursorPosition(x=1, y=2),
            focus_window=FocusWindow(app="Ghostty", title="zsh"),
            hover_region=HoverRegion(tile_b64=_TILE_B64),
        )
        mock_session.send_realtime_input.reset_mock()
        await session._maybe_stream_context(fresh_packet, with_text=True, force=True)
        texts = [
            c[1]["text"] for c in mock_session.send_realtime_input.call_args_list if "text" in c[1]
        ]
        assert texts and "pointer=the Submit button" in texts[0]
    finally:
        await session.close()


def test_resolver_prompt_includes_context_hints():
    from duplex_bridge.deixis import PointerContext, PointerReferentResolver

    prompt = PointerReferentResolver._build_prompt(
        PointerContext(
            app="Safari",
            window_title="Coffee - Wikipedia",
            accessibility_label="Coffee heading",
            selected_text="arabica",
            cursor_tile_x=37.2,
            cursor_tile_y=88.9,
        )
    )
    assert "cursor_in_tile=(37,89)" in prompt
    assert "app=Safari" in prompt
    assert "ax=Coffee heading" in prompt
    assert "selected=arabica" in prompt


# ---------------------------------------------------------------------------
# G. Week-9 live-smoke fix A1: resolver prompt states a known app as authoritative fact
# (2026-07-14 finding: Flash-Lite guessed "browser tab" for Notion, an Electron app whose
# pixels look like a web page, because app/window were folded into an optional hints line).
# ---------------------------------------------------------------------------


def test_resolver_prompt_states_known_app_as_authoritative_fact():
    from duplex_bridge.deixis import PointerContext, PointerReferentResolver

    prompt = PointerReferentResolver._build_prompt(
        PointerContext(app="Notion", window_title="Roadmap")
    )
    lowered = prompt.lower()

    # The app must be named as a known fact, not just listed in a trailing hints line.
    assert "notion" in lowered
    # An explicit anti-guessing instruction must be present.
    assert "never guess" in lowered and "app" in lowered
    # The model must be told to attribute the pointed-at element to that app.
    assert "belong" in lowered or "attribute" in lowered


def test_resolver_prompt_without_app_keeps_current_shape_no_authoritative_sentence():
    from duplex_bridge.deixis import PointerContext, PointerReferentResolver

    prompt = PointerReferentResolver._build_prompt(
        PointerContext(window_title=None, accessibility_label="a heading")
    )
    lowered = prompt.lower()

    # No app is known — the anti-guessing / authoritative-app language must not appear.
    assert "never guess" not in lowered
    assert "the app identity is known" not in lowered


def test_resolver_prompt_none_context_keeps_current_shape_no_authoritative_sentence():
    from duplex_bridge.deixis import PointerReferentResolver

    prompt = PointerReferentResolver._build_prompt(None)
    lowered = prompt.lower()

    assert "never guess" not in lowered
    assert "the app identity is known" not in lowered


# ---------------------------------------------------------------------------
# H. Week-9 live-smoke fix A2: live-model grounding rule — app= is authoritative for
# "which app am I in" questions; pointer= may be imprecise about app identity.
# ---------------------------------------------------------------------------


def test_system_instruction_pins_app_authoritative_grounding_rule():
    from duplex_bridge.providers.gemini_live import _SYSTEM_INSTRUCTION

    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "app=" in lowered
    assert "authoritative" in lowered
    assert "pointer=" in lowered
    # The instruction must warn that pointer= may be imprecise about app identity, and that
    # app-identity questions must never be answered from pointer= wording.
    assert "imprecise" in lowered or "may be wrong" in lowered or "may not" in lowered


# ---------------------------------------------------------------------------
# I. Week-9 live-smoke fix A3: app_under_cursor preferred over focus_window.app at all
# four call sites (deixis settle key, resolver hint, referent-app bookkeeping, staleness
# gate) — pointer and focus can be different apps.
# ---------------------------------------------------------------------------


def _packet_with_apps(
    focus_app: str | None,
    under_cursor_app: str | None,
    *,
    x: float = 500,
    y: float = 300,
    ax: str | None = None,
) -> ContextPacket:
    return ContextPacket(
        cursor=CursorPosition(x=x, y=y),
        focus_window=FocusWindow(app=focus_app, title="w" if focus_app else None),
        app_under_cursor=under_cursor_app,
        hover_region=HoverRegion(tile_b64=_TILE_B64, cursor_tile_x=10.0, cursor_tile_y=20.0),
        semantic=SemanticContext(accessibility_label=ax),
    )


async def test_deixis_settle_key_prefers_app_under_cursor_over_focus_window():
    """The settle key must re-arm on app_under_cursor changes even when focus_window.app
    and the cursor bucket are unchanged (pointer moved onto a different app's window while
    focus stayed elsewhere)."""
    resolver = _FakeResolver(["a terminal prompt", "a Notion page titled Roadmap"])
    session = GeminiLiveSession(model="m", deixis_resolver=resolver, deixis_settle_s=0.01)

    packet_a = _packet_with_apps("Ghostty", "Ghostty", ax="terminal output")
    session._maybe_schedule_deixis(packet_a)
    await asyncio.sleep(0.05)
    key_after_a = session._deixis_key

    # Same cursor bucket, same focused app, same AX — only app_under_cursor changes.
    packet_b = _packet_with_apps("Ghostty", "Notion", ax="terminal output")
    session._maybe_schedule_deixis(packet_b)

    assert session._deixis_key != key_after_a, (
        "app_under_cursor change must re-arm the settle key even with focus_window.app unchanged"
    )
    await asyncio.sleep(0.05)
    assert len(resolver.calls) == 2


async def test_deixis_settle_key_falls_back_to_focus_window_when_app_under_cursor_none():
    """With app_under_cursor absent, the settle key behaves exactly as before (focus_window.app)."""
    resolver = _FakeResolver(["a terminal prompt"])
    session = GeminiLiveSession(model="m", deixis_resolver=resolver, deixis_settle_s=0.01)

    packet = _packet_with_apps("Ghostty", None, ax="terminal output")
    session._maybe_schedule_deixis(packet)
    await asyncio.sleep(0.05)
    assert len(resolver.calls) == 1

    # An identical packet (still app_under_cursor=None) must NOT re-arm.
    key_before = session._deixis_key
    session._maybe_schedule_deixis(_packet_with_apps("Ghostty", None, ax="terminal output"))
    assert session._deixis_key == key_before


async def test_resolver_hint_app_prefers_app_under_cursor():
    """PointerContext.app handed to the resolver is app_under_cursor when present."""
    resolver = _FakeResolver(["something"])
    session = GeminiLiveSession(model="m", deixis_resolver=resolver, deixis_settle_s=0.01)

    packet = _packet_with_apps("Ghostty", "Notion")
    session._maybe_schedule_deixis(packet)
    await asyncio.sleep(0.05)

    assert len(resolver.calls) == 1
    context = resolver.calls[0][1]
    assert context.app == "Notion"


async def test_resolver_hint_window_title_dropped_when_pointer_app_differs():
    """The focused window's title only describes the tile when the cursor is over the
    focused app — otherwise the resolver would be told e.g. app='Notion' (window: 'zsh')."""
    resolver = _FakeResolver(["something", "something else"])
    session = GeminiLiveSession(model="m", deixis_resolver=resolver, deixis_settle_s=0.01)

    session._maybe_schedule_deixis(_packet_with_apps("Ghostty", "Notion"))
    await asyncio.sleep(0.05)
    assert resolver.calls[0][1].window_title is None

    # Cursor over the focused app itself — the title applies and is kept.
    session._maybe_schedule_deixis(_packet_with_apps("Ghostty", "Ghostty", x=700))
    await asyncio.sleep(0.05)
    assert resolver.calls[1][1].window_title == "w"


async def test_resolver_hint_app_falls_back_to_focus_window_when_app_under_cursor_none():
    resolver = _FakeResolver(["something"])
    session = GeminiLiveSession(model="m", deixis_resolver=resolver, deixis_settle_s=0.01)

    packet = _packet_with_apps("Ghostty", None)
    session._maybe_schedule_deixis(packet)
    await asyncio.sleep(0.05)

    assert len(resolver.calls) == 1
    context = resolver.calls[0][1]
    assert context.app == "Ghostty"


async def test_latest_pointer_referent_app_records_app_under_cursor():
    resolver = _FakeResolver(["the Coffee heading"])
    session = GeminiLiveSession(model="m", deixis_resolver=resolver, deixis_settle_s=0.01)

    packet = _packet_with_apps("Ghostty", "Notion")
    session._maybe_schedule_deixis(packet)
    await asyncio.sleep(0.05)

    assert session._latest_pointer_referent_app == "Notion"


async def test_latest_pointer_referent_app_falls_back_when_app_under_cursor_none():
    resolver = _FakeResolver(["the Coffee heading"])
    session = GeminiLiveSession(model="m", deixis_resolver=resolver, deixis_settle_s=0.01)

    packet = _packet_with_apps("Ghostty", None)
    session._maybe_schedule_deixis(packet)
    await asyncio.sleep(0.05)

    assert session._latest_pointer_referent_app == "Ghostty"


def test_current_pointer_referent_staleness_gate_prefers_app_under_cursor():
    """The staleness gate must compare against app_under_cursor when present — a referent
    resolved for the app under the cursor must not be dropped just because a DIFFERENT app
    is focused (and vice versa)."""
    session = GeminiLiveSession(model="m")
    session._latest_pointer_referent = "the Coffee heading"
    session._latest_pointer_referent_app = "Notion"

    # Focus is Ghostty, but the packet's app_under_cursor is Notion — must match (fresh).
    matching_packet = _packet_with_apps("Ghostty", "Notion")
    assert session._current_pointer_referent(matching_packet) == "the Coffee heading"

    # app_under_cursor differs from the referent's app — stale, must be dropped, even
    # though focus_window.app happens to equal the referent's app.
    stale_packet = _packet_with_apps("Notion", "Ghostty")
    assert session._current_pointer_referent(stale_packet) is None


def test_current_pointer_referent_staleness_gate_falls_back_when_app_under_cursor_none():
    """With app_under_cursor absent on the current packet, the gate falls back to
    focus_window.app (existing 445f759 behavior, unchanged)."""
    session = GeminiLiveSession(model="m")
    session._latest_pointer_referent = "the Coffee heading"
    session._latest_pointer_referent_app = "Ghostty"

    same_focus_packet = _packet_with_apps("Ghostty", None)
    other_focus_packet = _packet_with_apps("Notion", None)

    assert session._current_pointer_referent(same_focus_packet) == "the Coffee heading"
    assert session._current_pointer_referent(other_focus_packet) is None
