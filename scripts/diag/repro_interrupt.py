"""Headless repro: does a realtime visual send interrupt Gemini Live generation?

Reproduces the duplex-bridge barge-in flood WITHOUT a mic, speech, the pointer
agent, the VPIO helper, or the WebSocket server. It drives GeminiLiveSession
directly: triggers a model response with one synthetic audio turn, then — while
the model is still talking — fires a burst of visual-context sends and counts how
many trip ``server_content.interrupted`` (the barge-in flag).

Isolates the cause by channel (the test the live-loop can't cleanly run):
    --mode text    only send_realtime_input(text=...)   (sent every packet today)
    --mode image   only send_realtime_input(video=...)  (the tile; blessed channel)
    --mode both    both, exactly like send_visual_context does now

Default runs text then image and prints a comparison so you can see which one
interrupts. The only external dependency is a live Gemini connection (key read
from .env, same as measure_ttfb). No audio backend, no macOS permissions.

Usage:
    uv run python scripts/diag/repro_interrupt.py
    uv run python scripts/diag/repro_interrupt.py --mode text --sends 10 --hz 10
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/diag/ -> repo root
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "bench"))

from aimer_core import (  # noqa: E402
    ContextPacket,
    CursorPosition,
    FocusWindow,
    HoverRegion,
    SemanticContext,
)
from duplex_bridge.providers.gemini_live import GeminiLiveSession  # noqa: E402
from google.genai import types  # noqa: E402
from measure_ttfb import chunk_pcm, generate_speech_pcm, load_api_key  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("repro_interrupt")

DEFAULT_MODEL = "models/gemini-3.1-flash-live-preview"
# A prompt that elicits a multi-second response, so the model is still generating
# when we inject the visual sends.
TRIGGER_PHRASE = "Please tell me a slow, detailed story about the ocean, taking your time."
RESPONSE_TIMEOUT_S = 12.0

# Validation A/B (text-at-turn-start): a clear spoken question that reliably elicits a
# brief response, plus a realistic annotation injected right after activity_start.
VALIDATE_QUESTION = "Hello, please respond with one short sentence."
VALIDATE_ANNOTATION = (
    "[context] app=Safari title=Example Page cursor=(820,440) selected=the quick brown fox"
)


def generate_test_jpeg(size: int = 256) -> bytes:
    """Produce a real, decodable JPEG via ffmpeg (256x256 matches the real tile)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "tile.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=gray:s={size}x{size}",
                "-frames:v",
                "1",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        return out.read_bytes()


async def _fire_visual(session: GeminiLiveSession, mode: str, i: int, jpeg: bytes) -> None:
    """Fire one visual send on the raw realtime channel.

    Goes through session._session directly (not send_visual_context) so text and
    video can be isolated regardless of how send_visual_context is currently wired.
    """
    raw = session._session  # private: only valid while connected
    if mode in ("image", "both"):
        await raw.send_realtime_input(video=types.Blob(mime_type="image/jpeg", data=jpeg))
    if mode in ("text", "both"):
        text = f"[context] cursor=({(i * 37) % 1920},{(i * 17) % 1080})"
        await raw.send_realtime_input(text=text)


async def run_probe(mode: str, model: str, sends: int, hz: float, jpeg: bytes) -> dict[str, object]:
    """Trigger one turn, inject `sends` visual packets while it answers, count interrupts."""
    interrupts = 0
    audio_chunks = 0
    first_audio = asyncio.Event()

    def on_audio(_data: bytes) -> None:
        nonlocal audio_chunks
        audio_chunks += 1
        first_audio.set()

    def on_interrupt() -> None:
        nonlocal interrupts
        interrupts += 1

    session = GeminiLiveSession(model=model, manual_vad=True, thinking_level="minimal")
    session.on_audio_out(on_audio)
    session.on_interrupt(on_interrupt)
    await session.open()
    try:
        frames = chunk_pcm(generate_speech_pcm(TRIGGER_PHRASE))
        await session.send_activity_start()
        for frame in frames:
            await session.send_audio(frame)
            await asyncio.sleep(0.1)
        await session.send_activity_end()

        # Only probe once the model is actually generating (interrupted means
        # "a client message cut off the CURRENT generation").
        try:
            await asyncio.wait_for(first_audio.wait(), timeout=RESPONSE_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning("[%s] no audio within %.0fs — inconclusive", mode, RESPONSE_TIMEOUT_S)
            return {"mode": mode, "ok": False, "interrupts": 0, "audio_chunks": audio_chunks}

        interrupts_at_start = interrupts
        period = 1.0 / hz
        for i in range(sends):
            await _fire_visual(session, mode, i, jpeg)
            await asyncio.sleep(period)
        await asyncio.sleep(1.5)  # let trailing interrupted flags surface

        return {
            "mode": mode,
            "ok": True,
            "sends": sends,
            "interrupts": interrupts - interrupts_at_start,
            "audio_chunks": audio_chunks,
        }
    finally:
        await session.close()


def _verdict(r: dict[str, object]) -> str:
    if not r.get("ok"):
        return "NO RESPONSE (inconclusive)"
    n = int(r["interrupts"])  # type: ignore[call-overload]
    return f"INTERRUPTS ✗ ({n})" if n else "no interrupts ✓"


async def run_validation_turn(
    model: str, with_text: bool, annotation: str, frames: list[bytes]
) -> dict[str, object]:
    """Drive one manual-VAD turn, optionally injecting a text annotation right after
    activity_start, and measure whether the turn responds cleanly, wedges, or fires a
    premature response to the bare annotation (before the spoken question)."""
    interrupts = 0
    audio_chunks = 0
    first_audio_at: list[float | None] = [None]
    activity_end_at: list[float | None] = [None]

    def on_audio(_data: bytes) -> None:
        nonlocal audio_chunks
        audio_chunks += 1
        if first_audio_at[0] is None:
            first_audio_at[0] = time.perf_counter()

    def on_interrupt() -> None:
        nonlocal interrupts
        interrupts += 1

    session = GeminiLiveSession(model=model, manual_vad=True, thinking_level="minimal")
    session.on_audio_out(on_audio)
    session.on_interrupt(on_interrupt)
    await session.open()
    try:
        await session.send_activity_start()
        if with_text:
            await session._session.send_realtime_input(text=annotation)
        # Pause with only the annotation sent: a healthy manual-VAD turn must NOT respond
        # yet (it waits for activity_end). Any audio here is a premature response.
        await asyncio.sleep(1.5)
        premature = first_audio_at[0] is not None

        for frame in frames:
            await session.send_audio(frame)
            await asyncio.sleep(0.1)
        activity_end_at[0] = time.perf_counter()
        await session.send_activity_end()

        if not premature:
            for _ in range(int(RESPONSE_TIMEOUT_S / 0.1)):
                if first_audio_at[0] is not None:
                    break
                await asyncio.sleep(0.1)
        await asyncio.sleep(1.0)  # let the response's chunks accrue

        responded = first_audio_at[0] is not None
        delta_ms: float | None = None
        if responded and not premature and activity_end_at[0] is not None:
            delta_ms = (first_audio_at[0] - activity_end_at[0]) * 1000.0
        return {
            "with_text": with_text,
            "responded": responded,
            "premature": premature,
            "delta_ms": delta_ms,
            "interrupts": interrupts,
            "audio_chunks": audio_chunks,
        }
    finally:
        await session.close()


def _validation_verdict(control: dict[str, object], text_run: dict[str, object]) -> str:
    if not control["responded"]:
        return "INCONCLUSIVE — control turn produced no response; rerun"
    if text_run["premature"]:
        return "UNSAFE ✗ — annotation triggered a premature response before the question"
    if not text_run["responded"]:
        return "UNSAFE ✗ — turn wedged: no response after activity_end"
    if int(text_run["interrupts"]) > 0:  # type: ignore[call-overload]
        return f"SUSPECT ✗ — {text_run['interrupts']} interrupt(s) during the turn"
    return "SAFE ✓ — annotation rode the turn; one clean response after activity_end"


def _print_validation(control: dict[str, object], text_run: dict[str, object]) -> None:
    print("\n" + "=" * 60)
    for label, r in (("control", control), ("text-prefix", text_run)):
        dms = f"{r['delta_ms']:.0f}ms" if r["delta_ms"] is not None else "n/a"
        print(
            f"  {label:<12} resp={r['responded']!s:<5} prem={r['premature']!s:<5} "
            f"delta={dms:<8} intr={r['interrupts']} chunks={r['audio_chunks']}"
        )
    print("=" * 60)
    print(f"VERDICT: {_validation_verdict(control, text_run)}")


async def run_fix_smoke(model: str, jpeg: bytes) -> dict[str, object]:
    """End-to-end smoke of the fix: stream send_visual_context at 10 Hz while running a real
    manual-VAD turn (send_activity_start injects the cached tile + annotation, then audio,
    then activity_end). Confirms the visual stream no longer interrupts the turn."""
    interrupts = 0
    audio_chunks = 0
    first_audio = asyncio.Event()

    def on_audio(_data: bytes) -> None:
        nonlocal audio_chunks
        audio_chunks += 1
        first_audio.set()

    def on_interrupt() -> None:
        nonlocal interrupts
        interrupts += 1

    session = GeminiLiveSession(model=model, manual_vad=True, thinking_level="minimal")
    session.on_audio_out(on_audio)
    session.on_interrupt(on_interrupt)
    await session.open()

    stop = asyncio.Event()
    tile_b64 = base64.b64encode(jpeg).decode()

    async def stream_visual() -> None:
        i = 0
        while not stop.is_set():
            await session.send_visual_context(
                ContextPacket(
                    cursor=CursorPosition(x=100 + i, y=200),
                    focus_window=FocusWindow(app="Safari", title="Example Page"),
                    hover_region=HoverRegion(tile_b64=tile_b64),
                    semantic=SemanticContext(selected_text="the quick brown fox"),
                )
            )
            i += 1
            await asyncio.sleep(0.1)  # 10 Hz, like the pointer agent

    streamer = asyncio.create_task(stream_visual())
    try:
        await asyncio.sleep(1.0)  # visual stream runs (caches in manual VAD)
        frames = chunk_pcm(generate_speech_pcm(VALIDATE_QUESTION))
        await session.send_activity_start()
        for frame in frames:
            await session.send_audio(frame)
            await asyncio.sleep(0.1)
        await session.send_activity_end()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(first_audio.wait(), timeout=RESPONSE_TIMEOUT_S)
        await asyncio.sleep(1.5)
        return {
            "interrupts": interrupts,
            "responded": first_audio.is_set(),
            "audio_chunks": audio_chunks,
        }
    finally:
        stop.set()
        await streamer
        await session.close()


async def run_text_during_speech(
    model: str, n_texts: int, annotation: str, frames: list[bytes]
) -> dict[str, object]:
    """Stream the text annotation multiple times DURING the speech window (interleaved with
    audio, before activity_end) to test whether continuous text-during-speech is safe."""
    interrupts = 0
    audio_chunks = 0
    first_audio_at: list[float | None] = [None]

    def on_audio(_data: bytes) -> None:
        nonlocal audio_chunks
        audio_chunks += 1
        if first_audio_at[0] is None:
            first_audio_at[0] = time.perf_counter()

    def on_interrupt() -> None:
        nonlocal interrupts
        interrupts += 1

    session = GeminiLiveSession(model=model, manual_vad=True, thinking_level="minimal")
    session.on_audio_out(on_audio)
    session.on_interrupt(on_interrupt)
    await session.open()
    try:
        await session.send_activity_start()
        every = max(1, len(frames) // n_texts)
        sent_texts = 0
        for i, frame in enumerate(frames):
            if i % every == 0:
                await session._session.send_realtime_input(text=f"{annotation} sel#{i}")
                sent_texts += 1
            await session.send_audio(frame)
            await asyncio.sleep(0.1)
        # Did the model respond BEFORE we ended the turn? That means a text send pre-triggered
        # generation (bad) — with manual VAD it must wait for activity_end.
        premature = first_audio_at[0] is not None
        await session.send_activity_end()
        if not premature:
            for _ in range(int(RESPONSE_TIMEOUT_S / 0.1)):
                if first_audio_at[0] is not None:
                    break
                await asyncio.sleep(0.1)
        await asyncio.sleep(1.0)
        return {
            "texts_during_speech": sent_texts,
            "responded": first_audio_at[0] is not None,
            "premature": premature,
            "interrupts": interrupts,
            "audio_chunks": audio_chunks,
        }
    finally:
        await session.close()


async def _main(
    modes: list[str],
    model: str,
    sends: int,
    hz: float,
    validate_text: bool,
    text_stream: bool,
    smoke: bool,
) -> int:
    key = load_api_key()
    if not key:
        print("ERROR: GEMINI_API_KEY not found in .env", file=sys.stderr)
        return 1
    os.environ["GEMINI_API_KEY"] = key

    if validate_text:
        print(f"\n=== validate text-at-turn-start (model={model}) ===")
        frames = chunk_pcm(generate_speech_pcm(VALIDATE_QUESTION))
        control = await run_validation_turn(model, False, "", frames)
        await asyncio.sleep(3.0)
        text_run = await run_validation_turn(model, True, VALIDATE_ANNOTATION, frames)
        _print_validation(control, text_run)
        return 0

    if text_stream:
        print(f"\n=== validate continuous text DURING speech (model={model}) ===")
        frames = chunk_pcm(generate_speech_pcm(VALIDATE_QUESTION))
        r = await run_text_during_speech(model, 4, VALIDATE_ANNOTATION, frames)
        ok = r["interrupts"] == 0 and not r["premature"] and r["responded"]
        verdict = "SAFE ✓ — repeated text during speech did not interrupt or pre-trigger"
        if not ok:
            verdict = "UNSAFE ✗ — text during speech broke the turn (interrupt/premature/wedge)"
        print(
            f"\ntexts={r['texts_during_speech']} responded={r['responded']} "
            f"premature={r['premature']} interrupts={r['interrupts']} chunks={r['audio_chunks']}"
        )
        print(f"VERDICT: {verdict}")
        return 0

    if smoke:
        print(f"\n=== end-to-end fix smoke (model={model}) ===")
        r = await run_fix_smoke(model, generate_test_jpeg())
        ok = r["interrupts"] == 0 and r["responded"]
        verdict = "PASS ✓" if ok else "FAIL ✗"
        print(
            f"\n{verdict}  interrupts={r['interrupts']} "
            f"responded={r['responded']} audio_chunks={r['audio_chunks']}"
        )
        print("(manual VAD caches the 10 Hz stream; context injects at turn start.)")
        return 0

    jpeg = generate_test_jpeg() if any(m in ("image", "both") for m in modes) else b""
    results: list[dict[str, object]] = []
    for mode in modes:
        print(f"\n=== mode={mode}: trigger a turn, then fire {sends} sends at {hz:.0f} Hz ===")
        results.append(await run_probe(mode, model, sends, hz, jpeg))
        await asyncio.sleep(3.0)  # settle between probes

    print("\n" + "=" * 60)
    print(f"{'mode':<8}{'sends':>7}{'interrupts':>12}{'audio_chunks':>14}   verdict")
    print("-" * 60)
    for r in results:
        print(
            f"{str(r['mode']):<8}{str(r.get('sends', '-')):>7}"
            f"{str(r['interrupts']):>12}{str(r['audio_chunks']):>14}   {_verdict(r)}"
        )
    print("=" * 60)
    print("\ntext interrupts, image doesn't -> relocate the text; keep the tile on realtime-video.")
    print("both interrupt                 -> gate the image to the turn too.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=("text", "image", "both", "all"),
        default="all",
        help="which realtime channel to fire (default: all = text then image)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--sends", type=int, default=8, help="visual sends fired during the response")
    p.add_argument("--hz", type=float, default=10.0, help="send rate (default 10 = real tile rate)")
    p.add_argument(
        "--validate-text",
        action="store_true",
        help="A/B test: does a text annotation injected at turn start break the turn?",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="end-to-end: stream visual context at 10 Hz through a real manual-VAD turn",
    )
    p.add_argument(
        "--validate-text-stream",
        action="store_true",
        help="does the text annotation sent repeatedly DURING speech break the turn?",
    )
    args = p.parse_args()
    modes = ["text", "image"] if args.mode == "all" else [args.mode]
    return asyncio.run(
        _main(
            modes,
            args.model,
            args.sends,
            args.hz,
            args.validate_text,
            args.validate_text_stream,
            args.smoke,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
