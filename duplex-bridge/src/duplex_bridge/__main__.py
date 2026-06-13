"""Command line entry point for duplex-bridge."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from duplex_bridge.providers.gemini_live import GeminiLiveSession
from duplex_bridge.server import WebSocketContextServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="duplex-bridge",
        description="Aimer bridge from visual context packets to duplex model sessions.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="WebSocket server host. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port. Defaults to 8765.",
    )
    parser.add_argument(
        "--gemini-model",
        default="models/gemini-3.1-flash-live-preview",
        help=(
            "Gemini Live model name. Defaults to models/gemini-3.1-flash-live-preview "
            "(native-audio). Use models/gemini-2.0-flash-live-001 for the stable "
            "non-native-audio variant."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable for Gemini API key. Defaults to GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable local microphone capture and speaker playback.",
    )
    parser.add_argument(
        "--audio-backend",
        choices=("sounddevice", "software-aec", "vpio", "native-vpio"),
        default="sounddevice",
        help=(
            "Audio I/O backend. 'sounddevice' (default, no echo cancellation — use "
            "headphones or push-to-talk). 'software-aec' (numpy NLMS echo cancellation; "
            "needs duplex-bridge[aec]). 'native-vpio' (recommended speakers-on path: "
            "macOS hardware echo cancellation via a native Swift helper — build it with "
            "`just build-native`). 'vpio' (experimental, capture-only: PyObjC VPIO cancels "
            "the mic but playback is silent — see docs/vpio-backend-status.md)."
        ),
    )
    parser.add_argument(
        "--audio-activity-rms-threshold",
        type=float,
        default=300.0,
        help=(
            "RMS threshold for marking speech-like audio activity in diagnostics. "
            "Defaults to 300.0."
        ),
    )
    parser.add_argument(
        "--vad-silence-ms",
        type=int,
        default=None,
        help=(
            "Optional Gemini VAD silence duration in milliseconds. "
            "Unset preserves the baseline Gemini config."
        ),
    )
    parser.add_argument(
        "--vad-start-sensitivity",
        choices=("high", "low"),
        default=None,
        help=(
            "Optional Gemini VAD start-of-speech sensitivity. "
            "Unset preserves the baseline Gemini config."
        ),
    )
    parser.add_argument(
        "--turn-coverage",
        choices=("all", "activity_only"),
        default=None,
        help=(
            "Optional Gemini realtime input turn coverage. "
            "Unset preserves the baseline Gemini config."
        ),
    )
    parser.add_argument(
        "--manual-vad",
        action="store_true",
        help=(
            "Disable Gemini's server VAD and detect end-of-turn locally (activity_end "
            "after --end-of-turn-silence-ms of silence). Cuts end-of-speech→response "
            "latency from ~1340 ms to the model+network floor (~710 ms) by removing the "
            "server's ~630 ms silence wait."
        ),
    )
    parser.add_argument(
        "--end-of-turn-silence-ms",
        type=int,
        default=400,
        help=(
            "Trailing silence before client-side end-of-turn fires (manual VAD only). "
            "Lower is snappier but risks cutting off mid-sentence pauses. Defaults to 400."
        ),
    )
    parser.add_argument(
        "--onset-speech-ms",
        type=int,
        default=250,
        help=(
            "Sustained above-threshold audio required before a turn opens (manual VAD "
            "only), debouncing phantom turns from transient noise (clicks, keypresses). "
            "Onset frames are buffered so speech isn't clipped. 0 opens on the first "
            "voiced frame. Defaults to 250."
        ),
    )
    parser.add_argument(
        "--thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help="Optional Gemini thinking budget. 'minimal' minimizes first-audio latency.",
    )
    parser.add_argument(
        "--push-to-talk",
        action="store_true",
        help=(
            "Hold --ptt-key to talk; releasing ends the turn immediately. Implies "
            "manual VAD and yields the true model+network latency floor (~710 ms) with "
            "no end-of-turn silence wait. Requires duplex-bridge[ptt] (pynput)."
        ),
    )
    parser.add_argument(
        "--ptt-key",
        default="cmd_r",
        help="Push-to-talk key name (pynput Key name like cmd_r/shift_r, or a character). "
        "Defaults to cmd_r (right Command).",
    )
    parser.add_argument(
        "--escalate-full-frame",
        action="store_true",
        help=(
            "Force-send a cached downscaled full-display frame at the start of each turn "
            "(video channel — never interrupts) to give the model full layout context for "
            "relational-deixis utterances like 'compare these two windows'. Off by default; "
            "requires the pointer-agent to be capturing full frames (--full-frame)."
        ),
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    """Async main entry point."""
    # Create Gemini Live session
    # Push-to-talk drives turns from the keyboard, which requires manual VAD.
    manual_vad = args.manual_vad or args.push_to_talk
    session = GeminiLiveSession(
        model=args.gemini_model,
        api_key_env=args.api_key_env,
        audio_activity_rms_threshold=args.audio_activity_rms_threshold,
        vad_silence_ms=args.vad_silence_ms,
        vad_start_sensitivity=args.vad_start_sensitivity,
        turn_coverage=args.turn_coverage,
        manual_vad=manual_vad,
        thinking_level=args.thinking_level,
        escalate_with_full_frame=args.escalate_full_frame,
    )

    # Create WebSocket server
    server = WebSocketContextServer(
        session=session,
        host=args.host,
        port=args.port,
        path="/context",
    )
    mic_capture = None
    ptt_controller = None

    try:
        # Open session and start server
        await session.open()
        audio_enabled = not args.no_audio
        audio_started = False

        if audio_enabled:
            from duplex_bridge.audio_backends import make_backend
            from duplex_bridge.audio_input import MicCapture, MicCaptureConfig

            mic_config = MicCaptureConfig(
                manual_vad=manual_vad,
                activity_rms_threshold=args.audio_activity_rms_threshold,
                end_of_turn_silence_ms=args.end_of_turn_silence_ms,
                onset_speech_ms=args.onset_speech_ms,
                push_to_talk=args.push_to_talk,
            )
            # The backend owns both mic capture and speaker playback (one backend
            # owns both streams — required for the VPIO echo-cancellation backend).
            backend = make_backend(args.audio_backend, mic_config)
            mic_capture = MicCapture(mic_config, backend=backend)
            audio_started = await mic_capture.start(session, asyncio.get_running_loop())

            if args.push_to_talk and audio_started:
                from duplex_bridge.push_to_talk import PushToTalkController

                ptt_controller = PushToTalkController(
                    on_change=mic_capture.set_talking,
                    loop=asyncio.get_running_loop(),
                    key_name=args.ptt_key,
                )
                if not ptt_controller.start():
                    print(
                        "[duplex-bridge] push-to-talk unavailable "
                        "(install duplex-bridge[ptt] and grant Input Monitoring); "
                        "no audio turns will be sent"
                    )
                    ptt_controller = None

        await server.start()

        # Print banner after server is bound
        audio_status = (
            "disabled" if args.no_audio else ("enabled" if audio_started else "unavailable")
        )
        print(
            f"[duplex-bridge] listening on ws://{args.host}:{server.port}/context, "
            f"gemini model={args.gemini_model}, audio={audio_status}, "
            f"backend={args.audio_backend}"
        )
        if audio_started and args.audio_backend == "sounddevice":
            print("[duplex-bridge] use headphones to avoid speaker-to-mic feedback")
        if ptt_controller is not None:
            print(f"[duplex-bridge] push-to-talk: hold {args.ptt_key} to speak")

        # Run until Ctrl+C
        await asyncio.Event().wait()

    except KeyboardInterrupt:
        print("\n[duplex-bridge] shutting down")
    finally:
        # Clean shutdown
        if ptt_controller is not None:
            ptt_controller.stop()
        if mic_capture is not None:
            await mic_capture.stop()
        await server.stop()
        await session.close()

    return 0


def _load_dotenv_key(var: str) -> None:
    """Populate ``var`` from a .env file if it isn't already in the environment.

    Lets the bridge run with the key in .env (as the measurement scripts do) without
    requiring an explicit `export`. Searches the cwd and the workspace root.
    """
    if os.environ.get(var):
        return
    candidates = [Path.cwd() / ".env", *(p / ".env" for p in Path(__file__).resolve().parents)]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                if key.strip() == var and value.strip():
                    os.environ[var] = value.strip()
                    return
        return


def main() -> int:
    args = build_parser().parse_args()
    _load_dotenv_key(args.api_key_env)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
