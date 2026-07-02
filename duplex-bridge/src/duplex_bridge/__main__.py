"""Command line entry point for duplex-bridge."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from duplex_bridge.actions import (
    BROWSER_TOOL_SPECS,
    TOOL_DECLARATIONS,
    DelegateAgent,
    DelegateAgentConfig,
    DelegateBrowser,
    GeminiComputerUsePolicy,
    MacOSComputer,
    TaskManager,
    compare_products,
    rewrite_function_async,
    run_computer_use,
)
from duplex_bridge.actions.chrome import playwright_navigator
from duplex_bridge.actions.computer import Action, Policy
from duplex_bridge.providers.gemini_live import GeminiLiveSession
from duplex_bridge.server import WebSocketContextServer
from duplex_bridge.worker import BackgroundWorker, ToolDispatcher, make_tool_response_forwarder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def _unconfigured_computer_policy(goal: str, shot: bytes, history: list[Action]) -> Action:
    """Placeholder computer-use policy, used only when no Gemini API key is available."""
    return Action(
        "done", note="computer-use needs a vision policy — see docs/week7b-computer-use.md"
    )


def _make_computer_policy(model: str, api_key_env: str) -> Policy:
    """Fresh policy per computer_use run — GeminiComputerUsePolicy is stateful per goal."""
    if not os.environ.get(api_key_env):
        return _unconfigured_computer_policy
    return GeminiComputerUsePolicy(model=model, api_key_env=api_key_env)


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
        "--deixis-model",
        default="gemini-flash-lite-latest",
        help=(
            "Vision model for the decoupled pointer-referent resolver (resolve-on-settle, "
            "off the hot path). Defaults to gemini-flash-lite-latest."
        ),
    )
    parser.add_argument(
        "--no-deixis-resolver",
        action="store_true",
        help="Disable the decoupled deixis resolver (live model reads raw tiles only).",
    )
    parser.add_argument(
        "--computer-use-model",
        default="gemini-3.5-flash",
        help=(
            "Model for the computer_use vision policy (Gemini Interactions API with the "
            "built-in computer_use tool). Defaults to gemini-3.5-flash."
        ),
    )
    parser.add_argument(
        "--computer-use-max-steps",
        type=int,
        default=24,
        help="Max perceive->decide->act ticks per computer_use goal. Defaults to 24.",
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
    # Decoupled deixis: the resolver reads the pointed-at element off the hot path on
    # cursor settle, and the session injects its referent as the pointer= annotation.
    deixis_resolver = None
    if not args.no_deixis_resolver and os.environ.get(args.api_key_env):
        from duplex_bridge.deixis import PointerReferentResolver

        deixis_resolver = PointerReferentResolver(
            model=args.deixis_model, api_key_env=args.api_key_env
        )
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
        tools=TOOL_DECLARATIONS,
        deixis_resolver=deixis_resolver,
    )

    # Week 6: model-emitted tool calls run OFF the audio hot path. The session invokes the
    # on_tool_call callback from its recv loop (which also dispatches audio); the dispatcher
    # hands each call to the BackgroundWorker and returns immediately, so a long tool call
    # (web / code edit / file I/O / reasoning) never stalls the ~200 ms audio tick.
    # Week 9: results flow BACK — every finished job with a function-call id becomes a
    # FunctionResponse (WHEN_IDLE, so completions never barge into ongoing speech), and
    # long tools get an immediate silent "started" ack so the model keeps conversing.
    tool_worker = BackgroundWorker(on_result=make_tool_response_forwarder(session))

    def _ack_started(name: str, call_id: str) -> None:
        task = asyncio.ensure_future(
            session.send_tool_response(
                name=name, call_id=call_id, response={"status": "started"}, final=False
            )
        )
        task.add_done_callback(lambda t: t.cancelled() or t.exception())

    tool_dispatcher = ToolDispatcher(
        tool_worker,
        immediate_ack=("computer_use", "delegate_task", "confirm_task"),
        on_ack=_ack_started,
    )
    session.on_tool_call_cancellation(tool_dispatcher.cancel)

    # Week 9: the delegate seam. One shared desktop mutex (only one task may drive the
    # mouse/keyboard at a time; shell/AppleScript/browser calls parallelize freely), one
    # persistent browser (an isolated page per task), one DelegateAgent per delegated goal.
    desktop_mutex = asyncio.Lock()
    delegate_browser = DelegateBrowser(headless=False)  # headed: the user watches it work

    def _delegate_agent_factory(task_id: str) -> DelegateAgent:
        return DelegateAgent(
            config=DelegateAgentConfig(
                api_key_env=args.api_key_env,
                computer_model=args.computer_use_model,
                computer_max_steps=args.computer_use_max_steps,
            ),
            tool_handlers=delegate_browser.handlers_for_task(task_id),
            extra_tools=BROWSER_TOOL_SPECS,
            desktop_mutex=desktop_mutex,
        )

    task_manager = TaskManager(_delegate_agent_factory, on_task_end=delegate_browser.close_page)

    async def _delegate_task(a: dict[str, Any]) -> dict[str, Any]:
        context = " | ".join(session.pointer_history_for_delegate())
        return await task_manager.run(str(a.get("goal") or ""), context=context)

    async def _check_tasks(a: dict[str, Any]) -> dict[str, Any]:
        return await task_manager.summarize()

    async def _confirm_task(a: dict[str, Any]) -> dict[str, Any]:
        return await task_manager.confirm(str(a.get("task_id") or ""), bool(a.get("approved")))

    tool_dispatcher.register("delegate_task", _delegate_task)
    tool_dispatcher.register("check_tasks", _check_tasks)
    tool_dispatcher.register("confirm_task", _confirm_task)
    _live_navigator = playwright_navigator(headless=False)  # headed so the user sees Chrome open
    tool_dispatcher.register(
        "compare_products",
        lambda a: compare_products(list(a.get("products", [])), navigate=_live_navigator),
    )
    tool_dispatcher.register(
        "rewrite_function_async",
        lambda a: rewrite_function_async(a["file"], a["function"], a.get("new_source")),
    )
    # General cross-application fallback: drive any app via screenshot + mouse + keyboard,
    # decided by the Gemini computer-use tool (desktop environment, safety decisions honored).
    tool_dispatcher.register(
        "computer_use",
        lambda a: run_computer_use(
            a["goal"],
            MacOSComputer(),
            _make_computer_policy(args.computer_use_model, args.api_key_env),
            max_steps=args.computer_use_max_steps,
        ),
    )
    session.on_tool_call(tool_dispatcher.dispatch)

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
        await tool_worker.aclose()
        await delegate_browser.aclose()

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
