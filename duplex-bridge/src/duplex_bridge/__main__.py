"""Command line entry point for duplex-bridge."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any

from duplex_bridge.actions import (
    BROWSER_TOOL_SPECS,
    TOOL_DECLARATIONS,
    CommandSafetyClassifier,
    DelegateAgent,
    DelegateAgentConfig,
    DelegateBrowser,
    GeminiComputerUsePolicy,
    MacOSComputer,
    TaskManager,
    click_pointer,
    compare_products,
    rewrite_function_async,
    run_computer_use_with_timeout,
)
from duplex_bridge.actions.chrome import playwright_navigator
from duplex_bridge.actions.computer import Action, ComputerUseResult, Policy
from duplex_bridge.actions.computer_policy import (
    GeminiVisionLoopPolicy,
    wrap_with_vision_loop_fallback,
)
from duplex_bridge.providers.gemini_live import GeminiLiveSession
from duplex_bridge.server import WebSocketContextServer
from duplex_bridge.worker import BackgroundWorker, ToolDispatcher, make_tool_response_forwarder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


# Live finding: a hosted computer_use run can dangle indefinitely with no wall-clock bound
# (observed: two Teams-click calls that never returned). Every run is wrapped in this.
_COMPUTER_USE_TIMEOUT_S = 120.0


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
            "headphones or push-to-talk). 'vpio' (in-process macOS hardware echo "
            "cancellation via PyObjC; capture + playback verified working on-device "
            "2026-07-03 — the working speakers-on path). 'native-vpio' (same VPIO engine "
            "in a native Swift helper — build with `just build-native`; robust fallback "
            "if in-process playback is silent on your setup). 'software-aec' (numpy NLMS "
            "echo cancellation; needs duplex-bridge[aec]). See docs/vpio-backend-status.md."
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
        "--delegate-safety",
        choices=("confirm", "auto"),
        default="confirm",
        help=(
            "Claude-Code automode analog for delegated tasks. 'confirm' (default) pauses "
            "for spoken approval on every non-allowlisted / destructive action. 'auto' runs "
            "benign/allowlisted actions autonomously; destructive actions (DEFAULT_CONFIRM_"
            "PATTERNS) ALWAYS still escalate to the user — never silently run."
        ),
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
    # Headless: this browser drives background research (browser_* tools), and Playwright's
    # bundled "Chrome for Testing" flashing a visible window mid-task was a live-usability
    # bug. When the delegate wants the user to SEE a page, it opens it in their real default
    # browser via `open <url>` (run_shell) instead of surfacing this research browser.
    delegate_browser = DelegateBrowser(headless=True)
    # Allowlist + voice confirmation (the safety model the user chose): allowlisted
    # commands run autonomously; everything else pauses the task and the assistant asks.
    safety_classifier = CommandSafetyClassifier()

    def _delegate_agent_factory(task_id: str) -> DelegateAgent:
        return DelegateAgent(
            config=DelegateAgentConfig(
                api_key_env=args.api_key_env,
                computer_model=args.computer_use_model,
                computer_max_steps=args.computer_use_max_steps,
                safety_mode=args.delegate_safety,
            ),
            tool_handlers=delegate_browser.handlers_for_task(task_id),
            extra_tools=BROWSER_TOOL_SPECS,
            desktop_mutex=desktop_mutex,
            classifier=safety_classifier,
        )

    task_manager = TaskManager(_delegate_agent_factory, on_task_end=delegate_browser.close_page)
    # A Live reconnect is amnesiac; re-prime the fresh session with tasks still running
    # or paused on a spoken confirmation (rides the next turn's annotation as resume=).
    session.set_resume_context_provider(task_manager.pending_note)

    async def _delegate_task(a: dict[str, Any]) -> dict[str, Any]:
        context = " | ".join(session.pointer_history_for_delegate())
        return await task_manager.run(str(a.get("goal") or ""), context=context)

    async def _check_tasks(a: dict[str, Any]) -> dict[str, Any]:
        return await task_manager.summarize()

    async def _confirm_task(a: dict[str, Any]) -> dict[str, Any]:
        return await task_manager.confirm(
            str(a.get("task_id") or ""),
            bool(a.get("approved")),
            approve_all=bool(a.get("approve_all", False)),
        )

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
    # Holds the SAME desktop mutex as delegate tasks — exactly one loop may drive the mouse —
    # and threads a stop flag so cancellation doesn't leave an orphan run clicking around.
    #
    # Live finding: two Teams-click computer_use calls never returned a final result — the
    # hosted policy dangled with no wall-clock timeout. run_computer_use_with_timeout (task
    # 4) guarantees a final ComputerUseResult either way. The hosted policy also carries a
    # server-side "Input blocked" classifier that rejects some legitimate goals;
    # wrap_with_vision_loop_fallback (shared with the delegate's own computer_use call site)
    # retries once via the plain-generateContent GeminiVisionLoopPolicy when that happens.
    # A fresh pointer referent, if any, is folded into the goal text as a grounding hint.
    def _pointer_target() -> tuple[str, float, float] | None:
        # Defensive getattr: test doubles for GeminiLiveSession need not implement this.
        accessor = getattr(session, "pointer_click_target", None)
        return accessor() if callable(accessor) else None

    def _pointer_hint() -> str:
        target = _pointer_target()
        if target is None:
            return ""
        referent, x, y = target
        return f"\nuser is pointing at: {referent} at ({x:.0f}, {y:.0f})"

    async def _computer_use_direct(a: dict[str, Any]) -> Any:
        goal = str(a.get("goal") or "") + _pointer_hint()
        stop = threading.Event()

        async def _run(policy: Policy) -> ComputerUseResult:
            return await asyncio.to_thread(
                run_computer_use_with_timeout,
                goal,
                MacOSComputer(),
                policy,
                args.computer_use_max_steps,
                _COMPUTER_USE_TIMEOUT_S,
                stop.is_set,
            )

        async with desktop_mutex:
            try:
                result, used_fallback = await wrap_with_vision_loop_fallback(
                    _run,
                    lambda: _make_computer_policy(args.computer_use_model, args.api_key_env),
                    lambda: GeminiVisionLoopPolicy(
                        model=args.computer_use_model, api_key_env=args.api_key_env
                    ),
                )
            except asyncio.CancelledError:
                stop.set()
                raise
        payload: dict[str, Any] = {
            "status": "ok" if result.done else "incomplete",
            "steps": result.steps,
            "note": result.final_note,
        }
        if used_fallback:
            payload["via"] = "vision_loop_fallback"
        return payload

    # Live-fix (4c): deterministic click on what the user is pointing at — no screenshot
    # round-trip, no vision-model call. Only fires when a fresh referent exists; otherwise
    # returns an explanatory error so the model falls back to computer_use.
    async def _click_pointer_direct(_a: dict[str, Any]) -> dict[str, Any]:
        target = _pointer_target()
        if target is None:
            return {
                "status": "error",
                "error": (
                    "no fresh pointer referent — the user hasn't settled on anything "
                    "recently; use computer_use instead"
                ),
            }
        referent, x, y = target
        async with desktop_mutex:
            note = await asyncio.to_thread(click_pointer, MacOSComputer(), x, y, referent)
        return {"status": "ok", "note": note}

    tool_dispatcher.register("computer_use", _computer_use_direct)
    tool_dispatcher.register("click_pointer", _click_pointer_direct)
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
