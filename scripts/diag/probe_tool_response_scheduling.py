"""Headless probe: does the Live API honor NON_BLOCKING tools + two-stage FunctionResponses?

Phase-0 gate for the tool-result plumbing design (goofy-sleeping-sutton plan). The SDK types
declare ``FunctionDeclaration.behavior=NON_BLOCKING`` and ``FunctionResponse.scheduling``
/ ``will_continue`` for BidiGenerateContent, letting a long tool ACK immediately
("started", SILENT, will_continue=True) and deliver its real result minutes later — the
native ack-then-complete mechanic. Server semantics may differ from client types, so this
drives a real Live session:

    1. declare ``slow_op`` with behavior=NON_BLOCKING and ask the model to call it
    2. on the tool_call, immediately send FunctionResponse(scheduling=SILENT, will_continue=True)
    3. sleep ``--work-s`` seconds (the pretend tool work)
    4. send the final FunctionResponse(scheduling=WHEN_IDLE, will_continue=False)
    5. report: did the session survive, and did the model speak the completion?

PASS → Phase 1 wires the native two-response mechanic. FAIL → fall back to a single
blocking "started" response + completion injected via the text-annotation channel.

Usage:
    uv run python scripts/diag/probe_tool_response_scheduling.py
    uv run python scripts/diag/probe_tool_response_scheduling.py --work-s 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "bench"))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from measure_ttfb import load_api_key  # noqa: E402

DEFAULT_MODEL = "models/gemini-3.1-flash-live-preview"
MAGIC_RESULT = "the crystal contains 42 blue marbles"


async def probe(model: str, work_s: float, timeout_s: float) -> int:
    client = genai.Client(api_key=load_api_key())
    # Native-audio live models reject TEXT modality; assert on the server-side output
    # transcription instead (same AUDIO+transcription path the deictic eval uses).
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        tools=[
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="slow_op",
                        description=(
                            "Counts the marbles in the crystal. Slow background operation; "
                            "you will be notified when it completes."
                        ),
                        behavior=types.Behavior.NON_BLOCKING,
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={"subject": types.Schema(type=types.Type.STRING)},
                        ),
                    )
                ]
            )
        ],
    )
    got_call_id: str | None = None
    texts: list[str] = []
    survived = True
    followup_answered = False

    async with client.aio.live.connect(model=model, config=config) as session:
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[
                    types.Part(
                        text="Call slow_op to count the marbles in the crystal, then tell "
                        "me the result when it arrives."
                    )
                ],
            )
        )
        deadline = asyncio.get_running_loop().time() + timeout_s
        try:
            while asyncio.get_running_loop().time() < deadline:
                remaining = deadline - asyncio.get_running_loop().time()
                async for message in session.receive():
                    tool_call = getattr(message, "tool_call", None)
                    if tool_call is not None and got_call_id is None:
                        fc = tool_call.function_calls[0]
                        got_call_id = fc.id
                        print(f"[probe] got tool_call id={fc.id!r} name={fc.name!r}")
                        await session.send_tool_response(
                            function_responses=types.FunctionResponse(
                                id=fc.id,
                                name=fc.name,
                                response={"status": "started"},
                                scheduling=types.FunctionResponseScheduling.SILENT,
                                will_continue=True,
                            )
                        )
                        print(
                            f"[probe] sent SILENT will_continue=True ack; working {work_s:.1f}s ..."
                        )

                        async def _finish(call_id: str, call_name: str) -> None:
                            await asyncio.sleep(work_s)
                            await session.send_tool_response(
                                function_responses=types.FunctionResponse(
                                    id=call_id,
                                    name=call_name,
                                    response={"status": "done", "output": MAGIC_RESULT},
                                    scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
                                    will_continue=False,
                                )
                            )
                            print("[probe] sent final WHEN_IDLE will_continue=False response")

                        asyncio.ensure_future(_finish(fc.id or "", fc.name or "slow_op"))
                    server_content = getattr(message, "server_content", None)
                    if server_content is not None:
                        transcription = getattr(server_content, "output_transcription", None)
                        if transcription is not None and transcription.text:
                            texts.append(transcription.text)
                    # Once the completion was spoken, immediately probe session health with a
                    # follow-up turn — distinguishes "two-stage responses break the session"
                    # from "idle probe connection got reaped".
                    if "42" in " ".join(texts) and not followup_answered:
                        followup_answered = True
                        texts.append(" ||FOLLOWUP|| ")
                        await session.send_client_content(
                            turns=types.Content(
                                role="user",
                                parts=[types.Part(text="Say the word 'peach' and nothing else.")],
                            )
                        )
                    if "peach" in " ".join(texts).lower():
                        raise TimeoutError  # follow-up answered — session healthy; stop here
                if remaining <= 0:
                    break
        except TimeoutError:
            pass
        except Exception as exc:  # session died — the failure mode we're probing for
            survived = False
            print(f"[probe] session error: {type(exc).__name__}: {exc}")

    full_text = " ".join(texts)
    spoke_completion = "42" in full_text
    followup_ok = "peach" in full_text.lower()
    print(
        f"\n[probe] survived={survived} tool_called={got_call_id is not None} "
        f"spoke_completion={spoke_completion} followup_ok={followup_ok}"
    )
    print(f"[probe] model text: {full_text[:400]!r}")
    verdict = got_call_id is not None and spoke_completion and followup_ok
    outcome = (
        "PASS — native NON_BLOCKING ack-then-complete works and the session stays usable"
        if verdict
        else "FAIL — use blocking response + text-channel completion fallback"
    )
    print(f"[probe] VERDICT: {outcome}")
    return 0 if verdict else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--work-s", type=float, default=3.0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    return asyncio.run(probe(args.model, args.work_s, args.timeout_s))


if __name__ == "__main__":
    raise SystemExit(main())
