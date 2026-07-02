"""GeminiComputerUsePolicy — deterministic tests over a fake Interactions client.

Covers the seam adaptation (composite actions -> primitive FIFO, function_result round-trips,
0-999 coordinate denormalization to points) and the safety-decision behavior, with no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from duplex_bridge.actions.computer import Action, ComputerUseExecutor, FakeComputer
from duplex_bridge.actions.computer_policy import GeminiComputerUsePolicy

SCREEN = (1440, 900)
PNG = b"\x89PNG\r\n"


def _call(name: str, call_id: str = "c1", **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", id=call_id, name=name, arguments=arguments)


def _interaction(interaction_id: str, *steps: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(id=interaction_id, steps=list(steps))


def _text_output(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="model_output", content=[SimpleNamespace(text=text)])


class _FakeClient:
    """Duck-typed genai.Client: pops canned interactions, records every request."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        outer = self

        class _Interactions:
            def create(self, **kwargs: Any) -> SimpleNamespace:
                outer.requests.append(kwargs)
                return outer.responses.pop(0)

        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self.interactions = _Interactions()


def _policy(responses: list[SimpleNamespace], **kwargs: Any) -> GeminiComputerUsePolicy:
    return GeminiComputerUsePolicy(client=_FakeClient(responses), screen_size=SCREEN, **kwargs)


def test_first_request_carries_goal_text_screenshot_and_desktop_tool() -> None:
    policy = _policy([_interaction("i1", _call("click", x=500, y=500))])
    policy("open mail", PNG, [])
    client: Any = policy._client
    (request,) = client.requests
    assert request["model"] == "gemini-3.5-flash"
    assert "previous_interaction_id" not in request
    assert request["input"][0] == {"type": "text", "text": "open mail"}
    assert request["input"][1]["type"] == "image"
    (tool,) = request["tools"]
    assert tool["type"] == "computer_use"
    assert tool["environment"] == "desktop"
    assert "drag_and_drop" in tool["excluded_predefined_functions"]


def test_click_coordinates_denormalize_to_screen_points() -> None:
    policy = _policy([_interaction("i1", _call("click", x=500, y=500, intent="center"))])
    action = policy("g", PNG, [])
    assert action == Action("click", x=720, y=450, note="center")


def test_composite_type_expands_into_fifo_without_extra_api_calls() -> None:
    steps = _call("type", x=100, y=200, text="hi", press_enter=True)
    policy = _policy([_interaction("i1", steps)])  # exactly one canned response
    first = policy("g", PNG, [])
    assert first.type == "click" and (first.x, first.y) == (144, 180)
    assert policy("g", PNG, []) == Action("type", text="hi")
    assert policy("g", PNG, []) == Action("key", keys=("return",))


def test_function_result_round_trip_carries_call_id_and_fresh_screenshot() -> None:
    policy = _policy(
        [
            _interaction("i1", _call("click", call_id="c9", x=0, y=0)),
            _interaction("i2", _text_output("done!")),
        ]
    )
    policy("g", PNG, [])
    final = policy("g", b"fresh-shot", [])
    assert final.type == "done" and "done!" in final.note
    client: Any = policy._client
    second = client.requests[1]
    assert second["previous_interaction_id"] == "i1"
    (result_step,) = second["input"]
    assert result_step["type"] == "function_result"
    assert result_step["call_id"] == "c9"
    assert result_step["name"] == "click"
    assert result_step["result"][-1]["type"] == "image"


def test_no_function_calls_means_done_with_model_text() -> None:
    policy = _policy([_interaction("i1", _text_output("Nothing to do"))])
    action = policy("g", PNG, [])
    assert action == Action("done", note="Nothing to do")


def test_blocked_safety_decision_stops_the_run() -> None:
    call = _call(
        "click", x=1, y=1, safety_decision={"decision": "blocked", "explanation": "payment page"}
    )
    policy = _policy([_interaction("i1", call)])
    action = policy("g", PNG, [])
    assert action.type == "done"
    assert "blocked" in action.note and "payment page" in action.note


def test_require_confirmation_stops_and_asks_the_user() -> None:
    call = _call(
        "click",
        x=1,
        y=1,
        safety_decision={"decision": "require_confirmation", "explanation": "sends an email"},
    )
    policy = _policy([_interaction("i1", call)])
    action = policy("g", PNG, [])
    assert action.type == "done"
    assert "confirmation" in action.note and "sends an email" in action.note


def test_auto_acknowledge_executes_and_flags_the_result() -> None:
    call = _call(
        "click",
        x=500,
        y=500,
        safety_decision={"decision": "require_confirmation", "explanation": "x"},
    )
    policy = _policy(
        [_interaction("i1", call), _interaction("i2", _text_output("ok"))],
        auto_acknowledge_safety=True,
    )
    assert policy("g", PNG, []).type == "click"
    policy("g", PNG, [])
    client: Any = policy._client
    (result_step,) = client.requests[1]["input"]
    assert '"safety_acknowledgement": "true"' in result_step["result"][0]["text"]


def test_unsupported_action_reports_error_back_instead_of_crashing() -> None:
    policy = _policy(
        [
            _interaction("i1", _call("teleport", call_id="c2")),
            _interaction("i2", _text_output("ok")),
        ]
    )
    noop = policy("g", PNG, [])
    assert noop.type == "screenshot"  # applied as a no-op; the loop stays alive
    policy("g", PNG, [])
    client: Any = policy._client
    (result_step,) = client.requests[1]["input"]
    assert result_step["is_error"] is True
    assert "teleport" in result_step["result"][0]["text"]


def test_scroll_direction_maps_to_quartz_sign_convention() -> None:
    down = _call("scroll", direction="down", magnitude_in_pixels=300)
    policy = _policy([_interaction("i1", down)])
    assert policy("g", PNG, []) == Action("scroll", dy=-300)
    policy = _policy([_interaction("i1", _call("scroll", direction="up"))])
    assert policy("g", PNG, []).dy > 0


def test_key_combination_string_splits_into_key_tuple() -> None:
    policy = _policy([_interaction("i1", _call("key_combination", keys="ctrl+c"))])
    assert policy("g", PNG, []) == Action("key", keys=("ctrl", "c"))
    policy = _policy([_interaction("i1", _call("press_key", key="Enter"))])
    assert policy("g", PNG, []) == Action("key", keys=("return",))


async def test_end_to_end_with_executor_and_fake_computer() -> None:
    policy = _policy(
        [
            _interaction("i1", _call("click", x=500, y=500)),
            _interaction("i2", _call("type", text="hello", press_enter=True, call_id="c2")),
            _interaction("i3", _text_output("typed the greeting")),
        ]
    )
    computer = FakeComputer()
    result = await ComputerUseExecutor(computer, policy, max_steps=10).run("say hello")
    assert result.done
    assert result.final_note == "typed the greeting"
    assert [a.type for a in computer.calls] == ["click", "type", "key"]
