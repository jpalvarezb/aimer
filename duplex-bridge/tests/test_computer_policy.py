"""GeminiComputerUsePolicy — deterministic tests over a fake Interactions client.

Covers the seam adaptation (composite actions -> primitive FIFO, function_result round-trips,
0-999 coordinate denormalization to points) and the safety-decision behavior, with no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from duplex_bridge.actions.computer import Action, ComputerUseExecutor, FakeComputer
from duplex_bridge.actions.computer_policy import GeminiComputerUsePolicy, GeminiVisionLoopPolicy

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


def test_hotkey_maps_to_key_action() -> None:
    # Observed live: the model emits 'hotkey' for shortcuts; it must not be "unsupported".
    policy = _policy([_interaction("i1", _call("hotkey", keys="cmd+n"))])
    assert policy("g", PNG, []) == Action("key", keys=("cmd", "n"))
    policy = _policy([_interaction("i1", _call("hotkey", keys=["command", "shift", "t"]))])
    assert policy("g", PNG, []) == Action("key", keys=("command", "shift", "t"))


def test_hotkey_with_no_keys_reports_error_instead_of_empty_combo() -> None:
    policy = _policy([_interaction("i1", _call("hotkey"))])
    action = policy("g", PNG, [])
    assert action.type == "screenshot"
    (pending,) = policy._pending
    assert pending.is_error


def test_unmappable_predefined_functions_are_excluded_from_the_tool() -> None:
    # key_down/key_up/triple_click have no Computer primitive; excluding them stops the
    # model from wasting round-trips on actions we'd only report back as unsupported.
    policy = _policy([_interaction("i1", _call("click", x=1, y=1))])
    policy("g", PNG, [])
    client: Any = policy._client
    (tool,) = client.requests[0]["tools"]
    excluded = tool["excluded_predefined_functions"]
    assert {"drag_and_drop", "key_down", "key_up", "triple_click"} <= set(excluded)


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


# -- GeminiVisionLoopPolicy — the generateContent fallback over a custom action schema -----


class _FakeGenClient:
    """Duck-typed genai.Client: pops canned ``.text`` responses, records every request."""

    def __init__(self, texts: list[str]) -> None:
        outer = self

        class _Models:
            def generate_content(self, **kwargs: Any) -> SimpleNamespace:
                outer.requests.append(kwargs)
                return SimpleNamespace(text=outer.texts.pop(0))

        self.texts = texts
        self.requests: list[dict[str, Any]] = []
        self.models = _Models()


def _vision_policy(texts: list[str], **kwargs: Any) -> GeminiVisionLoopPolicy:
    kwargs.setdefault("screen_size", SCREEN)
    return GeminiVisionLoopPolicy(client=_FakeGenClient(texts), **kwargs)


def test_vision_loop_request_carries_screenshot_goal_and_json_mime_config() -> None:
    policy = _vision_policy(['{"action": "wait", "note": "x"}'])
    policy("open mail", PNG, [])
    client: Any = policy._client
    (request,) = client.requests
    assert request["model"] == "gemini-3.5-flash"
    assert request["config"] == {"response_mime_type": "application/json"}
    assert request["contents"][0].inline_data.data == PNG
    assert "open mail" in request["contents"][1]


def test_vision_loop_click_denormalizes_with_transparent_1000_grid() -> None:
    policy = _vision_policy(
        ['{"action": "click", "x": 500, "y": 250, "note": "center-ish"}'],
        screen_size=(1000, 1000),
    )
    action = policy("open mail", PNG, [])
    assert action == Action("click", x=500, y=250, note="center-ish")


def test_vision_loop_click_denormalizes_with_nontrivial_screen_size() -> None:
    policy = _vision_policy(['{"action": "click", "x": 500, "y": 500}'], screen_size=(1728, 1117))
    action = policy("g", PNG, [])
    assert action == Action("click", x=864, y=558, note="")


def test_vision_loop_double_click() -> None:
    policy = _vision_policy(['{"action": "click", "x": 0, "y": 0, "double": true}'])
    assert policy("g", PNG, []).type == "double_click"


def test_vision_loop_type_with_coords_queues_click_then_type() -> None:
    policy = _vision_policy(
        ['{"action": "type", "x": 500, "y": 500, "text": "hi", "note": "search box"}']
    )
    first = policy("g", PNG, [])
    assert first.type == "click" and (first.x, first.y) == (720, 450)
    assert policy("g", PNG, []) == Action("type", text="hi", note="search box")


def test_vision_loop_type_press_enter_appends_return() -> None:
    policy = _vision_policy(['{"action": "type", "text": "hello", "press_enter": true}'])
    assert policy("g", PNG, []) == Action("type", text="hello")
    assert policy("g", PNG, []) == Action("key", keys=("return",))


def test_vision_loop_key_mapping() -> None:
    policy = _vision_policy(['{"action": "key", "keys": ["cmd", "space"], "note": "spotlight"}'])
    assert policy("g", PNG, []) == Action("key", keys=("cmd", "space"), note="spotlight")


def test_vision_loop_key_with_no_keys_is_a_screenshot_noop() -> None:
    policy = _vision_policy(['{"action": "key", "note": "nothing to press"}'])
    action = policy("g", PNG, [])
    assert action.type == "screenshot"
    assert "unparseable" in action.note


def test_vision_loop_scroll_sign_convention() -> None:
    policy = _vision_policy(['{"action": "scroll", "direction": "down", "magnitude": 300}'])
    assert policy("g", PNG, []) == Action("scroll", dy=-300)

    policy = _vision_policy(['{"action": "scroll", "direction": "up"}'])
    assert policy("g", PNG, []).dy > 0


def test_vision_loop_scroll_with_coords_moves_then_scrolls() -> None:
    policy = _vision_policy(
        ['{"action": "scroll", "direction": "up", "x": 500, "y": 500, "magnitude": 100}']
    )
    move = policy("g", PNG, [])
    assert move == Action("move", x=720, y=450)
    scroll = policy("g", PNG, [])
    assert scroll == Action("scroll", x=720, y=450, dy=100)


def test_vision_loop_done_passes_note_through() -> None:
    policy = _vision_policy(['{"action": "done", "note": "finished the task"}'])
    assert policy("g", PNG, []) == Action("done", note="finished the task")


def test_vision_loop_wait_is_a_screenshot_noop() -> None:
    policy = _vision_policy(['{"action": "wait", "note": "loading"}'])
    action = policy("g", PNG, [])
    assert action.type == "screenshot" and action.note == "loading"


def test_vision_loop_code_fenced_json_parses() -> None:
    fenced = '```json\n{"action": "done", "note": "ok"}\n```'
    policy = _vision_policy([fenced])
    assert policy("g", PNG, []) == Action("done", note="ok")

    fenced_no_lang = '```\n{"action": "wait", "note": "x"}\n```'
    policy = _vision_policy([fenced_no_lang])
    assert policy("g", PNG, []).type == "screenshot"


def test_vision_loop_garbage_output_is_a_screenshot_noop_with_note() -> None:
    policy = _vision_policy(["not json at all"])
    action = policy("g", PNG, [])
    assert action.type == "screenshot"
    assert "unparseable" in action.note


def test_vision_loop_unsupported_action_is_a_screenshot_noop() -> None:
    policy = _vision_policy(['{"action": "teleport"}'])
    assert policy("g", PNG, []).type == "screenshot"


def test_vision_loop_three_consecutive_parse_failures_gives_up() -> None:
    policy = _vision_policy(["garbage", "still garbage", "more garbage"])
    assert policy("g", PNG, []).type == "screenshot"
    assert policy("g", PNG, []).type == "screenshot"
    third = policy("g", PNG, [])
    assert third.type == "done"
    assert "giving up" in third.note


def test_vision_loop_successful_action_resets_the_failure_counter() -> None:
    policy = _vision_policy(
        ["garbage", "still garbage", '{"action": "wait", "note": "ok"}', "garbage", "garbage"]
    )
    assert policy("g", PNG, []).type == "screenshot"  # failure 1
    assert policy("g", PNG, []).type == "screenshot"  # failure 2
    assert policy("g", PNG, []).type == "screenshot"  # success (wait), resets the counter
    assert policy("g", PNG, []).type == "screenshot"  # failure 1 again, not done yet
    assert policy("g", PNG, []).type == "screenshot"  # failure 2 again, still not done


def test_vision_loop_history_is_serialized_into_the_prompt_and_capped_at_20() -> None:
    history = [Action("click", x=i, y=i, note=f"step{i}") for i in range(25)]
    policy = _vision_policy(['{"action": "wait", "note": "x"}'])
    policy("do the thing", PNG, history)
    client: Any = policy._client
    (request,) = client.requests
    prompt = request["contents"][1]
    assert "do the thing" in prompt
    assert "step0" not in prompt  # only the last 20 of 25 kept
    assert "step5" in prompt
    assert "step24" in prompt


def test_vision_loop_empty_history_still_produces_a_valid_request() -> None:
    policy = _vision_policy(['{"action": "wait", "note": "x"}'])
    policy("g", PNG, [])
    client: Any = policy._client
    (request,) = client.requests
    assert "(none yet)" in request["contents"][1]


# -- wrap_with_vision_loop_fallback: the shared hosted-block -> vision-loop retry helper ----
#
# Live-fix 4a: delegate.py's _computer_use and __main__.py's live computer_use handler each
# had their own copy of "try the hosted policy; on an Input-blocked / soft-safety-block
# result, retry via GeminiVisionLoopPolicy". This pulls that into one shared helper so both
# call sites share one implementation.


async def test_wrap_with_vision_loop_fallback_retries_on_input_blocked_exception() -> None:
    from duplex_bridge.actions.computer import ComputerUseResult
    from duplex_bridge.actions.computer_policy import wrap_with_vision_loop_fallback

    calls: list[str] = []

    async def _run(policy: Any) -> ComputerUseResult:
        calls.append(policy)
        if policy == "primary":
            raise RuntimeError("Error code: 400 - {'error': {'message': 'Input blocked: nope'}}")
        return ComputerUseResult(goal="g", steps=1, done=True, final_note="done via fallback")

    result, used_fallback = await wrap_with_vision_loop_fallback(
        _run, lambda: "primary", lambda: "fallback"
    )

    assert used_fallback is True
    assert result.final_note == "done via fallback"
    assert calls == ["primary", "fallback"]


async def test_wrap_with_vision_loop_fallback_retries_on_soft_safety_block_note() -> None:
    from duplex_bridge.actions.computer import ComputerUseResult
    from duplex_bridge.actions.computer_policy import wrap_with_vision_loop_fallback

    async def _run(policy: Any) -> ComputerUseResult:
        if policy == "primary":
            return ComputerUseResult(
                goal="g",
                steps=1,
                done=True,
                final_note="blocked by the model's safety policy: payment page",
            )
        return ComputerUseResult(goal="g", steps=1, done=True, final_note="done via fallback")

    result, used_fallback = await wrap_with_vision_loop_fallback(
        _run, lambda: "primary", lambda: "fallback"
    )

    assert used_fallback is True
    assert result.final_note == "done via fallback"


async def test_wrap_with_vision_loop_fallback_passes_through_non_block_results_untouched() -> None:
    from duplex_bridge.actions.computer import ComputerUseResult
    from duplex_bridge.actions.computer_policy import wrap_with_vision_loop_fallback

    def _fallback_factory() -> Any:
        raise AssertionError("fallback must not be constructed when the primary succeeds")

    async def _run(policy: Any) -> ComputerUseResult:
        return ComputerUseResult(goal="g", steps=1, done=True, final_note="all good")

    result, used_fallback = await wrap_with_vision_loop_fallback(
        _run, lambda: "primary", _fallback_factory
    )

    assert used_fallback is False
    assert result.final_note == "all good"


async def test_wrap_with_vision_loop_fallback_reraises_non_block_exceptions() -> None:
    from duplex_bridge.actions.computer_policy import wrap_with_vision_loop_fallback

    async def _run(policy: Any) -> Any:
        raise RuntimeError("some other network error")

    with pytest.raises(RuntimeError, match="some other network error"):
        await wrap_with_vision_loop_fallback(_run, lambda: "primary", lambda: "fallback")
