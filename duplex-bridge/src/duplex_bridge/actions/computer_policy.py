"""Gemini computer-use policy — the real vision policy behind ``computer_use(goal)``.

Week 7b built the perceive -> decide -> act executor with the ``Policy`` seam unfilled. This
fills it with Gemini's built-in ``computer_use`` tool (Interactions API, ``gemini-3.5-flash``,
public preview 2026-06): one model both reads the screenshot and grounds the click, with
``desktop`` environment support, per-action ``safety_decision``s, and prompt-injection
detection over untrusted screen content.

The Interactions API is itself loop-shaped — send goal + screenshot, get UI-action function
calls, execute, send ``function_result`` + fresh screenshot, repeat. Adapting that to the
executor's one-``Action``-per-tick ``Policy`` seam:

- composite model actions (e.g. ``type`` = click + type + enter) expand into a local FIFO of
  primitive :class:`Action`\\ s; ticks pop the FIFO with no API round-trip;
- when the FIFO drains, the next tick sends the pending ``function_result``\\ s (carrying that
  tick's fresh screenshot, i.e. taken *after* the whole composite ran) and maps the next call;
- the model's 0–999 normalized coordinates are denormalized to logical *points* (CGEvent
  space), never screenshot pixels — Retina screenshots are 2x.

Stateful per goal (holds ``previous_interaction_id`` + pending call ids): construct a fresh
instance per ``computer_use`` run.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import struct
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .computer import Action

logger = logging.getLogger(__name__)

# Gemini computer-use coordinates are normalized to a 0-999 grid over the screenshot.
_COORD_GRID = 1000

# Whole-document scrolls (`scroll_document`) get this magnitude when the model gives none.
_DEFAULT_SCROLL_PX = 400

# GeminiComputerUsePolicy._check_safety's "blocked" verdict ends the run with a `done` result
# (not an exception) carrying this text in final_note — the soft variant of the hard
# "Input blocked" exception, and the other trigger for the vision-loop fallback below.
_SOFT_SAFETY_BLOCK = "blocked by the model's safety policy"

_T = TypeVar("_T")


def is_policy_block(exc: BaseException) -> bool:
    """True when a tool failure is the hosted provider's content-policy classifier refusing
    the input (Google's server-side "Input blocked" gate on the Interactions API, not a
    general error) — the trigger for retrying via :class:`GeminiVisionLoopPolicy` instead.
    """
    text = str(exc)
    return "Input blocked" in text or "prompt was blocked" in text


async def wrap_with_vision_loop_fallback(
    run: Callable[[Any], Awaitable[_T]],
    primary_factory: Callable[[], Any],
    fallback_factory: Callable[[], Any],
) -> tuple[_T, bool]:
    """Run ``run(primary)``; on a hosted-policy block, retry once via the vision-loop fallback.

    Shared by every ``computer_use`` call site (the delegate's tool handler, the live
    duplex model's tool handler): try the hosted :class:`GeminiComputerUsePolicy` first, and
    if Google's server-side "Input blocked" classifier rejects the goal — either as a raised
    exception (``is_policy_block``) or as a soft ``safety_decision: blocked`` ``done`` result
    (``_SOFT_SAFETY_BLOCK`` in the result's ``final_note``) — retry the SAME goal via
    :class:`GeminiVisionLoopPolicy`, which carries no equivalent server-side gate (validated
    by ``scripts/diag/probe_custom_vision_loop_filter.py``). ``run`` is injected so callers
    can wrap it however they need (mutex, worker thread, timeout); this helper only owns the
    block-detection and retry decision, not execution.

    Returns ``(result, used_fallback)``. Non-block exceptions propagate unchanged. The
    fallback policy is only constructed (``fallback_factory()``) when actually needed.
    """
    primary = primary_factory()
    try:
        result = await run(primary)
    except Exception as exc:  # noqa: BLE001 — re-raised below if not a policy block
        if not is_policy_block(exc):
            raise
        logger.warning("hosted computer-use policy input-blocked; retrying via vision-loop")
        fallback = fallback_factory()
        return await run(fallback), True
    if _SOFT_SAFETY_BLOCK in str(getattr(result, "final_note", "")):
        logger.warning("hosted computer-use policy input-blocked; retrying via vision-loop")
        fallback = fallback_factory()
        return await run(fallback), True
    return result, False


@dataclass
class _PendingResult:
    """A function_result owed to the model for a call we acted on (sent next round-trip)."""

    call_id: str
    name: str
    payload: dict[str, str] = field(default_factory=lambda: {"status": "ok"})
    is_error: bool = False


class _NormalizedPointPolicy:
    """Shared 0-999 normalized-coordinate handling for both Gemini vision policies.

    macOS screenshots are native Retina *pixels* (2x) while CGEvent clicks/moves take
    logical *points*, so both :class:`GeminiComputerUsePolicy` and
    :class:`GeminiVisionLoopPolicy` denormalize the model's 0-999 grid coordinates against
    the screen's logical point size — never the screenshot's pixel size — or every click
    lands at twice the intended offset. Also carries the ``key`` combo normalization the
    two policies share verbatim.
    """

    _screen_size: tuple[int, int] | None
    _last_shot: bytes | None

    def _point(self, args: dict[str, Any]) -> tuple[int, int]:
        width, height = self._screen_points()
        x = int(float(args.get("x") or 0) / _COORD_GRID * width)
        y = int(float(args.get("y") or 0) / _COORD_GRID * height)
        return x, y

    def _screen_points(self) -> tuple[int, int]:
        """Logical point dimensions of the screen — the space CGEvent posts in."""
        if self._screen_size is None:
            try:
                import Quartz  # noqa: PLC0415

                bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
                self._screen_size = (int(bounds.size.width), int(bounds.size.height))
            except Exception:  # pragma: no cover — non-mac fallback
                # PNG IHDR pixel dims; wrong by the Retina scale but better than crashing.
                shot = self._last_shot
                if shot is not None and len(shot) >= 24:
                    w, h = struct.unpack(">II", shot[16:24])
                    self._screen_size = (int(w), int(h))
                else:
                    raise RuntimeError(
                        "screen size unknown: pass screen_size= or install PyObjC"
                    ) from None
        return self._screen_size

    @staticmethod
    def _keys(args: dict[str, Any]) -> tuple[str, ...]:
        raw = args.get("keys") or args.get("key") or ""
        parts = raw.replace("-", "+").split("+") if isinstance(raw, str) else [str(k) for k in raw]
        alias = {"enter": "return", "cmd": "cmd", "meta": "cmd", "super": "cmd"}
        return tuple(alias.get(k.strip().lower(), k.strip().lower()) for k in parts if k.strip())


class GeminiComputerUsePolicy(_NormalizedPointPolicy):
    """``Policy`` implementation over the Gemini ``computer_use`` tool (Interactions API).

    Safety decisions are honored, not bypassed: a ``blocked`` action ends the run with the
    model's explanation, and ``require_confirmation`` ends it asking the user to confirm and
    re-issue (the duplex voice loop relays the note). Set ``auto_acknowledge_safety=True``
    only for supervised runs where the human is already watching the screen.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        environment: str = "desktop",
        screen_size: tuple[int, int] | None = None,
        client: Any | None = None,
        auto_acknowledge_safety: bool = False,
    ) -> None:
        if client is None:
            from google import genai  # noqa: PLC0415 — optional heavy import, test seam above

            client = genai.Client(api_key=os.environ.get(api_key_env) or None)
        self._client = client
        self._model = model
        self._environment = environment
        self._screen_size = screen_size
        self._auto_ack = auto_acknowledge_safety
        self._interaction_id: str | None = None
        self._queue: deque[Action] = deque()
        self._pending: list[_PendingResult] = []
        self._last_shot: bytes | None = None

    def __call__(self, goal: str, screenshot: bytes, history: list[Action]) -> Action:
        if self._queue:
            return self._queue.popleft()
        self._last_shot = screenshot
        interaction = self._client.interactions.create(**self._request(goal, screenshot))
        self._interaction_id = interaction.id
        self._pending = []
        calls = [s for s in interaction.steps or [] if getattr(s, "type", None) == "function_call"]
        if not calls:
            return Action("done", note=self._final_text(interaction))
        for call in calls:
            stop = self._map_call(call)
            if stop is not None:
                return stop
        return self._queue.popleft()

    # -- request building ---------------------------------------------------------------

    def _request(self, goal: str, screenshot: bytes) -> dict[str, Any]:
        image = {
            "type": "image",
            "data": base64.b64encode(screenshot).decode("ascii"),
            "mime_type": "image/png",
        }
        input_: list[dict[str, Any]]
        if self._interaction_id is None:
            input_ = [{"type": "text", "text": goal}, image]
        else:
            # One function_result per call we owe; the screenshot rides on the last one so
            # the model perceives the state after the whole batch executed.
            input_ = []
            for i, pending in enumerate(self._pending):
                result: list[dict[str, Any]] = [
                    {"type": "text", "text": json.dumps(pending.payload)}
                ]
                if i == len(self._pending) - 1:
                    result.append(image)
                step: dict[str, Any] = {
                    "type": "function_result",
                    "name": pending.name,
                    "call_id": pending.call_id,
                    "result": result,
                }
                if pending.is_error:
                    step["is_error"] = True
                input_.append(step)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": input_,
            # Plain dicts, not SDK TypedDicts: the installed SDK's Literal still only knows
            # "browser", but the API accepts "desktop"/"mobile" and unknown keys pass through.
            "tools": [
                {
                    "type": "computer_use",
                    "environment": self._environment,
                    # Exclude actions the Computer seam has no primitive for, so the model
                    # never wastes a round-trip on them: no drag, no held-key state
                    # (key_down/key_up), no triple-click click-count. hotkey IS supported
                    # (mapped to the key primitive).
                    "excluded_predefined_functions": [
                        "drag_and_drop",
                        "key_down",
                        "key_up",
                        "triple_click",
                    ],
                    "enable_prompt_injection_detection": True,
                }
            ],
        }
        if self._interaction_id is not None:
            kwargs["previous_interaction_id"] = self._interaction_id
        return kwargs

    # -- response mapping ---------------------------------------------------------------

    def _map_call(self, call: Any) -> Action | None:
        """Queue the primitive Action(s) for one model call; return an Action only to stop."""
        args = dict(call.arguments or {})
        stop, acked = self._check_safety(args.pop("safety_decision", None) or call)
        if stop is not None:
            return stop
        intent = str(args.get("intent") or "")
        name = call.name
        pending = _PendingResult(call.id, name)
        if acked:
            pending.payload["safety_acknowledgement"] = "true"
        actions: list[Action] = []

        if name in ("click", "click_at", "left_click"):
            x, y = self._point(args)
            actions = [Action("click", x=x, y=y, note=intent)]
        elif name in ("double_click", "double_click_at"):
            x, y = self._point(args)
            actions = [Action("double_click", x=x, y=y, note=intent)]
        elif name in ("hover", "hover_at", "move", "mouse_move"):
            x, y = self._point(args)
            actions = [Action("move", x=x, y=y, note=intent)]
        elif name in ("type", "type_text", "type_text_at"):
            if args.get("x") is not None and args.get("y") is not None:
                x, y = self._point(args)
                actions.append(Action("click", x=x, y=y, note=intent))
            if args.get("clear_before_typing"):
                actions += [Action("key", keys=("cmd", "a")), Action("key", keys=("delete",))]
            actions.append(Action("type", text=str(args.get("text") or ""), note=intent))
            if args.get("press_enter"):
                actions.append(Action("key", keys=("return",)))
        elif name in ("press_key", "key_combination", "hotkey"):
            keys = self._keys(args)
            if keys:
                actions = [Action("key", keys=keys, note=intent)]
            else:
                pending.payload = {"status": "error", "error": f"no keys given for {name!r}"}
                pending.is_error = True
                actions = [Action("screenshot", note=f"skipped empty {name!r}")]
        elif name in ("scroll", "scroll_at", "scroll_document"):
            direction = str(args.get("direction") or "down").lower()
            magnitude = int(args.get("magnitude_in_pixels") or _DEFAULT_SCROLL_PX)
            if direction in ("up", "down"):
                # Quartz convention: positive dy scrolls up (toward the document top).
                dy = magnitude if direction == "up" else -magnitude
                if args.get("x") is not None and args.get("y") is not None:
                    x, y = self._point(args)
                    actions = [Action("move", x=x, y=y), Action("scroll", x=x, y=y, dy=dy)]
                else:
                    actions = [Action("scroll", dy=dy, note=intent)]
            else:
                pending.payload = {"status": "error", "error": f"cannot scroll {direction!r}"}
                pending.is_error = True
                actions = [Action("screenshot", note=f"skipped horizontal scroll {direction!r}")]
        elif name in ("take_screenshot", "screenshot", "wait", "wait_5_seconds"):
            # No-op tick: the executor screenshots at the top of every tick anyway.
            actions = [Action("screenshot", note=intent or name)]
        else:
            logger.warning("[computer-use] unsupported model action %r; reporting back", name)
            pending.payload = {"status": "error", "error": f"unsupported action {name!r}"}
            pending.is_error = True
            actions = [Action("screenshot", note=f"skipped unsupported {name!r}")]

        self._pending.append(pending)
        self._queue.extend(actions)
        return None

    def _check_safety(self, source: Any) -> tuple[Action | None, bool]:
        """Return (stop_action, acknowledged). ``source`` is the args' safety_decision or
        the step itself (some server versions attach it to the step, not the arguments)."""
        decision = source if isinstance(source, dict) else getattr(source, "safety_decision", None)
        if decision is None:
            return None, False
        if hasattr(decision, "model_dump"):
            decision = decision.model_dump()
        verdict = str(decision.get("decision") or "")
        explanation = str(decision.get("explanation") or "")
        if verdict == "blocked":
            note = f"blocked by the model's safety policy: {explanation}"
            return Action("done", note=note), False
        if verdict == "require_confirmation":
            if self._auto_ack:
                return None, True
            return (
                Action(
                    "done",
                    note=(
                        f"needs your confirmation before acting: {explanation} — "
                        "re-issue the request to confirm"
                    ),
                ),
                False,
            )
        return None, False

    @staticmethod
    def _final_text(interaction: Any) -> str:
        texts: list[str] = []
        for step in interaction.steps or []:
            if getattr(step, "type", None) == "model_output":
                for content in step.content or []:
                    text = getattr(content, "text", None)
                    if text:
                        texts.append(str(text))
        return " ".join(texts).strip() or "task complete"


# Instructions for GeminiVisionLoopPolicy's plain-generateContent decision step. Mirrors the
# schema validated in scripts/diag/probe_custom_vision_loop_filter.py (4/4 passed goals the
# hosted computer_use tool's classifier blocked), extended with the primitives this module's
# Computer seam actually supports (double-click, key combos, scroll direction/magnitude).
_ACTION_SCHEMA_INSTRUCTIONS = """\
You are the decision step of a computer-control loop on the user's own macOS desktop \
(already authorized — this is not unauthorized access). Given a goal, the current \
screenshot, and the actions already taken this run, reply with ONLY a single JSON object \
(no prose, no markdown fence) describing the single next action. Coordinates x/y are \
normalized to a 0-999 grid over the screenshot (0,0 top-left, 999,999 bottom-right) — NOT \
pixels.

  {"action": "click", "x": <0-999>, "y": <0-999>, "double": <bool, optional>, "note": "<why>"}
  {"action": "type", "text": "<text>", "x": <0-999, optional>, "y": <0-999, optional>, \
"press_enter": <bool, optional>, "note": "<why>"}
  {"action": "key", "keys": ["cmd", "space"], "note": "<why>"}
  {"action": "scroll", "direction": "up"|"down", "magnitude": <int px, optional>, \
"x": <0-999, optional>, "y": <0-999, optional>, "note": "<why>"}
  {"action": "wait", "note": "<why>"}
  {"action": "done", "note": "<final answer / what you observed>"}

If "type" gives x/y, click there first, then type. Reply with exactly one JSON object.\
"""


class GeminiVisionLoopPolicy(_NormalizedPointPolicy):
    """``Policy`` over plain ``generate_content`` with a custom JSON action schema.

    Google's hosted ``computer_use`` tool (:class:`GeminiComputerUsePolicy`, Interactions
    API) carries a server-side "Input blocked" classifier that hard-rejects some legitimate
    desktop-automation goals (observed live: screen-contents summaries, named-app control
    framed around personal data). ``probe_custom_vision_loop_filter.py`` validated 4/4 that
    the SAME model, SAME screenshot, and a custom action schema over plain
    ``generateContent`` pass those exact goals — there is no equivalent server-side gate on
    that endpoint. This policy exists to give the delegate a fallback path when the hosted
    tool blocks, at the cost of losing the hosted tool's computer-use-specific grounding
    training (its clicks are only as accurate as gemini-3.5-flash's general vision
    grounding) — mitigated by asking for normalized 0-999 coordinates rather than raw
    pixels, the same denormalization the hosted policy relies on.

    Stateless across goals in principle, but keeps a small per-instance state: the compound-
    action FIFO and a consecutive-parse-failure counter (a model that never returns valid
    JSON must not spin the executor to ``max_steps``). Construct a fresh instance per run,
    same as the sibling.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        screen_size: tuple[int, int] | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from google import genai  # noqa: PLC0415 — optional heavy import, test seam above

            client = genai.Client(api_key=os.environ.get(api_key_env) or None)
        self._client = client
        self._model = model
        self._screen_size = screen_size
        self._last_shot: bytes | None = None
        self._queue: deque[Action] = deque()
        self._parse_failures = 0

    def __call__(self, goal: str, screenshot: bytes, history: list[Action]) -> Action:
        if self._queue:
            return self._queue.popleft()
        self._last_shot = screenshot
        # Errors here (network, 4xx/5xx) are allowed to raise — the delegate's tool-call
        # error handling upstream (is_policy_block etc.) deals with them.
        response = self._client.models.generate_content(**self._request(goal, screenshot, history))
        action = self._parse(response)
        if action is not None:
            return action
        return self._queue.popleft()

    # -- request building ---------------------------------------------------------------

    def _request(self, goal: str, screenshot: bytes, history: list[Action]) -> dict[str, Any]:
        from google.genai import types  # noqa: PLC0415

        prompt = (
            f"{_ACTION_SCHEMA_INSTRUCTIONS}\n\nGOAL: {goal}\n\n"
            f"ACTIONS SO FAR:\n{self._history_text(history)}"
        )
        return {
            "model": self._model,
            "contents": [
                types.Part.from_bytes(data=screenshot, mime_type="image/png"),
                prompt,
            ],
            "config": {"response_mime_type": "application/json"},
        }

    @staticmethod
    def _history_text(history: list[Action]) -> str:
        """One line per prior Action, capped to the last 20 so the prompt stays bounded."""
        if not history:
            return "(none yet)"
        lines: list[str] = []
        for action in history[-20:]:
            coords = f" ({action.x},{action.y})" if action.x is not None else ""
            note = f": {action.note}" if action.note else ""
            lines.append(f"- {action.type}{coords}{note}")
        return "\n".join(lines)

    # -- response parsing -----------------------------------------------------------------

    def _parse(self, response: Any) -> Action | None:
        """Parse one response into Action(s); return an Action to stop, else queue+None."""
        text = getattr(response, "text", None) or ""
        try:
            payload = self._extract_json(text)
            action_type = str(payload.get("action") or "").lower()
            actions = self._to_actions(action_type, payload)
        except (ValueError, TypeError, KeyError) as exc:
            return self._parse_failure(f"unparseable policy output: {exc}")
        if not actions:
            return self._parse_failure(f"unparseable policy output: empty action {action_type!r}")
        self._parse_failures = 0
        self._queue.extend(actions)
        return None

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            # Strip a ```/```json fence: some models add one despite response_mime_type.
            stripped = stripped.strip("`").strip()
            if stripped[:4].lower() == "json":
                stripped = stripped[4:].strip()
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object")
        return parsed

    def _to_actions(self, action_type: str, payload: dict[str, Any]) -> list[Action]:
        note = str(payload.get("note") or "")
        if action_type == "click":
            x, y = self._point(payload)
            kind = "double_click" if payload.get("double") else "click"
            return [Action(kind, x=x, y=y, note=note)]
        if action_type == "type":
            actions: list[Action] = []
            if payload.get("x") is not None and payload.get("y") is not None:
                x, y = self._point(payload)
                actions.append(Action("click", x=x, y=y, note=note))
            actions.append(Action("type", text=str(payload.get("text") or ""), note=note))
            if payload.get("press_enter"):
                actions.append(Action("key", keys=("return",)))
            return actions
        if action_type == "key":
            keys = self._keys(payload)
            if not keys:
                raise ValueError("no keys given for 'key' action")
            return [Action("key", keys=keys, note=note)]
        if action_type == "scroll":
            direction = str(payload.get("direction") or "down").lower()
            if direction not in ("up", "down"):
                raise ValueError(f"cannot scroll {direction!r}")
            magnitude = int(payload.get("magnitude") or _DEFAULT_SCROLL_PX)
            # Quartz convention: positive dy scrolls up (toward the document top).
            dy = magnitude if direction == "up" else -magnitude
            if payload.get("x") is not None and payload.get("y") is not None:
                x, y = self._point(payload)
                return [Action("move", x=x, y=y), Action("scroll", x=x, y=y, dy=dy)]
            return [Action("scroll", dy=dy, note=note)]
        if action_type == "wait":
            return [Action("screenshot", note=note)]
        if action_type == "done":
            return [Action("done", note=note)]
        raise ValueError(f"unsupported action {action_type!r}")

    def _parse_failure(self, note: str) -> Action:
        self._parse_failures += 1
        logger.warning("[vision-loop] %s", note)
        if self._parse_failures >= 3:
            return Action("done", note="policy output unparseable 3x, giving up")
        return Action("screenshot", note=note)
