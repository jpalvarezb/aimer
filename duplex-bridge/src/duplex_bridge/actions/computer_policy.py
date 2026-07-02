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
from dataclasses import dataclass, field
from typing import Any

from .computer import Action

logger = logging.getLogger(__name__)

# Gemini computer-use coordinates are normalized to a 0-999 grid over the screenshot.
_COORD_GRID = 1000

# Whole-document scrolls (`scroll_document`) get this magnitude when the model gives none.
_DEFAULT_SCROLL_PX = 400


@dataclass
class _PendingResult:
    """A function_result owed to the model for a call we acted on (sent next round-trip)."""

    call_id: str
    name: str
    payload: dict[str, str] = field(default_factory=lambda: {"status": "ok"})
    is_error: bool = False


class GeminiComputerUsePolicy:
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
                    # No drag primitive on the Computer seam yet.
                    "excluded_predefined_functions": ["drag_and_drop"],
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
        elif name in ("press_key", "key_combination"):
            actions = [Action("key", keys=self._keys(args), note=intent)]
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

    def _keys(self, args: dict[str, Any]) -> tuple[str, ...]:
        raw = args.get("keys") or args.get("key") or ""
        parts = raw.replace("-", "+").split("+") if isinstance(raw, str) else [str(k) for k in raw]
        alias = {"enter": "return", "cmd": "cmd", "meta": "cmd", "super": "cmd"}
        return tuple(alias.get(k.strip().lower(), k.strip().lower()) for k in parts if k.strip())

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
    def _final_text(interaction: Any) -> str:
        texts: list[str] = []
        for step in interaction.steps or []:
            if getattr(step, "type", None) == "model_output":
                for content in step.content or []:
                    text = getattr(content, "text", None)
                    if text:
                        texts.append(str(text))
        return " ".join(texts).strip() or "task complete"
