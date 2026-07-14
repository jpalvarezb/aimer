"""General computer-use actuator — cross-application host control.

Week 7 shipped *bespoke* per-app actions (a specific IDE edit, a specific Chrome compare). Real
production is **cross-application**: the user points at anything in any app and says what they
want. That needs general computer use — OS-level *screenshot + mouse + keyboard* behind one seam,
driven by a vision **policy** in a perceive -> decide -> act loop — not a handler per task.

Layers (all provider/OS-neutral except the concrete `MacOSComputer`):
  - ``Computer``            — ABC of OS primitives (screenshot/click/move/type/key/scroll).
  - ``MacOSComputer``       — macOS: Quartz ``CGEvent`` input + ``screencapture`` (lazy import).
  - ``FakeComputer``        — records actions, returns a canned screenshot (tests / dry-run).
  - ``ComputerUseExecutor`` — the loop: screenshot -> policy(goal, shot, hist) -> Action -> apply,
                              until the policy returns ``done`` or ``max_steps``.
  - ``Policy``              — ``(goal, screenshot, history) -> Action``; a real one wraps a
                              computer-use vision model; a scripted one drives tests.

Exposed to the duplex model as a single ``computer_use(goal)`` tool, dispatched off the audio hot
path by the Week-6 worker. The executor loop is verified deterministically with ``FakeComputer`` +
a scripted policy; driving a real desktop (``MacOSComputer`` + a live policy) needs Accessibility +
Screen Recording permission and is run by the user.
"""

from __future__ import annotations

import inspect
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Action:
    """One computer-use step. ``type`` selects the primitive; ``done`` ends the loop."""

    type: str  # click | double_click | move | type | key | scroll | done
    x: int | None = None
    y: int | None = None
    text: str | None = None
    keys: tuple[str, ...] = ()
    dy: int = 0
    note: str = ""  # model rationale, or the final answer when type == "done"


@dataclass
class ComputerUseResult:
    """Outcome of a computer-use run."""

    goal: str
    steps: int
    done: bool
    actions: list[Action] = field(default_factory=list)
    final_note: str = ""


# A policy looks at the goal + current screenshot + prior actions and returns the next Action.
Policy = Callable[[str, bytes, "list[Action]"], "Action | Awaitable[Action]"]


class Computer(ABC):
    """OS primitives a policy drives — one seam; macOS/Windows/Linux/fake implement it."""

    @abstractmethod
    def screenshot(self) -> bytes:
        """Return a PNG screenshot of the current screen."""

    @abstractmethod
    def click(self, x: int, y: int, *, double: bool = False) -> None: ...
    @abstractmethod
    def move(self, x: int, y: int) -> None: ...
    @abstractmethod
    def type_text(self, text: str) -> None: ...
    @abstractmethod
    def key(self, keys: tuple[str, ...]) -> None: ...
    @abstractmethod
    def scroll(self, x: int, y: int, dy: int) -> None: ...

    def apply(self, action: Action) -> None:
        """Dispatch an Action to the matching primitive (no-op for screenshot/done)."""
        if action.type in ("click", "double_click"):
            self.click(action.x or 0, action.y or 0, double=action.type == "double_click")
        elif action.type == "move":
            self.move(action.x or 0, action.y or 0)
        elif action.type == "type":
            self.type_text(action.text or "")
        elif action.type == "key":
            self.key(action.keys)
        elif action.type == "scroll":
            self.scroll(action.x or 0, action.y or 0, action.dy)
        elif action.type in ("screenshot", "done"):
            return
        else:
            raise ValueError(f"unknown action type: {action.type!r}")


class ComputerUseExecutor:
    """Run a perceive -> decide -> act loop until the policy says ``done`` or ``max_steps``.

    This is the cross-application core: it never knows which app it is driving — the policy decides
    from the screenshot, and the primitives are OS-level. Runs off the audio hot path.
    """

    def __init__(
        self,
        computer: Computer,
        policy: Policy,
        max_steps: int = 12,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self._computer = computer
        self._policy = policy
        self._max_steps = max_steps
        # Checked at the top of every tick: the run may live in a worker thread that
        # cannot be cancelled from the loop, so cancellation/shutdown flips this flag
        # instead — otherwise an orphaned run keeps driving the real mouse.
        self._should_stop = should_stop

    async def run(self, goal: str) -> ComputerUseResult:
        actions: list[Action] = []
        for step in range(self._max_steps):
            if self._should_stop is not None and self._should_stop():
                logger.info("[computer-use] stopped externally after %d step(s)", step)
                return ComputerUseResult(goal, step, False, actions, "stopped by caller")
            shot = self._computer.screenshot()
            decision = self._policy(goal, shot, actions)
            action = await decision if inspect.isawaitable(decision) else decision
            if action.type == "done":
                logger.info("[computer-use] done in %d step(s): %s", step, action.note)
                return ComputerUseResult(goal, step, True, actions, action.note)
            self._computer.apply(action)
            actions.append(action)
        logger.info("[computer-use] hit max_steps=%d for goal %r", self._max_steps, goal)
        return ComputerUseResult(goal, self._max_steps, False, actions, "max steps reached")


class FakeComputer(Computer):
    """Records primitive calls and returns a canned screenshot — for tests / dry-run."""

    def __init__(self, screenshot: bytes = b"\x89PNG\r\n") -> None:
        self._screenshot = screenshot
        self.calls: list[Action] = []
        self.shots = 0

    def screenshot(self) -> bytes:
        self.shots += 1
        return self._screenshot

    def click(self, x: int, y: int, *, double: bool = False) -> None:
        self.calls.append(Action("double_click" if double else "click", x=x, y=y))

    def move(self, x: int, y: int) -> None:
        self.calls.append(Action("move", x=x, y=y))

    def type_text(self, text: str) -> None:
        self.calls.append(Action("type", text=text))

    def key(self, keys: tuple[str, ...]) -> None:
        self.calls.append(Action("key", keys=tuple(keys)))

    def scroll(self, x: int, y: int, dy: int) -> None:
        self.calls.append(Action("scroll", x=x, y=y, dy=dy))


# macOS keycodes for the keys a computer-use policy commonly needs (extend as required).
# fmt: off
_MAC_KEYCODES: dict[str, int] = {
    "return": 36,
    "enter": 36,
    "tab": 48,
    "space": 49,
    "delete": 51,
    "backspace": 51,
    "escape": 53,
    "esc": 53,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "home": 115,
    "end": 119,
    "pageup": 116,
    "page_up": 116,
    "pagedown": 121,
    "page_down": 121,
    # ANSI letters/digits/punctuation — needed for shortcuts like cmd+a / cmd+c / cmd+t.
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26,
    "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35,
    "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45,
    "m": 46, ".": 47, "`": 50,
}
# fmt: on
_MAC_MODIFIERS = {"cmd", "command", "shift", "ctrl", "control", "alt", "option"}


class MacOSComputer(Computer):
    """macOS computer-use via Quartz ``CGEvent`` input + ``screencapture`` (lazy PyObjC import).

    Requires Accessibility (to post input events) and Screen Recording (for screenshots). Posting
    synthetic input drives the real desktop, so this is exercised live by the user, not in CI.
    """

    def screenshot(self) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as fh:
            path = fh.name
            subprocess.run(["screencapture", "-x", "-t", "png", path], check=True, timeout=10)
            return Path(path).read_bytes()

    def _mouse(self, x: int, y: int, *, double: bool) -> None:
        import Quartz  # noqa: PLC0415

        for kind in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
            event = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
            if double:
                Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, 2)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def click(self, x: int, y: int, *, double: bool = False) -> None:
        self._mouse(x, y, double=double)

    def move(self, x: int, y: int) -> None:
        import Quartz  # noqa: PLC0415

        event = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventMouseMoved, (x, y), Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def type_text(self, text: str) -> None:
        import Quartz  # noqa: PLC0415

        for ch in text:
            for down in (True, False):
                event = Quartz.CGEventCreateKeyboardEvent(None, 0, down)
                Quartz.CGEventKeyboardSetUnicodeString(event, len(ch), ch)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def key(self, keys: tuple[str, ...]) -> None:
        import Quartz  # noqa: PLC0415

        mods, base = [], None
        for k in keys:
            (mods.append(k.lower()) if k.lower() in _MAC_MODIFIERS else None)
            if k.lower() not in _MAC_MODIFIERS:
                base = k.lower()
        flags = 0
        flag_map = {
            "cmd": Quartz.kCGEventFlagMaskCommand,
            "command": Quartz.kCGEventFlagMaskCommand,
            "shift": Quartz.kCGEventFlagMaskShift,
            "ctrl": Quartz.kCGEventFlagMaskControl,
            "control": Quartz.kCGEventFlagMaskControl,
            "alt": Quartz.kCGEventFlagMaskAlternate,
            "option": Quartz.kCGEventFlagMaskAlternate,
        }
        for m in mods:
            flags |= flag_map.get(m, 0)
        keycode = _MAC_KEYCODES.get(base or "")
        if keycode is None:
            logger.warning("[computer-use] unmapped key %r; skipping", base)
            return
        for down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, keycode, down)
            if flags:
                Quartz.CGEventSetFlags(event, flags)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def scroll(self, x: int, y: int, dy: int) -> None:
        import Quartz  # noqa: PLC0415

        event = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitPixel, 1, dy)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def run_computer_use(
    goal: str,
    computer: Computer,
    policy: Policy,
    max_steps: int = 12,
    should_stop: Callable[[], bool] | None = None,
) -> Any:
    """Sync entry point for the worker: build an executor and run the loop to completion."""
    import asyncio  # noqa: PLC0415

    executor = ComputerUseExecutor(computer, policy, max_steps=max_steps, should_stop=should_stop)
    return asyncio.run(executor.run(goal))


def run_computer_use_with_timeout(
    goal: str,
    computer: Computer,
    policy: Policy,
    max_steps: int = 12,
    timeout_s: float = 120.0,
    should_stop: Callable[[], bool] | None = None,
) -> ComputerUseResult:
    """Sync entry point with a per-goal wall-clock timeout: ALWAYS returns a final result.

    Live finding: a hosted-policy ``computer_use`` run can dangle indefinitely (observed:
    two Teams-click calls that never produced a result), leaving the tool call silent
    forever since ``ToolDispatcher`` is waiting on a coroutine that never resolves. Wrapping
    the executor run in ``asyncio.wait_for`` guarantees a final ``ComputerUseResult`` either
    way — on timeout, ``done`` is ``False`` and ``final_note`` explains why, exactly like a
    ``max_steps`` exhaustion, so the caller always gets a real ``FunctionResponse`` and never
    silence.
    """
    import asyncio  # noqa: PLC0415

    async def _run() -> ComputerUseResult:
        executor = ComputerUseExecutor(
            computer, policy, max_steps=max_steps, should_stop=should_stop
        )
        try:
            return await asyncio.wait_for(executor.run(goal), timeout=timeout_s)
        except TimeoutError:
            logger.warning("[computer-use] timed out after %.1fs for goal %r", timeout_s, goal)
            return ComputerUseResult(
                goal=goal,
                steps=max_steps,
                done=False,
                actions=[],
                final_note=f"computer_use timed out after {timeout_s:g}s: {goal}",
            )

    return asyncio.run(_run())


def click_pointer(computer: Computer, x: float, y: float, referent: str | None) -> str:
    """Deterministic click at the cursor's settled coordinates — no policy, no vision loop.

    Fast path for the common case: the live model calls ``computer_use`` for a goal that is
    just "click what the user is pointing at" and a fresh pointer referent already tells us
    exactly what/where that is. Skips the whole perceive->decide->act loop (no screenshot
    round-trip, no vision-model call) and issues one click via the ``Computer`` seam.
    """
    computer.apply(Action("click", x=int(x), y=int(y)))
    if referent:
        return f"clicked ({int(x)}, {int(y)}) — the pointed-at element: {referent}"
    return f"clicked ({int(x)}, {int(y)})"
