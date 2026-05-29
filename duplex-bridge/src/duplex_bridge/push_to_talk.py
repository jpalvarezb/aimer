"""Push-to-talk key listener for the duplex bridge.

Holds a configurable key to speak; releasing it ends the turn immediately. The
listener runs on its own thread (provided by pynput) and marshals state changes
back onto the asyncio loop via ``call_soon_threadsafe``.

Optional: requires the ``pynput`` extra (``uv pip install -e "duplex-bridge[ptt]"``)
and macOS Input Monitoring permission for the global key listener.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class PushToTalkController:
    """Translate hold/release of a key into talking on/off callbacks."""

    def __init__(
        self,
        on_change: Callable[[bool], None],
        loop: asyncio.AbstractEventLoop,
        key_name: str = "cmd_r",
    ) -> None:
        self._on_change = on_change
        self._loop = loop
        self._key_name = key_name
        self._listener: Any = None
        self._target: Any = None
        self._held = False

    def start(self) -> bool:
        """Start the global key listener. Returns False if pynput is unavailable."""
        try:
            keyboard = importlib.import_module("pynput.keyboard")
        except ImportError:
            logger.warning("[ptt] pynput unavailable; install duplex-bridge[ptt]")
            return False

        self._target = _resolve_key(keyboard, self._key_name)
        if self._target is None:
            logger.warning("[ptt] unknown key %r; push-to-talk disabled", self._key_name)
            return False

        try:
            self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.start()
        except Exception as exc:  # pragma: no cover - platform/permission dependent
            logger.warning("[ptt] failed to start key listener: %s", exc)
            return False

        logger.info("[ptt] hold %s to talk", self._key_name)
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _on_press(self, key: Any) -> None:
        if key == self._target and not self._held:
            self._held = True
            self._loop.call_soon_threadsafe(self._on_change, True)

    def _on_release(self, key: Any) -> None:
        if key == self._target and self._held:
            self._held = False
            self._loop.call_soon_threadsafe(self._on_change, False)


def _resolve_key(keyboard: Any, name: str) -> Any:
    """Map a key name to a pynput key (e.g. 'cmd_r' → Key.cmd_r, 'a' → KeyCode('a'))."""
    key_enum = getattr(keyboard, "Key", None)
    if key_enum is not None and hasattr(key_enum, name):
        return getattr(key_enum, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    return None
