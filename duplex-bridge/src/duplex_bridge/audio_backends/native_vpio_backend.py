"""Native macOS VPIO backend — hardware echo cancellation via a Swift helper.

The PyObjC ``vpio`` backend echo-cancels *capture* but is silent on *playback*
(driving CoreAudio's render path through PyObjC is the wall — see
``docs/vpio-backend-status.md``). Production voice apps do audio I/O in native
code, so this backend moves the whole VPIO ``AVAudioEngine`` (capture **and**
playback through an ``AVAudioPlayerNode``, which is VPIO's echo reference) into a
small Swift subprocess (``native/`` → ``aimer-vpio-helper``) and exchanges raw PCM
with it over the helper's stdio. It slots into the existing ``AudioBackend`` seam
with no changes to ``MicCapture``, turn detection, or the session.

Wire protocol (length-prefixed binary PCM; the helper logs on stderr):
- bridge → helper stdin (model audio): ``[4-byte LE uint32 N][N bytes 24 kHz mono int16]``.
  A zero-length frame (``N = 0``) is the barge-in flush sentinel: it tells the helper
  to drop everything already scheduled on its player node so playback stops at once.
- helper → bridge stdout (echo-cancelled mic): ``[4-byte LE uint32 N][N bytes 16 kHz mono int16]``,
  fixed ``N = 3200`` (100 ms), matching ``CaptureFormat(16000, 1, 1600)`` and the VAD timing.

The helper must be built first (``just build-native``). If the binary cannot be
found or spawned, ``start`` returns ``False`` for graceful fallback, exactly like
the ``sounddevice`` and ``vpio`` backends.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from duplex_bridge.audio_backends.base import CaptureFormat, OnFrame
from duplex_bridge.session import DuplexSession

logger = logging.getLogger(__name__)

_LEN_PREFIX = 4  # bytes, little-endian uint32 length header
_CAPTURE_RATE = 16_000
_FRAME_SAMPLES = 1_600  # 100 ms at 16 kHz; matches the VAD/send cadence (see base.py)
_FRAME_BYTES = _FRAME_SAMPLES * 2  # 3200 bytes int16 mono, what the helper emits
_MODEL_RATE = 24_000  # rate of session.on_audio_out chunks (the helper resamples)
# Zero-length stdin frame = "flush playback" (barge-in). The helper drops everything
# already scheduled on its player node so the assistant stops mid-utterance.
_FLUSH_FRAME = (0).to_bytes(_LEN_PREFIX, "little")

_HELPER_ENV = "AIMER_VPIO_HELPER"
_HELPER_BIN = "aimer-vpio-helper"
# Bound the playback write queue so a wedged helper stdin can't make writes block
# the reader. On overflow we drop the OLDEST model audio (stale anyway) rather than
# stall — mirrors SpeakerOutput._enqueue's drop-oldest policy.
_WRITE_QUEUE_MAX = 64
_TERMINATE_GRACE_S = 2.0


def _repo_root() -> Path:
    # …/duplex-bridge/src/duplex_bridge/audio_backends/native_vpio_backend.py
    #  parents: [0]=audio_backends [1]=duplex_bridge [2]=src [3]=duplex-bridge [4]=repo root
    return Path(__file__).resolve().parents[4]


def _resolve_helper_argv() -> list[str] | None:
    """Resolve the helper command: env override → built binary → PATH.

    ``$AIMER_VPIO_HELPER`` is parsed with ``shlex`` so it may carry arguments
    (e.g. ``python tests/native_helper_fake.py`` for headless tests).
    """
    env = os.environ.get(_HELPER_ENV)
    if env:
        argv = shlex.split(env)
        if argv:
            return argv
    built = _repo_root() / "native" / ".build" / "release" / _HELPER_BIN
    if built.is_file() and os.access(built, os.X_OK):
        return [str(built)]
    found = shutil.which(_HELPER_BIN)
    if found:
        return [found]
    return None


class NativeVpioBackend:
    """Mic capture + playback through the native Swift VPIO helper subprocess."""

    def __init__(self, *, model_rate: int = _MODEL_RATE) -> None:
        self._fmt = CaptureFormat(
            sample_rate=_CAPTURE_RATE, channels=1, frame_samples=_FRAME_SAMPLES
        )
        self._model_rate = model_rate
        self._on_frame: OnFrame | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False

        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_WRITE_QUEUE_MAX)

        self._dropped_frames = 0  # malformed/short capture frames discarded
        self._model_chunks_dropped = 0  # playback chunks dropped on write backpressure
        self._interruptions = 0  # barge-in flushes issued to the helper

    @property
    def capture_config(self) -> CaptureFormat:
        return self._fmt

    @property
    def stats(self) -> dict[str, int]:
        return {
            "dropped_frames": self._dropped_frames,
            "model_chunks_dropped": self._model_chunks_dropped,
            "interruptions": self._interruptions,
        }

    async def start(
        self,
        on_frame: OnFrame,
        session: DuplexSession,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        if self._running:
            return True

        argv = _resolve_helper_argv()
        if argv is None:
            logger.warning(
                "[native-vpio] helper not found; build it with `just build-native` "
                "or set %s. Falling back.",
                _HELPER_ENV,
            )
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            logger.warning("[native-vpio] could not spawn helper %r: %s", argv, exc)
            return False

        self._proc = proc
        self._on_frame = on_frame
        self._loop = loop
        self._running = True

        # The helper owns playback: model audio is framed and written to its stdin.
        session.on_audio_out(self._on_model_audio)
        # On barge-in, drop buffered playback so the assistant stops promptly.
        session.on_interrupt(self._on_interrupt)

        self._reader_task = loop.create_task(self._reader_loop())
        self._writer_task = loop.create_task(self._writer_loop())
        self._stderr_task = loop.create_task(self._stderr_loop())

        logger.info("[native-vpio] echo-cancelling helper started (%s)", argv[0])
        return True

    async def stop(self) -> None:
        self._running = False

        for task in (self._reader_task, self._writer_task, self._stderr_task):
            if task is not None:
                task.cancel()
        for task in (self._reader_task, self._writer_task, self._stderr_task):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._reader_task = self._writer_task = self._stderr_task = None

        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_GRACE_S)
            except (TimeoutError, asyncio.TimeoutError):
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
        self._proc = None

        # Drain any queued playback so a restart starts clean.
        while not self._write_queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._write_queue.get_nowait()

        self._on_frame = None
        self._loop = None

    async def _reader_loop(self) -> None:
        """Read length-prefixed capture frames from the helper and emit them."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        stdout = proc.stdout
        try:
            while True:
                header = await stdout.readexactly(_LEN_PREFIX)
                n = int.from_bytes(header, "little")
                payload = await stdout.readexactly(n)
                # We are already on the event loop here (no thread boundary), but the
                # seam contract is "deliver on_frame on the loop" — calling directly
                # satisfies it.
                if self._on_frame is not None:
                    self._on_frame(payload)
        except asyncio.IncompleteReadError:
            if self._running:
                logger.warning("[native-vpio] helper closed stdout (exited); audio stopped")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("[native-vpio] reader error: %s", exc)

    async def _writer_loop(self) -> None:
        """Drain queued model-audio frames to the helper's stdin."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        stdin = proc.stdin
        try:
            while True:
                framed = await self._write_queue.get()
                stdin.write(framed)
                await stdin.drain()
        except asyncio.CancelledError:
            raise
        except (BrokenPipeError, ConnectionResetError):
            if self._running:
                logger.warning("[native-vpio] helper stdin closed; playback stopped")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[native-vpio] writer error: %s", exc)

    async def _stderr_loop(self) -> None:
        """Surface the helper's stderr logs through our logger."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            async for line in proc.stderr:
                logger.info("[native-vpio:helper] %s", line.decode(errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("[native-vpio] stderr reader error: %s", exc)

    def _on_interrupt(self) -> None:
        """Barge-in: drop pending model audio and flush the helper's player node.

        Clearing the queue stops feeding stale audio; the flush sentinel (placed
        after the clear so it's next out) makes the helper drop everything already
        scheduled in CoreAudio — without it the assistant keeps talking until that
        backlog drains.
        """
        if not self._running:
            return
        while not self._write_queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._write_queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self._write_queue.put_nowait(_FLUSH_FRAME)
        self._interruptions += 1

    def _on_model_audio(self, audio: Any) -> None:
        """Frame one model-audio chunk and enqueue it for the helper (drop-oldest)."""
        if not self._running or not audio:
            return
        framed = len(audio).to_bytes(_LEN_PREFIX, "little") + bytes(audio)
        try:
            self._write_queue.put_nowait(framed)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._write_queue.get_nowait()
                self._model_chunks_dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                self._write_queue.put_nowait(framed)
