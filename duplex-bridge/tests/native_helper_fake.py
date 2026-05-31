"""Fake ``aimer-vpio-helper`` for headless tests of ``NativeVpioBackend``.

Speaks the same length-prefixed stdio protocol as the real Swift helper, so the
backend can be exercised against a real subprocess with no macOS/audio hardware:

- Emits ``--frames N`` capture frames on stdout, each ``[4-byte LE 3200][3200 bytes]``;
  frame ``i`` is filled with byte ``(--frame-byte + i) & 0xFF`` so tests can assert order.
- With ``--record PATH``: for every model-audio frame read from stdin, appends its
  payload length as a line to ``PATH`` (lets tests verify the bridge → helper path).
- With ``--hold``: keep reading stdin until EOF instead of exiting, so the process
  stays alive for teardown tests.

Run as: ``python native_helper_fake.py --frames 3 [--frame-byte 0] [--record p] [--hold]``.
"""

from __future__ import annotations

import argparse
import sys

_FRAME_BYTES = 3200
_LEN_PREFIX = 4


def _emit_frames(count: int, base_byte: int) -> None:
    out = sys.stdout.buffer
    for i in range(count):
        payload = bytes([(base_byte + i) & 0xFF]) * _FRAME_BYTES
        out.write(len(payload).to_bytes(_LEN_PREFIX, "little"))
        out.write(payload)
    out.flush()


def _read_exact(stream: object, n: int) -> bytes:
    buf = bytearray()
    read = sys.stdin.buffer.read
    while len(buf) < n:
        chunk = read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def _consume_stdin(record_path: str | None) -> None:
    rec = open(record_path, "a", encoding="utf-8") if record_path else None  # noqa: SIM115
    try:
        while True:
            header = _read_exact(sys.stdin.buffer, _LEN_PREFIX)
            if len(header) < _LEN_PREFIX:
                break
            n = int.from_bytes(header, "little")
            payload = _read_exact(sys.stdin.buffer, n)
            if len(payload) < n:
                break
            if rec is not None:
                rec.write(f"{n}\n")
                rec.flush()
    finally:
        if rec is not None:
            rec.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--frame-byte", type=int, default=0)
    parser.add_argument("--record", default=None)
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()

    print("[fake-helper] started", file=sys.stderr, flush=True)
    _emit_frames(args.frames, args.frame_byte)
    if args.hold or args.record:
        _consume_stdin(args.record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
