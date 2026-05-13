from __future__ import annotations

from aimer_core import ContextPacket, CursorPosition
from pointer_agent.capture.base import CaptureProvider
from pointer_agent.telemetry import run_telemetry


class FakeCaptureProvider(CaptureProvider):
    def capture(self) -> ContextPacket:
        return ContextPacket(cursor=CursorPosition(x=1, y=2, screen_id=3))


async def test_run_telemetry_respects_limit() -> None:
    packets: list[ContextPacket] = []

    async def sink(packet: ContextPacket) -> None:
        packets.append(packet)

    await run_telemetry(FakeCaptureProvider(), interval_hz=1_000, sink=sink, limit=3)

    assert len(packets) == 3
    assert packets[0].cursor.screen_id == 3
