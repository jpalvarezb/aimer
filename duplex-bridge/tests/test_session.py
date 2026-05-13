from __future__ import annotations

import pytest
from duplex_bridge.providers.gemini_live import GeminiLiveSession
from duplex_bridge.session import DuplexSession


def test_gemini_live_session_implements_duplex_interface() -> None:
    assert issubclass(GeminiLiveSession, DuplexSession)


async def test_gemini_live_session_is_week_3_stub() -> None:
    session = GeminiLiveSession()

    with pytest.raises(NotImplementedError):
        await session.open()
