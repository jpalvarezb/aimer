from __future__ import annotations

import pytest
from duplex_bridge.providers.gemini_live import GeminiLiveSession
from duplex_bridge.session import DuplexSession


def test_gemini_live_session_implements_duplex_interface() -> None:
    assert issubclass(GeminiLiveSession, DuplexSession)


def test_gemini_live_session_constructor() -> None:
    """Verify GeminiLiveSession can be constructed with required model parameter."""
    session = GeminiLiveSession(model="gemini-2.0-flash-exp")
    assert session.model == "gemini-2.0-flash-exp"
    assert session.api_key_env == "GEMINI_API_KEY"
