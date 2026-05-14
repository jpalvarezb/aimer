from __future__ import annotations

from duplex_bridge.providers.gemini_live import GeminiLiveSession
from duplex_bridge.session import DuplexSession


def test_gemini_live_session_implements_duplex_interface() -> None:
    assert issubclass(GeminiLiveSession, DuplexSession)


def test_gemini_live_session_constructor() -> None:
    """Verify GeminiLiveSession can be constructed with required model parameter."""
    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")
    assert session.model == "gemini-3.1-flash-live-preview"
    assert session.api_key_env == "GEMINI_API_KEY"
