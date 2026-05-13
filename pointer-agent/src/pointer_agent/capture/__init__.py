"""Platform capture provider dispatch."""

from __future__ import annotations

import sys

if sys.platform == "darwin":
    from pointer_agent.capture.macos import MacOSCaptureProvider as PlatformCaptureProvider
elif sys.platform == "win32":
    from pointer_agent.capture.windows import WindowsCaptureProvider as PlatformCaptureProvider
elif sys.platform.startswith("linux"):
    from pointer_agent.capture.linux import LinuxCaptureProvider as PlatformCaptureProvider
else:
    msg = f"Unsupported platform: {sys.platform}"
    raise RuntimeError(msg)

__all__ = ["PlatformCaptureProvider"]
