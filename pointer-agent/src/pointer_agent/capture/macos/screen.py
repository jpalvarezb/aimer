"""Uses ScreenCaptureKit (macOS 14+).

Falls back to None on older macOS or when Screen Recording permission is not granted.
"""

from __future__ import annotations

import logging
import platform

from aimer_core import CursorPosition, HoverRegion

_REQUIRED_MACOS = (14, 0)
_warned_once = False

logger = logging.getLogger(__name__)


def _warn_once(message: str) -> None:
    global _warned_once

    if not _warned_once:
        logger.warning(message)
        _warned_once = True


def _screen_capture_kit_available() -> bool:
    """Return True when the runtime can use ScreenCaptureKit."""

    version = platform.mac_ver()[0]
    try:
        major, minor, *_ = (int(part) for part in version.split("."))
    except ValueError:
        _warn_once("ScreenCaptureKit unavailable: unable to determine macOS version")
        return False

    if (major, minor) < _REQUIRED_MACOS:
        _warn_once("ScreenCaptureKit unavailable: macOS 14.0 or newer is required")
        return False

    try:
        import ScreenCaptureKit  # noqa: F401
    except ImportError:
        _warn_once("ScreenCaptureKit unavailable: pyobjc ScreenCaptureKit import failed")
        return False

    return True


def capture_hover_region(_cursor: CursorPosition) -> HoverRegion | None:
    """Return the cursor-adjacent hover region, if pixel capture is enabled."""

    if not _screen_capture_kit_available():
        return None

    # Week 2: implement SCScreenshotManager.captureImage
    return None


# Week 2 notes:
"""
SCK async calls:
SCScreenshotManager.captureImageWithFilter:configuration:completionHandler:
is async/block-based. In Python, wrap it with a concurrent.futures.Future:

    fut = Future()
    def handler(image, error): fut.set_result((image, error))
    SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
        filt, cfg, handler
    )
    image, error = fut.result(timeout=0.1)

Do not use dispatch_semaphore from Python; PyObjC futures are cleaner.

SCContentFilter construction:
SCContentFilter needs SCShareableContent, and
SCShareableContent.getShareableContentWithCompletionHandler_ is also async.
Cache the shareable content in a module-level singleton refreshed every 5 seconds;
do not call it per frame. For Week 2, capture the full display with:

    SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])

No CGWindowList fallback:
If _screen_capture_kit_available() is False, return None. Do not silently fall
back to CGWindowListCreateImage; this module deliberately keeps one SCK path.

Permission denial:
SCK returns a None image with an NSError whose domain is SCStreamErrorDomain.
Catch that, log once, and return None, matching the version/import guard behavior.

JPEG encoding:
SCK returns a CGImage. Convert to JPEG bytes through Quartz CGImageDestination
at quality 0.8:

    data = NSMutableData.data()
    dest = CGImageDestinationCreateWithData(data, "public.jpeg", 1, None)
    CGImageDestinationAddImage(
        dest,
        cg_image,
        {"kCGImageDestinationLossyCompressionQuality": 0.8},
    )
    CGImageDestinationFinalize(dest)
    return bytes(data)

Tile coordinate handling:
For a 256x256-point region, pass a sourceRect in logical points centered on the
cursor. SCK handles scale conversion, so the returned CGImage will be
256 * display_scale pixels wide. Downsample that CGImage to exactly 256x256
pixels before JPEG encoding using CGContext at 256x256 with a scale transform.
That keeps the wire payload bounded and lets bbox.width match tile pixel width
for consumers that do not want to do scale math.
"""
