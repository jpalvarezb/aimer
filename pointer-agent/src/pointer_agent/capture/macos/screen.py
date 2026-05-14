"""Uses ScreenCaptureKit (macOS 14+).

Falls back to None on older macOS or when Screen Recording permission is not granted.
"""

from __future__ import annotations

import base64
import logging
import platform
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from aimer_core import BoundingBox, CursorPosition, HoverRegion

_REQUIRED_MACOS = (14, 0)
_SHAREABLE_CONTENT_TTL_S = 5.0
_SHAREABLE_CONTENT_TIMEOUT_S = 1.0
_CAPTURE_TIMEOUT_S = 0.1
_TILE_SIZE = 256
_JPEG_QUALITY = 0.8

_warned_once = False
_shareable_content: Any | None = None
_shareable_content_ts = 0.0

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
        parts = [int(part) for part in version.split(".") if part]
        major = parts[0]
        minor = parts[1] if len(parts) > 1 else 0
    except (IndexError, ValueError):
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


def capture_hover_region(
    cursor: CursorPosition,
    display_scale: float = 1.0,
) -> HoverRegion | None:
    """Return the cursor-adjacent hover region, if pixel capture is enabled."""

    if not _screen_capture_kit_available():
        return None

    tile = _capture_tile(cursor, display_scale)
    if tile is None:
        return None

    bbox = _tile_bbox(cursor, display_scale=display_scale)
    tile_b64 = base64.b64encode(tile).decode("ascii")
    return HoverRegion(type="unknown", bbox=bbox, tile_b64=tile_b64)


def _capture_tile(
    cursor: CursorPosition,
    display_scale: float,
    size: int = _TILE_SIZE,
) -> bytes | None:
    if not _screen_capture_kit_available():
        return None

    try:
        import ScreenCaptureKit
    except ImportError:
        _warn_once("ScreenCaptureKit unavailable: pyobjc ScreenCaptureKit import failed")
        return None

    content = _shareable_content_for_capture(ScreenCaptureKit)
    display = _display_for_cursor(content, cursor.screen_id) if content is not None else None
    if display is None:
        _warn_once("ScreenCaptureKit unavailable: no capture display found")
        return None

    bbox = _tile_bbox(cursor, display_scale=display_scale, display=display, size=size)
    filt = ScreenCaptureKit.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
        display,
        [],
    )
    cfg = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
    _set_objc_value(cfg, "sourceRect", _cg_rect(bbox.x, bbox.y, bbox.width, bbox.height))
    _set_objc_value(cfg, "width", int(size * max(display_scale, 1.0)))
    _set_objc_value(cfg, "height", int(size * max(display_scale, 1.0)))

    cg_image = _capture_image(ScreenCaptureKit, filt, cfg)
    if cg_image is None:
        return None

    cg_image = _downsample_image(cg_image, size)
    return _encode_jpeg(cg_image)


def _shareable_content_for_capture(screen_capture_kit: Any) -> Any | None:
    global _shareable_content, _shareable_content_ts

    now = time.monotonic()
    if _shareable_content is not None and (now - _shareable_content_ts) < _SHAREABLE_CONTENT_TTL_S:
        return _shareable_content

    fut: Future[tuple[Any, Any]] = Future()

    def handler(content: Any, error: Any) -> None:
        if not fut.done():
            fut.set_result((content, error))

    try:
        screen_capture_kit.SCShareableContent.getShareableContentWithCompletionHandler_(
            handler,
        )
        content, error = fut.result(timeout=_SHAREABLE_CONTENT_TIMEOUT_S)
    except FutureTimeoutError:
        _warn_once("ScreenCaptureKit unavailable: shareable content lookup timed out")
        return None
    except Exception as exc:
        _warn_once(f"ScreenCaptureKit unavailable: shareable content lookup failed ({exc})")
        return None

    if error is not None:
        _warn_once(f"ScreenCaptureKit unavailable: shareable content lookup failed ({error})")
        return None
    if content is None:
        _warn_once("ScreenCaptureKit unavailable: shareable content lookup returned no content")
        return None

    _shareable_content = content
    _shareable_content_ts = now
    return content


def _capture_image(screen_capture_kit: Any, filt: Any, cfg: Any) -> Any | None:
    fut: Future[tuple[Any, Any]] = Future()

    def handler(image: Any, error: Any) -> None:
        if not fut.done():
            fut.set_result((image, error))

    try:
        screen_capture_kit.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
            filt,
            cfg,
            handler,
        )
        image, error = fut.result(timeout=_CAPTURE_TIMEOUT_S)
    except FutureTimeoutError:
        _warn_once("ScreenCaptureKit capture timed out")
        return None
    except Exception as exc:
        _warn_once(f"ScreenCaptureKit capture failed ({exc})")
        return None

    if error is not None:
        _warn_once(_capture_error_message(error))
        return None
    if image is None:
        _warn_once("ScreenCaptureKit capture returned no image")
        return None

    return image


def _capture_error_message(error: Any) -> str:
    domain = _objc_value(error, "domain")
    if domain == "SCStreamErrorDomain":
        return "ScreenCaptureKit capture denied: grant Screen Recording permission"
    return f"ScreenCaptureKit capture failed ({error})"


def _display_for_cursor(content: Any, screen_id: int) -> Any | None:
    displays = list(_objc_value(content, "displays", []) or [])
    for display in displays:
        display_id = _objc_value(display, "displayID")
        if display_id is not None and int(display_id) == screen_id:
            return display
    return None


def _tile_bbox(
    cursor: CursorPosition,
    *,
    display_scale: float,
    display: Any | None = None,
    size: int = _TILE_SIZE,
) -> BoundingBox:
    bounds = _display_bounds_in_points(cursor.screen_id, display_scale, display)
    width = min(float(size), bounds.width) if bounds.width > 0 else float(size)
    height = min(float(size), bounds.height) if bounds.height > 0 else float(size)
    x = cursor.x - (width / 2.0)
    y = cursor.y - (height / 2.0)

    if bounds.width > 0:
        x = min(max(x, bounds.x), bounds.x + bounds.width - width)
    if bounds.height > 0:
        y = min(max(y, bounds.y), bounds.y + bounds.height - height)

    return BoundingBox(x=x, y=y, width=width, height=height)


def _display_bounds_in_points(
    screen_id: int,
    display_scale: float,
    display: Any | None = None,
) -> BoundingBox:
    try:
        from AppKit import NSScreen

        for screen in NSScreen.screens():
            if int(screen.deviceDescription()["NSScreenNumber"]) == screen_id:
                frame = screen.frame()
                return BoundingBox(
                    x=float(frame.origin.x),
                    y=float(frame.origin.y),
                    width=float(frame.size.width),
                    height=float(frame.size.height),
                )
    except Exception:
        pass

    if display is not None:
        width = _objc_value(display, "width")
        height = _objc_value(display, "height")
        if width is not None and height is not None:
            scale = max(display_scale, 1.0)
            return BoundingBox(
                x=0.0,
                y=0.0,
                width=float(width) / scale,
                height=float(height) / scale,
            )

    return BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0)


def _cg_rect(x: float, y: float, width: float, height: float) -> Any:
    import Quartz

    return Quartz.CGRectMake(x, y, width, height)


def _downsample_image(cg_image: Any, size: int) -> Any:
    import Quartz

    width, height = _cg_image_size(cg_image, Quartz)
    if width == size and height == size:
        return cg_image

    color_space = Quartz.CGColorSpaceCreateDeviceRGB()
    alpha_info = getattr(Quartz, "kCGImageAlphaPremultipliedLast", 0)
    context = Quartz.CGBitmapContextCreate(None, size, size, 8, 0, color_space, alpha_info)
    if context is None:
        _warn_once("ScreenCaptureKit downsample failed: bitmap context unavailable")
        return cg_image

    interpolation = getattr(Quartz, "kCGInterpolationHigh", 0)
    if hasattr(Quartz, "CGContextSetInterpolationQuality"):
        Quartz.CGContextSetInterpolationQuality(context, interpolation)
    Quartz.CGContextDrawImage(context, Quartz.CGRectMake(0, 0, size, size), cg_image)
    downsampled = Quartz.CGBitmapContextCreateImage(context)
    return downsampled if downsampled is not None else cg_image


def _encode_jpeg(cg_image: Any) -> bytes | None:
    try:
        import Quartz
        from Foundation import NSMutableData  # type: ignore[import-not-found]

        data = NSMutableData.data()
        dest = Quartz.CGImageDestinationCreateWithData(data, "public.jpeg", 1, None)
        if dest is None:
            _warn_once("ScreenCaptureKit JPEG encode failed: destination unavailable")
            return None

        quality_key = getattr(
            Quartz,
            "kCGImageDestinationLossyCompressionQuality",
            "kCGImageDestinationLossyCompressionQuality",
        )
        Quartz.CGImageDestinationAddImage(dest, cg_image, {quality_key: _JPEG_QUALITY})
        if not Quartz.CGImageDestinationFinalize(dest):
            _warn_once("ScreenCaptureKit JPEG encode failed: finalize returned false")
            return None
        return bytes(data)
    except Exception as exc:
        _warn_once(f"ScreenCaptureKit JPEG encode failed ({exc})")
        return None


def _cg_image_size(cg_image: Any, quartz: Any) -> tuple[int, int]:
    if hasattr(quartz, "CGImageGetWidth") and hasattr(quartz, "CGImageGetHeight"):
        return int(quartz.CGImageGetWidth(cg_image)), int(quartz.CGImageGetHeight(cg_image))
    return int(_objc_value(cg_image, "width", 0)), int(_objc_value(cg_image, "height", 0))


def _set_objc_value(obj: Any, name: str, value: Any) -> None:
    setter_name = f"set{name[0].upper()}{name[1:]}_"
    setter = getattr(obj, setter_name, None)
    if setter is not None:
        setter(value)
        return
    setattr(obj, name, value)


def _objc_value(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    return value() if callable(value) else value
