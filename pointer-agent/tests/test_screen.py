from __future__ import annotations

from types import SimpleNamespace

import pytest
from aimer_core import BoundingBox, CursorPosition
from pointer_agent.capture.macos import screen
from pointer_agent.capture.macos.screen import _marker_point_px

JPEG_BYTES = b"\xff\xd8fake-jpeg"


@pytest.fixture(autouse=True)
def reset_screen_state() -> None:
    screen._shareable_content = None
    screen._shareable_content_ts = 0.0
    screen._warned_once = False


class FakeData(bytearray):
    @classmethod
    def data(cls) -> FakeData:
        return cls()


class FakeDisplay:
    def displayID(self) -> int:
        return 0

    def width(self) -> int:
        return 512

    def height(self) -> int:
        return 512


class FakeContent:
    def displays(self) -> list[FakeDisplay]:
        return [FakeDisplay()]


class FakeImage:
    def __init__(self, width: int = 512, height: int = 512) -> None:
        self.width = width
        self.height = height


class FakeError:
    def domain(self) -> str:
        return "SCStreamErrorDomain"

    def __str__(self) -> str:
        return "Screen Recording denied"


class FakeFilter:
    @classmethod
    def alloc(cls) -> FakeFilter:
        return cls()

    def initWithDisplay_excludingWindows_(
        self,
        display: FakeDisplay,
        windows: list[object],
    ) -> FakeFilter:
        self.display = display
        self.windows = windows
        return self


class FakeConfig:
    @classmethod
    def alloc(cls) -> FakeConfig:
        return cls()

    def init(self) -> FakeConfig:
        return self

    def setSourceRect_(self, value: object) -> None:
        self.sourceRect = value

    def setWidth_(self, value: int) -> None:
        self.width = value

    def setHeight_(self, value: int) -> None:
        self.height = value


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    image: FakeImage | None = None,
    error: FakeError | None = None,
) -> None:
    monkeypatch.setattr(screen, "_screen_capture_kit_available", lambda: True)
    monkeypatch.setattr(
        screen,
        "_display_bounds_in_points",
        lambda _screen_id, _display_scale, _display=None: BoundingBox(
            x=0.0,
            y=0.0,
            width=1_000.0,
            height=800.0,
        ),
    )

    class FakeShareableContent:
        @staticmethod
        def getShareableContentWithCompletionHandler_(handler: object) -> None:
            handler(FakeContent(), None)

    class FakeScreenshotManager:
        @staticmethod
        def captureImageWithFilter_configuration_completionHandler_(
            _filt: object,
            _cfg: object,
            handler: object,
        ) -> None:
            handler(image if image is not None else FakeImage(), error)

    fake_sck = SimpleNamespace(
        SCContentFilter=FakeFilter,
        SCShareableContent=FakeShareableContent,
        SCScreenshotManager=FakeScreenshotManager,
        SCStreamConfiguration=FakeConfig,
    )
    fake_quartz = SimpleNamespace(
        CGRectMake=lambda x, y, width, height: (x, y, width, height),
        CGImageGetWidth=lambda cg_image: cg_image.width,
        CGImageGetHeight=lambda cg_image: cg_image.height,
        CGColorSpaceCreateDeviceRGB=lambda: object(),
        CGBitmapContextCreate=lambda *_args: {},
        CGContextSetInterpolationQuality=lambda *_args: None,
        CGContextDrawImage=lambda *_args: None,
        CGBitmapContextCreateImage=lambda _context: FakeImage(256, 256),
        CGContextSetRGBStrokeColor=lambda *_args, **_kwargs: None,
        CGContextSetLineWidth=lambda *_args, **_kwargs: None,
        CGContextAddArc=lambda *_args, **_kwargs: None,
        CGContextStrokePath=lambda *_args, **_kwargs: None,
        CGImageDestinationCreateWithData=lambda data, *_args: data,
        CGImageDestinationAddImage=lambda dest, *_args: dest.extend(JPEG_BYTES),
        CGImageDestinationFinalize=lambda _dest: True,
        kCGImageAlphaPremultipliedLast=1,
        kCGInterpolationHigh=1,
        kCGImageDestinationLossyCompressionQuality="quality",
    )
    fake_foundation = SimpleNamespace(NSMutableData=FakeData)

    monkeypatch.setitem(__import__("sys").modules, "ScreenCaptureKit", fake_sck)
    monkeypatch.setitem(__import__("sys").modules, "Quartz", fake_quartz)
    monkeypatch.setitem(__import__("sys").modules, "Foundation", fake_foundation)


def test_capture_tile_returns_jpeg_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)

    tile = screen._capture_tile(CursorPosition(x=500.0, y=500.0, screen_id=0), 2.0)

    assert tile is not None
    assert tile.startswith(b"\xff\xd8")


def test_capture_hover_region_returns_hover_region(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)

    region = screen.capture_hover_region(CursorPosition(x=500.0, y=500.0, screen_id=0), 2.0)

    assert region is not None
    assert region.type == "unknown"
    assert region.bbox == BoundingBox(x=372.0, y=372.0, width=256.0, height=256.0)
    assert region.tile_b64


def test_capture_hover_region_clamps_to_display_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)

    region = screen.capture_hover_region(CursorPosition(x=10.0, y=10.0, screen_id=0), 2.0)

    assert region is not None
    assert region.bbox is not None
    assert region.bbox.x == 0.0
    assert region.bbox.y == 0.0


def test_permission_denied_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch, image=None, error=FakeError())

    region = screen.capture_hover_region(CursorPosition(x=500.0, y=500.0, screen_id=0), 2.0)

    assert region is None


# ---------------------------------------------------------------------------
# _marker_point_px — pure math, no Quartz needed
# ---------------------------------------------------------------------------


def test_marker_point_px_y_flip() -> None:
    """Quartz y is height_px - off_y_pt * scale (origin bottom-left)."""
    cursor = CursorPosition(x=300.0, y=200.0, screen_id=0)
    bbox = BoundingBox(x=172.0, y=72.0, width=256.0, height=256.0)
    display_scale = 2.0
    width_px = height_px = 512

    qx, qy = _marker_point_px(cursor, bbox, display_scale, width_px, height_px)

    # off_x_pt = 300 - 172 = 128; qx = 128 * 2 = 256
    assert qx == pytest.approx(256.0)
    # off_y_pt = 200 - 72 = 128; qy = 512 - 128 * 2 = 256
    assert qy == pytest.approx(256.0)


def test_marker_point_px_edge_clamped_cursor() -> None:
    """When bbox is clamped to screen edge, cursor offset != size/2."""
    # Cursor is at (10, 10) — near top-left corner; bbox is clamped to (0, 0)
    cursor = CursorPosition(x=10.0, y=10.0, screen_id=0)
    bbox = BoundingBox(x=0.0, y=0.0, width=256.0, height=256.0)
    display_scale = 2.0
    width_px = height_px = 512

    qx, qy = _marker_point_px(cursor, bbox, display_scale, width_px, height_px)

    # off_x_pt = 10 - 0 = 10; qx = 10 * 2 = 20 (NOT 256 = size/2)
    assert qx == pytest.approx(20.0)
    assert qx != pytest.approx(width_px / 2)
    # off_y_pt = 10 - 0 = 10; qy = 512 - 10 * 2 = 492 (NOT 256)
    assert qy == pytest.approx(492.0)
    assert qy != pytest.approx(height_px / 2)


# ---------------------------------------------------------------------------
# _draw_cursor_marker — uses fake Quartz
# ---------------------------------------------------------------------------


def test_draw_cursor_marker_returns_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """_draw_cursor_marker returns a non-None image (fake CGBitmapContextCreateImage sentinel)."""
    _install_fakes(monkeypatch)

    fake_input = FakeImage(512, 512)
    result = screen._draw_cursor_marker(fake_input, 256.0, 256.0, 512, 512)

    assert result is not None


# ---------------------------------------------------------------------------
# cursor_tile_x/y populated by capture_hover_region
# ---------------------------------------------------------------------------


def test_capture_hover_region_populates_cursor_tile_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cursor_tile_x/y == cursor - bbox.origin (logical points)."""
    _install_fakes(monkeypatch)

    # Cursor at (500, 500); bbox for display 1000x800 = (372, 372, 256, 256)
    cursor = CursorPosition(x=500.0, y=500.0, screen_id=0)
    region = screen.capture_hover_region(cursor, 2.0)

    assert region is not None
    assert region.cursor_tile_x == pytest.approx(500.0 - region.bbox.x)
    assert region.cursor_tile_y == pytest.approx(500.0 - region.bbox.y)


def test_capture_hover_region_cursor_tile_offsets_at_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When bbox is clamped, cursor offsets are still computed correctly."""
    _install_fakes(monkeypatch)

    cursor = CursorPosition(x=10.0, y=10.0, screen_id=0)
    region = screen.capture_hover_region(cursor, 2.0)

    assert region is not None
    assert region.bbox.x == 0.0
    assert region.bbox.y == 0.0
    # cursor.x - bbox.x = 10 - 0 = 10
    assert region.cursor_tile_x == pytest.approx(10.0)
    assert region.cursor_tile_y == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# capture_full_frame — basic smoke test
# ---------------------------------------------------------------------------


def test_capture_full_frame_returns_full_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """capture_full_frame returns a FullFrame with expected downscaled dims."""
    # Display is 1000x800 logical points (from _install_fakes bounds helper).
    # long_edge_px=1000: landscape, so out_w=1000, out_h=round(1000/(1000/800))=800
    _install_fakes(monkeypatch)

    cursor = CursorPosition(x=500.0, y=400.0, screen_id=0)
    frame = screen.capture_full_frame(cursor, display_scale=2.0, long_edge_px=1000)

    assert frame is not None
    assert frame.width_px == 1000
    assert frame.height_px == 800
    assert frame.frame_b64  # non-empty base64
    # cursor_frame offsets are logical-pt distance from display origin
    assert frame.cursor_frame_x == pytest.approx(500.0)  # cursor.x - bounds.x (bounds.x=0)
    assert frame.cursor_frame_y == pytest.approx(400.0)


def test_capture_full_frame_returns_none_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """capture_full_frame returns None when capture returns an error."""
    _install_fakes(monkeypatch, image=None, error=FakeError())

    cursor = CursorPosition(x=500.0, y=400.0, screen_id=0)
    frame = screen.capture_full_frame(cursor, display_scale=2.0)

    assert frame is None
