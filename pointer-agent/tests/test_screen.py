from __future__ import annotations

from types import SimpleNamespace

import pytest
from aimer_core import BoundingBox, CursorPosition
from pointer_agent.capture.macos import screen

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
