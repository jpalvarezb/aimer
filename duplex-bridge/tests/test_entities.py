"""Week-5 entity extraction: seam, router, and off-hot-path pipeline (no model needed)."""

from __future__ import annotations

import asyncio
import base64
import time

from aimer_core.schema import (
    ContextPacket,
    CursorPosition,
    Entity,
    FocusWindow,
    HoverRegion,
    SemanticContext,
)
from duplex_bridge.entities import (
    DEFAULT_ROUTES,
    EntityPipeline,
    EntityRouter,
    ExtractionContext,
    MockExtractor,
    RecordingHandler,
    context_from_packet,
    tile_bytes_from_packet,
)
from duplex_bridge.entities._extract_common import build_prompt, parse_entities
from duplex_bridge.session import DuplexSession


def _packet_with_tile(tile: bytes = b"jpeg-bytes") -> ContextPacket:
    return ContextPacket(
        cursor=CursorPosition(x=10, y=20),
        focus_window=FocusWindow(app="Chrome", title="Maps"),
        semantic=SemanticContext(accessibility_label="1 Infinite Loop", selected_text=None),
        hover_region=HoverRegion(tile_b64=base64.b64encode(tile).decode()),
    )


# --- router ---------------------------------------------------------------


def test_default_routes_cover_targets():
    assert DEFAULT_ROUTES["place"] == "maps"
    assert DEFAULT_ROUTES["date"] == "calendar"
    assert DEFAULT_ROUTES["todo"] == "calendar"
    assert DEFAULT_ROUTES["code_span"] == "ide"
    assert DEFAULT_ROUTES["product"] == "none"
    assert DEFAULT_ROUTES["unknown"] == "none"


def test_route_payloads_are_actionable():
    r = EntityRouter()
    place = r.route_one(Entity(type="place", value="1600 Amphitheatre Pkwy"))
    assert place is not None and place.target == "maps"
    assert "maps.apple.com" in place.payload["url"]
    assert "Amphitheatre" in place.payload["url"]  # query is URL-encoded into the link

    date = r.route_one(Entity(type="date", value="March 14, 2026"))
    assert date is not None and date.target == "calendar"
    assert date.payload["title"] == "March 14, 2026"

    code = r.route_one(Entity(type="code_span", value="flex-grow"))
    assert code is not None and code.target == "ide"
    assert code.payload["symbol"] == "flex-grow"


def test_unrouted_types_return_none():
    r = EntityRouter()
    assert r.route_one(Entity(type="product", value="iPhone")) is None
    assert r.route_one(Entity(type="unknown", value="???")) is None


def test_dispatch_invokes_registered_handlers():
    r = EntityRouter()
    maps_h, cal_h, ide_h = RecordingHandler(), RecordingHandler(), RecordingHandler()
    r.register("maps", maps_h)
    r.register("calendar", cal_h)
    r.register("ide", ide_h)
    actions = r.dispatch(
        [
            Entity(type="place", value="Mountain View"),
            Entity(type="date", value="tomorrow"),
            Entity(type="code_span", value="async def"),
            Entity(type="product", value="ignored"),  # routes to none -> no action
        ]
    )
    assert [a.target for a in actions] == ["maps", "calendar", "ide"]
    assert len(maps_h.received) == 1
    assert len(cal_h.received) == 1
    assert len(ide_h.received) == 1


# --- pipeline -------------------------------------------------------------


async def test_pipeline_process_tile_extracts_and_routes():
    extractor = MockExtractor([Entity(type="place", value="Mountain View")])
    handler = RecordingHandler()
    router = EntityRouter(handlers={"maps": handler})
    pipe = EntityPipeline(extractor, router)

    entities, actions = await pipe.process_tile(b"jpeg")
    assert [e.value for e in entities] == ["Mountain View"]
    assert [a.target for a in actions] == ["maps"]
    assert len(handler.received) == 1


async def test_pipeline_process_packet_populates_extracted_entities():
    extractor = MockExtractor([Entity(type="code_span", value="flex-grow")])
    pipe = EntityPipeline(extractor)
    packet = _packet_with_tile()
    entities, actions = await pipe.process_packet(packet)
    assert packet.extracted_entities == entities
    assert entities[0].type == "code_span"
    assert [a.target for a in actions] == ["ide"]


async def test_pipeline_no_tile_is_noop():
    pipe = EntityPipeline(MockExtractor([Entity(type="place", value="x")]))
    packet = ContextPacket(cursor=CursorPosition(x=0, y=0))  # no hover_region
    entities, actions = await pipe.process_packet(packet)
    assert entities == [] and actions == []


async def test_schedule_runs_off_hot_path_without_blocking():
    # A slow extractor must not block the caller: schedule() returns immediately and the
    # "audio tick" loop keeps running while extraction proceeds in the background.
    class SlowExtractor(MockExtractor):
        async def extract(self, tile_jpeg, context=None):
            await asyncio.sleep(0.2)
            return await super().extract(tile_jpeg, context)

    pipe = EntityPipeline(SlowExtractor([Entity(type="place", value="MV")]))
    packet = _packet_with_tile()

    t0 = time.perf_counter()
    task = pipe.schedule(packet)
    # schedule must return effectively immediately (does not await the 0.2s extraction)
    assert (time.perf_counter() - t0) < 0.05

    ticks = 0
    while not task.done():
        await asyncio.sleep(0.01)  # simulate the audio tick continuing to run
        ticks += 1
    entities, actions = await task
    assert ticks > 5  # the loop kept ticking during background extraction
    assert entities[0].value == "MV"
    assert actions[0].target == "maps"


# --- projection + parsing -------------------------------------------------


def test_context_from_packet_projects_hints():
    ctx = context_from_packet(_packet_with_tile())
    assert ctx.app == "Chrome"
    assert ctx.window_title == "Maps"
    assert ctx.accessibility_label == "1 Infinite Loop"


def test_tile_bytes_roundtrip():
    packet = _packet_with_tile(b"\xff\xd8\xff-tile")
    assert tile_bytes_from_packet(packet) == b"\xff\xd8\xff-tile"


def test_parse_entities_tolerates_fences_and_noise():
    raw = (
        'Here you go:\n```json\n[{"type":"place","value":"Mountain View"},'
        '{"type":"bogus","value":"x"},{"value":"no type"},{"type":"date"}]\n```'
    )
    ents = parse_entities(raw)
    # bogus type -> unknown; missing value -> dropped; missing type -> unknown
    assert [(e.type, e.value) for e in ents] == [
        ("place", "Mountain View"),
        ("unknown", "x"),
        ("unknown", "no type"),
    ]


def test_parse_entities_empty_on_garbage():
    assert parse_entities("no json here") == []
    assert parse_entities("") == []


def test_build_prompt_includes_context_hints():
    p = build_prompt(ExtractionContext(app="Safari", accessibility_label="Send"))
    assert "app=Safari" in p
    assert "ax=Send" in p
    assert "JSON array" in p


# --- bridge wiring: extraction runs off the hot path, throttled ----------


class _StubSession(DuplexSession):
    async def open(self):
        pass

    async def send_audio(self, frames: bytes):
        pass

    async def send_visual_context(self, packet):
        pass

    def on_audio_out(self, callback):
        pass

    def on_tool_call(self, callback):
        pass

    async def close(self):
        pass


def _packet_ctx(ax: str) -> ContextPacket:
    return ContextPacket(
        cursor=CursorPosition(x=1, y=2),
        focus_window=FocusWindow(app="Chrome", title="t"),
        semantic=SemanticContext(accessibility_label=ax),
        hover_region=HoverRegion(tile_b64=base64.b64encode(b"jpeg").decode()),
    )


async def test_server_schedules_entities_and_populates_packet():
    from duplex_bridge.server import WebSocketContextServer

    extractor = MockExtractor([Entity(type="place", value="Mountain View")])
    handler = RecordingHandler()
    pipe = EntityPipeline(extractor, EntityRouter(handlers={"maps": handler}))
    server = WebSocketContextServer(session=_StubSession(), entity_pipeline=pipe)

    packet = _packet_ctx("addr A")
    server._maybe_schedule_entities(packet)
    assert server._entity_task is not None
    await server._entity_task
    assert packet.extracted_entities[0].value == "Mountain View"
    assert len(handler.received) == 1


async def test_server_dedupes_unchanged_context():
    extractor = MockExtractor([Entity(type="place", value="MV")])
    server_pkg = __import__("duplex_bridge.server", fromlist=["WebSocketContextServer"])
    server = server_pkg.WebSocketContextServer(
        session=_StubSession(), entity_pipeline=EntityPipeline(extractor)
    )
    p1 = _packet_ctx("same")
    server._maybe_schedule_entities(p1)
    await server._entity_task
    # identical context -> no new extraction
    server._maybe_schedule_entities(_packet_ctx("same"))
    assert extractor.calls == 1
    # changed context -> extracts again
    server._maybe_schedule_entities(_packet_ctx("different"))
    await server._entity_task
    assert extractor.calls == 2


async def test_server_inflight_guard_skips_pile_up():
    class SlowExtractor(MockExtractor):
        async def extract(self, tile_jpeg, context=None):
            await asyncio.sleep(0.15)
            return await super().extract(tile_jpeg, context)

    extractor = SlowExtractor([Entity(type="place", value="MV")])
    server_pkg = __import__("duplex_bridge.server", fromlist=["WebSocketContextServer"])
    server = server_pkg.WebSocketContextServer(
        session=_StubSession(), entity_pipeline=EntityPipeline(extractor)
    )
    server._maybe_schedule_entities(_packet_ctx("a"))
    first = server._entity_task
    # a different context while the first is still running must NOT pile up a second task
    server._maybe_schedule_entities(_packet_ctx("b"))
    assert server._entity_task is first
    await first
    assert extractor.calls == 1
