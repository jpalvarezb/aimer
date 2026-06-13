"""Typed entity router — maps extracted entities to host-app targets and dispatches them.

This is the "routes them (Maps / Calendar / IDE)" half of Week 5. It is deliberately
separate from actuation: it turns a typed ``Entity`` into a concrete, target-specific
*intent* (a Maps URL, a calendar event title, an IDE symbol lookup) and hands it to a
registered handler. The Week-7 host-app actuators register the real handlers; tests and
the duplex bridge register lightweight ones. Routing decisions run off the duplex hot
path, before the audio turn reaches the duplex model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import quote

from aimer_core.schema import Entity, EntityType

RouteTarget = Literal["maps", "calendar", "ide", "none"]

# Default entity-type -> host-app target mapping. "product"/"unknown" have no host
# target yet (would be a shopping/search actuator) and route to "none".
DEFAULT_ROUTES: dict[EntityType, RouteTarget] = {
    "place": "maps",
    "date": "calendar",
    "todo": "calendar",
    "code_span": "ide",
    "product": "none",
    "unknown": "none",
}


@dataclass(frozen=True)
class RoutedAction:
    """A typed entity resolved to a host-app target plus an actionable payload."""

    target: RouteTarget
    entity: Entity
    payload: dict[str, str]


def _build_payload(target: RouteTarget, entity: Entity) -> dict[str, str]:
    """Turn an entity into a concrete, target-specific intent."""
    if target == "maps":
        return {"url": f"https://maps.apple.com/?q={quote(entity.value)}", "query": entity.value}
    if target == "calendar":
        # A minimal event intent; the Week-7 Calendar actuator fills date/time.
        return {"title": entity.value}
    if target == "ide":
        # "go to / explain this symbol" intent for the IDE actuator.
        return {"symbol": entity.value}
    return {}


class EntityRouter:
    """Routes typed entities to host-app targets and dispatches to registered handlers.

    ``handlers`` maps a target to a callback invoked with the ``RoutedAction``. Unset
    targets simply produce a ``RoutedAction`` with no side effect (useful for tests and
    for staging actions before actuators exist).
    """

    def __init__(
        self,
        routes: dict[EntityType, RouteTarget] | None = None,
        handlers: dict[RouteTarget, Callable[[RoutedAction], None]] | None = None,
    ) -> None:
        self._routes: dict[EntityType, RouteTarget] = {**DEFAULT_ROUTES, **(routes or {})}
        self._handlers: dict[RouteTarget, Callable[[RoutedAction], None]] = dict(handlers or {})

    def register(self, target: RouteTarget, handler: Callable[[RoutedAction], None]) -> None:
        """Register (or replace) the handler invoked when an action routes to ``target``."""
        self._handlers[target] = handler

    def route_one(self, entity: Entity) -> RoutedAction | None:
        """Resolve a single entity to a RoutedAction, or None if it has no host target."""
        target = self._routes.get(entity.type, "none")
        if target == "none":
            return None
        return RoutedAction(target=target, entity=entity, payload=_build_payload(target, entity))

    def dispatch(self, entities: list[Entity]) -> list[RoutedAction]:
        """Route every entity and invoke the matching handler. Returns the routed actions."""
        actions: list[RoutedAction] = []
        for entity in entities:
            action = self.route_one(entity)
            if action is None:
                continue
            handler = self._handlers.get(action.target)
            if handler is not None:
                handler(action)
            actions.append(action)
        return actions


@dataclass
class RecordingHandler:
    """A handler that records the actions it received — for tests and dry-run wiring."""

    received: list[RoutedAction] = field(default_factory=list)

    def __call__(self, action: RoutedAction) -> None:
        self.received.append(action)
