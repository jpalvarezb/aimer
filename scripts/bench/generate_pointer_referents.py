"""Generate pointer_referent fixtures with the RUNTIME deixis resolver (Week 9).

Replaces the never-committed out-of-band step of the Week-4 decoupling experiment: the
``pointer_referent`` values in the injected fixture were produced by an ad hoc script that
never landed. This regenerates them with the exact component the bridge runs in production
— :class:`duplex_bridge.deixis.PointerReferentResolver` — so the bench validates the real
resolver, not a look-alike.

For each task in the base fixture, the resolver reads the task's cursor tile + AX hints
and the referent is written back as ``pointer_referent``; ``eval_deictic.py`` then injects
it as the ``pointer=`` annotation (its existing behavior when the field is present).

Usage:
    uv run python scripts/bench/generate_pointer_referents.py            # full 58-task set
    uv run python scripts/bench/generate_pointer_referents.py --limit 5  # smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))
sys.path.insert(0, str(_ROOT / "aimer-core" / "src"))

from duplex_bridge.deixis import PointerContext, PointerReferentResolver  # noqa: E402

DEFAULT_TASKS = _HERE / "fixtures" / "deictic_tasks_web_ax.jsonl"
DEFAULT_OUT = _HERE / "fixtures" / "deictic_tasks_web_ax_resolved.jsonl"


def _load_key() -> None:
    if os.environ.get("GEMINI_API_KEY"):
        return
    env_path = _ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY=") and "=" in line:
                os.environ["GEMINI_API_KEY"] = line.partition("=")[2].strip()
                return


async def _resolve_all(tasks: list[dict], model: str, parallel: int) -> list[str]:
    resolver = PointerReferentResolver(model=model)
    gate = asyncio.Semaphore(parallel)

    async def _one(task: dict) -> str:
        tile_path = _HERE / task["image_path"]
        context = PointerContext(
            app=task.get("app"),
            window_title=task.get("window_title"),
            accessibility_label=task.get("accessibility_label"),
            selected_text=task.get("selected_text"),
            cursor_tile_x=task.get("cursor_tile_x"),
            cursor_tile_y=task.get("cursor_tile_y"),
        )
        async with gate:
            return await resolver.resolve(tile_path.read_bytes(), context)

    return list(await asyncio.gather(*(_one(t) for t in tasks)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=6)
    args = parser.parse_args()

    _load_key()
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if args.limit:
        tasks = tasks[: args.limit]

    started = time.perf_counter()
    referents = asyncio.run(_resolve_all(tasks, args.model, args.parallel))
    elapsed = time.perf_counter() - started

    empty = 0
    with args.out.open("w") as fh:
        for task, referent in zip(tasks, referents, strict=True):
            record = dict(task)
            record["pointer_referent"] = referent
            record["pointer_resolver"] = args.model
            if not referent:
                empty += 1
            fh.write(json.dumps(record) + "\n")

    print(f"resolved {len(tasks)} tasks in {elapsed:.1f}s ({empty} empty referents)")
    print(f"-> {args.out}")
    for task, referent in list(zip(tasks, referents, strict=True))[:3]:
        print(f"  {task['id']}: {referent[:100]}")
    return 0 if empty < len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
