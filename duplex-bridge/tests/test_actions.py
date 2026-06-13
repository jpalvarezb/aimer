"""Week-7 host actions: IDE async-rewrite + Chrome compare, end-to-end through the worker."""

from __future__ import annotations

import ast
from pathlib import Path

from duplex_bridge.actions import (
    TOOL_DECLARATIONS,
    compare_products,
    rewrite_function_async,
)
from duplex_bridge.worker import BackgroundWorker, JobResult, ToolDispatcher


def _fake_fetch(name: str) -> dict:
    return {
        "title": name,
        "summary": f"{name} is a product.",
        "url": f"https://x/{name}",
        "thumbnail": None,
    }


_SYNC_SRC = """import time
import requests


def fetch_user(user_id):
    resp = requests.get(f"https://api.example.com/users/{user_id}")
    time.sleep(0.1)
    return resp.json()


def untouched():
    return 42
"""


# --- IDE async rewrite ----------------------------------------------------


def test_ide_deterministic_rewrite_makes_function_async(tmp_path):
    f = tmp_path / "svc.py"
    f.write_text(_SYNC_SRC)
    r = rewrite_function_async(f, "fetch_user")
    assert r.applied and r.is_async and r.error is None
    tree = ast.parse(f.read_text())
    funcs = {
        n.name: type(n).__name__ for n in tree.body if isinstance(n, ast.AST) and hasattr(n, "name")
    }
    assert funcs.get("fetch_user") == "AsyncFunctionDef"
    assert funcs.get("untouched") == "FunctionDef"  # siblings untouched


def test_ide_model_provided_rewrite_is_applied(tmp_path):
    f = tmp_path / "svc.py"
    f.write_text(_SYNC_SRC)
    new = (
        "async def fetch_user(user_id):\n"
        "    async with httpx.AsyncClient() as c:\n"
        '        resp = await c.get(f"https://api.example.com/users/{user_id}")\n'
        "    await asyncio.sleep(0.1)\n"
        "    return resp.json()"
    )
    r = rewrite_function_async(f, "fetch_user", new_source=new)
    assert r.applied and r.is_async
    body = f.read_text()
    assert "httpx.AsyncClient" in body and "await asyncio.sleep" in body
    assert ast.parse(body)  # still valid Python


def test_ide_missing_function_errors(tmp_path):
    f = tmp_path / "svc.py"
    f.write_text(_SYNC_SRC)
    r = rewrite_function_async(f, "nope")
    assert not r.applied and "not found" in (r.error or "")


def test_ide_already_async_is_reported(tmp_path):
    f = tmp_path / "svc.py"
    f.write_text("async def already():\n    return 1\n")
    r = rewrite_function_async(f, "already")
    assert not r.applied and r.is_async and "already async" in (r.error or "")


# --- Chrome compare -------------------------------------------------------


def test_chrome_renders_comparison_and_opens(tmp_path):
    calls: list[str] = []
    res = compare_products(
        ["iPhone 15", "Pixel 8"],
        fetch=_fake_fetch,
        navigate=lambda url, shot: calls.append(url) or {"title": "Compare: iPhone 15 vs Pixel 8"},
        out_dir=tmp_path,
    )
    assert res.opened and res.page_path is not None
    assert calls == [res.url] and res.url is not None and res.url.startswith("file://")
    page = Path(res.page_path).read_text()
    assert "iPhone 15" in page and "Pixel 8" in page and "is a product" in page


def test_chrome_compare_needs_two_products():
    res = compare_products(["only one"])
    assert not res.opened and "need >=2" in (res.error or "")


# --- tool declarations ----------------------------------------------------


def test_tool_declarations_cover_both_actions():
    names = {d["name"] for d in TOOL_DECLARATIONS}
    assert names == {"rewrite_function_async", "compare_products"}


# --- end-to-end: simulated model tool call -> dispatcher -> worker -> action


async def test_compare_products_end_to_end_through_worker(tmp_path):
    """The exact path a live tool call takes: dispatch -> worker -> handler -> action."""
    results: list[JobResult] = []
    worker = BackgroundWorker(on_result=results.append)
    dispatcher = ToolDispatcher(worker)
    nav_calls: list[str] = []
    dispatcher.register(
        "compare_products",
        lambda a: compare_products(
            list(a.get("products", [])),
            fetch=_fake_fetch,
            navigate=lambda url, shot: nav_calls.append(url) or {"title": "cmp"},
            out_dir=tmp_path,
        ),
    )

    # the tool call the model would emit
    dispatcher.dispatch(
        {"name": "compare_products", "args": {"products": ["PS5", "Xbox Series X"]}}
    )
    await worker.drain()

    assert len(nav_calls) == 1
    assert results[0].ok and results[0].result.opened
    assert "PS5" in Path(results[0].result.page_path).read_text()
    await worker.aclose()


async def test_rewrite_async_end_to_end_through_worker(tmp_path):
    f = tmp_path / "svc.py"
    f.write_text(_SYNC_SRC)
    results: list[JobResult] = []
    worker = BackgroundWorker(on_result=results.append)
    dispatcher = ToolDispatcher(worker)
    dispatcher.register(
        "rewrite_function_async",
        lambda a: rewrite_function_async(a["file"], a["function"], a.get("new_source")),
    )

    dispatcher.dispatch(
        {"name": "rewrite_function_async", "args": {"file": str(f), "function": "fetch_user"}}
    )
    await worker.drain()

    assert results[0].ok and results[0].result.applied and results[0].result.is_async
    assert isinstance(ast.parse(f.read_text()).body[2], ast.AsyncFunctionDef)
    await worker.aclose()
