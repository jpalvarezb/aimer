"""IDE host action — rewrite a function to async in a real source file.

Two paths, both ending in a verifiable on-disk edit:
  - model-provided: the duplex model supplies the full async ``new_source`` (the live path);
    we splice it in over the original function.
  - deterministic: no ``new_source`` — we transform the function with ``ast`` (``def`` ->
    ``async def`` and ``await``-wrap known blocking calls), so the headless demo/test produces a
    real sync->async rewrite with no model in the loop.

Either way the result is re-parsed and the rewritten function is asserted to be an
``async def`` before the file is written, so a bad edit never lands.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Call names converted to ``await <call>`` when async-ifying (their async equivalents are what a
# developer expects). Conservative by design — only these names are awaited.
_DEFAULT_AWAITABLE_CALLS = frozenset(
    {
        "sleep",
        "get",
        "post",
        "put",
        "delete",
        "patch",
        "request",
        "fetch",
        "read",
        "write",
        "connect",
        "send",
        "recv",
        "query",
        "execute",
        "download",
        "upload",
    }
)


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of an async-rewrite action."""

    file: str
    function: str
    applied: bool
    is_async: bool
    new_source: str | None = None
    error: str | None = None


class _AwaitInjector(ast.NodeTransformer):
    """Wrap calls to known-blocking functions in ``await`` (skip already-awaited calls)."""

    def __init__(self, awaitable: frozenset[str]) -> None:
        self._awaitable = awaitable

    def visit_Await(self, node: ast.Await) -> ast.AST:
        return node  # don't descend — never double-await

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if _call_name(node.func) in self._awaitable:
            return ast.Await(value=node)
        return node


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    return None


def _to_async(node: ast.FunctionDef, awaitable: frozenset[str]) -> ast.AsyncFunctionDef:
    """Build an AsyncFunctionDef from a FunctionDef, await-wrapping known blocking calls."""
    async_node = ast.AsyncFunctionDef(
        name=node.name,
        args=node.args,
        body=[_AwaitInjector(awaitable).visit(stmt) for stmt in node.body],
        decorator_list=node.decorator_list,
        returns=node.returns,
        type_comment=node.type_comment,
    )
    return ast.fix_missing_locations(async_node)


def _span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int]:
    """1-based inclusive (start, end) line span of the function, including any decorators."""
    start = node.lineno
    if node.decorator_list:
        start = min(start, min(d.lineno for d in node.decorator_list))
    assert node.end_lineno is not None
    return start, node.end_lineno


def _indent(source: str, spaces: int) -> str:
    if spaces == 0:
        return source
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in source.splitlines())


def rewrite_function_async(
    file: str | Path,
    function: str,
    new_source: str | None = None,
    awaitable_calls: frozenset[str] | None = None,
) -> RewriteResult:
    """Rewrite ``function`` in ``file`` to async. Returns a verifiable :class:`RewriteResult`.

    With ``new_source`` (model-provided), splice it over the function. Without it, apply a
    deterministic ``ast`` transform. The edit is verified (re-parsed, function is ``async def``)
    before the file is written.
    """
    path = Path(file)
    try:
        original = path.read_text()
        tree = ast.parse(original)
    except (OSError, SyntaxError) as exc:
        return RewriteResult(str(path), function, applied=False, is_async=False, error=str(exc))

    target = _find_function(tree, function)
    if target is None:
        return RewriteResult(
            str(path),
            function,
            applied=False,
            is_async=False,
            error=f"function {function!r} not found",
        )

    col = target.col_offset
    if new_source is not None:
        replacement = _indent(new_source.strip("\n"), col)
    else:
        if isinstance(target, ast.AsyncFunctionDef):
            return RewriteResult(
                str(path), function, applied=False, is_async=True, error="already async"
            )
        awaitable = awaitable_calls if awaitable_calls is not None else _DEFAULT_AWAITABLE_CALLS
        replacement = _indent(ast.unparse(_to_async(target, awaitable)), col)

    start, end = _span(target)
    lines = original.splitlines()
    rewritten = "\n".join(lines[: start - 1] + replacement.splitlines() + lines[end:])
    if original.endswith("\n"):
        rewritten += "\n"

    # Verify before writing: must parse and the function must now be async.
    try:
        verify = _find_function(ast.parse(rewritten), function)
    except SyntaxError as exc:
        return RewriteResult(
            str(path),
            function,
            applied=False,
            is_async=False,
            new_source=replacement,
            error=f"rewrite did not parse: {exc}",
        )
    is_async = isinstance(verify, ast.AsyncFunctionDef)
    if not is_async:
        return RewriteResult(
            str(path),
            function,
            applied=False,
            is_async=False,
            new_source=replacement,
            error="rewrite is not async",
        )

    path.write_text(rewritten)
    logger.info("[ide] rewrote %s::%s to async", path.name, function)
    return RewriteResult(str(path), function, applied=True, is_async=True, new_source=replacement)
