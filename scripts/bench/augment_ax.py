"""Augment the web deictic fixtures with a faithful accessibility_label per task.

Production always carries ``ContextPacket.semantic.accessibility_label`` (macOS AX:
first non-empty of AXDescription / AXTitle / AXRoleDescription / AXValue of the element
under the cursor). The harvested web fixtures omit it, so the eval has been running in a
*degraded* condition vs. the real system. This script re-derives a faithful accessible
name for each task's pointed-at element so we can A/B the AX signal's effect on the score.

Faithfulness rules (so this measures the real system, not the gold answer):
  - The AX name is the element's *own* accessible name, computed in-page the way an AX
    tree would expose it (aria-label -> alt -> visible text -> role) -- NOT derived from
    ``expected_referent``.
  - All 58 tasks are *point* (hover) tasks, so only ``accessibility_label`` is populated;
    ``selected_text`` stays empty (production only fills it on a real text selection).
  - Dynamic pages (news.ycombinator.com) drift from the frozen tiles, so their AX is left
    EMPTY -- identical in both A/B arms, so they wash out of the delta. Stable pages
    (Wikipedia / MDN / httpbin) are re-derived from the same selector + viewport the
    harvester used.

Run:  uv run --package duplex-bridge python scripts/bench/augment_ax.py
Out:  scripts/bench/fixtures/deictic_tasks_web_ax.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

_HERE = Path(__file__).resolve().parent
_IN = _HERE / "fixtures" / "deictic_tasks_web.jsonl"
_OUT = _HERE / "fixtures" / "deictic_tasks_web_ax.jsonl"

# Pages whose content drifts between harvest and now: leave AX empty (washes out of A/B).
_DYNAMIC_HOSTS = {"news.ycombinator.com"}

# Mirror the harvester's render conditions exactly (harvest_web_tasks.py).
_VIEWPORT = {"width": 1280, "height": 1600}
_SCALE = 2

# Computes a faithful accessible name in-page: aria-label -> alt (img) -> label (form
# field) -> visible text -> role. Mirrors macOS AX's first-present-of cascade.
_ACCESSIBLE_NAME_JS = """
(el) => {
  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
  const aria = el.getAttribute && el.getAttribute('aria-label');
  if (norm(aria)) return norm(aria);
  const tag = el.tagName;
  if (tag === 'IMG') { return norm(el.getAttribute('alt')) || 'image'; }
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
    const lb = el.getAttribute('aria-labelledby');
    if (lb) {
      const t = lb.split(/\\s+/)
        .map(i => { const e = document.getElementById(i); return e ? e.innerText : ''; })
        .join(' ');
      if (norm(t)) return norm(t);
    }
    if (el.id) {
      const l = document.querySelector('label[for="' + el.id + '"]');
      if (l && norm(l.innerText)) return norm(l.innerText);
    }
    const pl = el.closest('label');
    if (pl && norm(pl.innerText)) return norm(pl.innerText);
    if (norm(el.getAttribute('placeholder'))) return norm(el.getAttribute('placeholder'));
    return 'textbox';
  }
  const t = norm(el.innerText || el.textContent);
  if (t) return t;
  return norm(el.getAttribute('role')) || tag.toLowerCase();
}
"""

_SEL_RE = re.compile(r"^(?P<css>.*)\[(?P<n>\d+)\]$")


def _parse_notes(notes: str) -> tuple[str, str | None, int]:
    """Return (url, css, nth) parsed from a task's notes field."""
    url = notes.split("|")[0].strip()
    css: str | None = None
    nth = 0
    m = re.search(r"selector=(.+)$", notes)
    if m:
        raw = m.group(1).strip()
        sm = _SEL_RE.match(raw)
        if sm:
            css = sm.group("css").strip()
            nth = int(sm.group("n"))
        else:
            css = raw
    return url, css, nth


def _host(url: str) -> str:
    return url.split("/")[2] if "://" in url else "?"


def main() -> None:
    tasks = [json.loads(line) for line in _IN.read_text().splitlines() if line.strip()]

    # Group by URL so each page loads once, preserving task order.
    by_url: dict[str, list[dict]] = {}
    for t in tasks:
        url, _, _ = _parse_notes(t.get("notes", ""))
        by_url.setdefault(url, []).append(t)

    populated = 0
    skipped_dynamic = 0
    failed = 0
    samples: list[tuple[str, str, str]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for url, group in by_url.items():
            host = _host(url)
            if host in _DYNAMIC_HOSTS:
                skipped_dynamic += len(group)
                continue
            page = browser.new_page(viewport=_VIEWPORT, device_scale_factor=_SCALE)
            try:
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                except Exception as exc:  # noqa: BLE001
                    print(f"  LOAD FAIL {url}: {exc}")
                    failed += len(group)
                    continue
                page.wait_for_timeout(800)
                try:
                    page.keyboard.press("Escape")
                    page.wait_for_timeout(300)
                except Exception:  # noqa: BLE001
                    pass

                for t in group:
                    _, css, nth = _parse_notes(t.get("notes", ""))
                    if not css:
                        failed += 1
                        continue
                    try:
                        loc = page.locator(css).nth(nth)
                        if loc.count() == 0:
                            print(f"  NO MATCH {t['id']}: {css}[{nth}]")
                            failed += 1
                            continue
                        name = loc.evaluate(_ACCESSIBLE_NAME_JS)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  EVAL FAIL {t['id']}: {exc}")
                        failed += 1
                        continue
                    name = (name or "").strip()[:80]
                    if not name:
                        failed += 1
                        continue
                    t["accessibility_label"] = name
                    populated += 1
                    if len(samples) < 14:
                        kind = ",".join(t.get("tags", []))
                        samples.append((t["id"], kind, name))
            finally:
                page.close()
        browser.close()

    with _OUT.open("w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")

    print("\n=== AX augmentation summary ===")
    print(f"total tasks         : {len(tasks)}")
    print(f"AX populated        : {populated}")
    print(f"skipped (dynamic HN): {skipped_dynamic}")
    print(f"failed/empty        : {failed}")
    print(f"written             : {_OUT}")
    print("\n=== sample AX values (eyeball: element name, NOT the gold answer) ===")
    for tid, kind, name in samples:
        print(f"  {tid} [{kind}] -> {name!r}")


if __name__ == "__main__":
    main()
