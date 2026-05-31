from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "aimer-core" / "src"))
sys.path.insert(0, str(ROOT / "duplex-bridge" / "src"))
# This tests/ dir, so the moved `testkit` package resolves as a top-level import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
