# Computer-Use Models for Aimer: Research & Recommendation

*Compiled 2026-06-01. Sources: Perplexity Sonar Deep Research (benchmarks + Chinese-model dossiers), WebSearch/WebFetch (US-frontier integration + specialist-open). Note: the original Perplexity swarm completed only 2 of 4 research legs before the Perplexity API quota was exhausted (`401 insufficient_quota`); the US-frontier and specialist-open legs were re-run via WebSearch.*

---

## 1. TL;DR

Adopt a **two-model split**, matched to Aimer's two layers:

- **Local grounding / entity-extraction layer (Week 5):** **Qwen3-VL-4B/8B** (Apache-2.0) running locally via **MLX on Apple Silicon**, with **Holo1.5-7B** as the GUI-specialist alternative and **Gemini Flash-Lite** as a cloud escalation path for low-confidence tiles. This layer is single-tile localization on the 711 ms voice hot path — you do **not** want a frontier API here (latency + privacy).
- **Agentic actuator layer (Week 7):** a **frontier computer-use brain behind the orchestrator/DuplexSession seam, off the hot path** (Week 6 async worker). Primary pick **Claude Opus 4.x computer-use** (OSWorld-Verified leader, OS-agnostic, returns structured actions you execute via CGEvent/AX). **Watch OpenAI Desktop Codex closely** — it shipped macOS-first in April 2026 and may become the better native fit. Full-task reliability (not grounding) is the binding constraint, and the best *open* actuators top out ~30 points below frontier — you cannot ship a 47%-success actuator that edits real code.

**The key architectural insight:** the two layers have *opposite* constraints. Grounding is cheap, 10 Hz-frequent, latency-critical, privacy-sensitive → **local small model**. Actuation is expensive, infrequent, latency-tolerant, reliability-critical → **frontier API**. Don't try to serve both with one model.

---

## 2. The landscape (top contenders, mid-2026)

| Model | Origin | Access | Headline benchmark | Latency / speed | License / cost |
|---|---|---|---|---|---|
| **Claude Opus 4.8** | US (Anthropic) | Closed API + computer-use tool | **OSWorld-Verified 83.4%** (leader, at/above human band); ScreenSpot-Pro **87.9%** (leader) | ~50 tok/s | Closed; ~$0.10–0.50 per 50-step run |
| **GPT-5.x + OpenAI CUA / Desktop Codex** | US (OpenAI) | Closed API + Operator/Codex products | ScreenSpot-Pro ~86%; CUA OSWorld 38.1%; WebArena ~58% | **~187 tok/s** (fastest) | Closed; per-token + image |
| **Gemini 3.x Pro Computer Use** | US (Google) | Closed API (browser-anchored) | ScreenSpot-Pro ~84%; Gemini-2.5/Agent-S2 OSWorld 41.4% | Flash ~650 ms short queries | Closed; cheapest of the three |
| **UI-TARS-2 / 1.5-7B** | China (ByteDance) | 1.5-7B **open (Apache-2.0)**; 2 closed MoE | OSWorld 47.5 (v2) / 42.5 (1.5); ScreenSpot-v2 94.2; ScreenSpot-Pro 61.6; AndroidWorld 64–73 | self-host (7B) | Apache-2.0 (1.5 only) |
| **Qwen3-VL (GUI-Owl-1.5)** | China (Alibaba) | **Open (Apache-2.0)** 4B/8B/32B/235B | **ScreenSpot-Pro 80.3 (open SOTA)**; OSWorld-Verified 56.5; ScreenSpot 95.8 (32B) | self-host, MLX-ready | Apache-2.0 |
| **GLM-5V-Turbo / GLM-4.5V** | China (Zhipu) | 4.5V open (MIT, 106B MoE); Turbo closed | **AndroidWorld 75.7 (leader)**; OSWorld 62.3; WebVoyager 88.5 | API | MIT (4.5V) / closed (Turbo) |
| **Holo1.5** | EU (H Company) | **Open weights** 3B/7B/72B | SOTA *computer-use localization*; beats Qwen2.5-VL, **Sonnet 4**, UI-TARS-1.5 | self-host (3B/7B Mac-friendly) | Open weights (verify commercial terms) |
| **UGround-V1-7B / OS-Atlas / Aguvis / ShowUI-2B** | US/academic | **Open** | Grounding specialists; OS-Atlas trained on macOS-inclusive 13M-element corpus | self-host (2B–7B) | Open (Apache/MIT-class) |
| **Ferret-UI Lite (3B)** | US (Apple) | On-device research model | Apple's own on-device GUI agent (Feb 2026) | **on-device, Mac-native** | Apple research |

---

## 3. Who actually leads

- **Overall full-task computer use:** **Claude Opus 4.8** is the closest thing to an outright leader — it tops *both* OSWorld-Verified (83.4%, now at/above the human band) *and* the hardest grounding set ScreenSpot-Pro (87.9%) simultaneously. **GPT-5.x** and **Gemini 3.x** are co-leaders on grounding but their *agent wrappers* (CUA/Operator) lag on full-task OSWorld (38–41%).
- **Grounding has largely saturated** on easy sets (ScreenSpot-v2 ~92–95% for 7B specialists). The binding constraint is **end-to-end task success** (multi-step plan + execute), where even leaders sit far below humans on WindowsAgentArena (~19.5% vs 74.5%) and broad multi-app workflows score as low as 5%.
- **China is highly competitive and leads specific mirrors:** GLM-5V-Turbo tops AndroidWorld; Qwen3-VL (GUI-Owl-1.5) is the **open-weight grounding SOTA** (ScreenSpot-Pro 80.3). For anything self-hosted, the frontier is Chinese-open (Qwen, UI-TARS) + EU-open (Holo1.5).
- **Caveat:** many 2026 figures come from aggregator/mirror leaderboards (BenchLM, LLM-Stats) with heterogeneous eval setups — treat cross-benchmark comparisons as directional, not apples-to-apples.

---

## 4. Recommendation for Aimer's two layers

### (a) Local grounding / entity-extraction — **Qwen3-VL-4B/8B (MLX)**
**Why:** This layer (Week 5; `extracted_entities: list[Entity]` + `HoverRegion` stubs already in `aimer-core/schema.py`) is pure localization/classification on a single 256×256 cursor tile — the **easy-ScreenSpot regime** (one dominant element under the cursor), not the hard ScreenSpot-Pro professional regime. A 4B/8B open model is sufficient and is the *right* choice because this layer:
- sits **on the 711 ms voice hot path** → can't afford an API round-trip;
- is **privacy-critical** (cursor tile = screen contents) → local-first/redact-at-source posture demands on-device;
- runs at up-to-10 Hz → frontier per-call cost is untenable.

Qwen3-VL wins on four concrete points: **Apache-2.0** (clean commercial use, unlike GLM/closed UI-TARS-2), **first-class MLX/GGUF support on Apple Silicon**, it's the **strict upgrade of Aimer's already-named Qwen2.5-VL-7B candidate**, and it doubles as a general VLM for typed-entity emission (place/date/product/code_span/todo), not just clicks.

- **Primary:** Qwen3-VL-4B (4-bit, MLX) on the hot path; 8B as an A/B.
- **GUI-specialist alternative:** **Holo1.5-7B** (open weights, beats Qwen2.5-VL + Sonnet 4 on localization) if pure click-grounding accuracy matters more than general entity typing. Also consider **UGround-V1-7B** / **OS-Atlas** (macOS-inclusive training corpus) / **Ferret-UI Lite-3B** (Apple's on-device model — the lightest option, worth tracking).
- **Cloud escalation:** Gemini Flash-Lite for low-confidence tiles only — not the default.

### (b) Agentic actuator — **Claude Opus 4.x computer-use (frontier API, off hot path)**
**Why:** Week 7 actions ("rewrite this function async", "compare these products") are long-horizon plan+execute, where the local-vs-frontier gap is largest and most decision-relevant. Best *open/self-hostable* actuators top out at UI-TARS-2 47.5 / Agent S3 63.5 / CoAct-1 60.76 OSWorld — vs Opus 4.8's 83.4%. **For a product where a wrong action edits real code or clicks the wrong thing, that ~20–35-point reliability gap is the whole ballgame.**

The actuator does **not** inherit the grounding layer's constraints: it runs **off the hot path** (Week 6 async worker), actions are user-initiated and infrequent (frontier cost is fine), and it slots cleanly behind the **orchestrator-above-DuplexSession keystone** as a swappable provider — exactly like the duplex provider. Two Aimer-specific advantages: you **already capture the AX/accessibility tree**, and a11y-tree + Set-of-Marks input beats pixel-only — so feed Claude AX labels + the cursor tile as grounded marks to lift reliability above raw-pixel benchmarks.

- **Primary:** Claude Opus 4.x computer-use — OS-agnostic, returns **structured click/type actions you execute locally** via CGEvent/AX (you own actuation; the model only decides the next action and you verify).
- **Strong fallback / watch:** **OpenAI Desktop Codex** — shipped **macOS-first** (April 2026), parallel agent sessions, ~187 tok/s (3–4× Claude). It may become the better *native-macOS* fit; build the actuator behind a provider seam so you can A/B Claude vs Codex.
- **Avoid for now:** Gemini Computer Use (browser-anchored, weaker for native-app/file workflows); local UI-TARS-2 (not yet reliable enough to drive real edits).

**Design constraint to internalize:** frontier computer-use tools today ship primarily as **browser/VM sandboxes**, not native-macOS drivers. Aimer must **own the macOS actuation layer** (CGEvent/AX cursor+keyboard) and use the model only for the decision/plan step. OpenAI Desktop Codex is the first major exception (macOS-native), which is why it's worth tracking closely.

---

## 5. Phased adoption path

1. **Week 5 (grounding) — prototype now.** Stand up Qwen3-VL-4B via MLX, feed it the existing 256×256 cursor tile, emit `extracted_entities`. Benchmark TTFT on-device against the 711 ms budget. A/B 4B vs 8B vs Holo1.5-7B on a small Aimer-specific tile set. Keep Gemini Flash-Lite wired as an escalation toggle.
2. **Week 6 (plumbing) — build the actuator seam.** Add the async background worker + an `Actuator` provider ABC above DuplexSession (mirror of the existing provider pattern). Define the structured action schema (click/type/scroll/key) that you execute locally via CGEvent/AX and verify.
3. **Week 7 (actuator) — Claude Opus first, Codex A/B.** Wire Claude Opus 4.x computer-use as the first actuator provider; feed it AX-tree Set-of-Marks + cursor tile. Demo "rewrite this function async" + "compare these products". Add OpenAI Desktop Codex as the second provider and A/B native-macOS reliability + latency.
4. **Defer:** local/self-hosted actuator (UI-TARS-2, Holo3) until open full-task success crosses ~70% OSWorld — revisit at Week 8 eval time. Gemini Computer Use unless a browser-heavy use case emerges.

---

## 6. Open risks / unknowns to validate

- **On-device grounding latency:** no published TTFT for Qwen3-VL-4B on Apple Silicon for single small-image queries — must measure against the 711 ms hot-path budget before committing.
- **Holo1.5 commercial license:** "open weights" confirmed, exact commercial terms not — verify on HuggingFace before shipping.
- **Frontier actuator native-macOS gap:** Claude/Gemini computer-use ship as browser/VM sandboxes; confirm the "model decides, Aimer actuates via AX/CGEvent" pattern works with their action schemas. Desktop Codex is the macOS-native bet but is newest/least-documented.
- **Benchmark heterogeneity:** OSWorld-Verified 83.4% (Opus) vs research-subset 60–63% figures aren't directly comparable — don't over-index on any single leaderboard number.
- **Cost at scale:** $0.10–0.50 per 50-step actuator run is fine for infrequent user-initiated actions; model the cost if actuation frequency grows.
- **Quota/ops:** the Perplexity research quota ran out mid-swarm — top up `perplexity.ai/settings/api` before re-running the full 4-leg deep research, or keep WebSearch as the fallback path.
