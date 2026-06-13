# Deictic Visual Context: Tile vs. Full-Frame — Research & Recommendation

*Compiled 2026-06-01. Source: deep-research harness (23 sources, 96 claims extracted, 25
adversarially verified — 18 confirmed / 7 killed — synthesized to 10 findings). All grounding
evidence is 2024–2026 arXiv (R-VLM, MEGA-GUI, ScreenSpot-Pro, "Zoom in Click out", CropVLM,
Zoom-Refine, Set-of-Mark) plus the Huang & White CHI 2012 gaze-cursor study.*

---

## 1. TL;DR

The Week-4 deictic question — **send just the cursor tile, or the complete screen?** — is a false
binary. The evidence kills both extremes:

- **Full-frame-by-default is wrong.** Single-pass full-screen grounding on dense/high-res GUIs runs
  **6.96%–50.53%** — the model drowns in irrelevant pixels. This is the *dominant* grounding failure
  mode.
- **Tile-only is also wrong.** A single tight crop physically cannot hold two referents or a
  cross-element relation ("compare these two", "move that there"), loses layout/text context, and
  has a *minimum* useful size (a "context floor").

**Recommendation: a tiered, calibrated design.**

1. **Default deixis payload = the cursor tile**, kept at its current 512px (the calibrated
   context-floor crop — do **not** shrink it), with a **lightweight cursor marker** drawn on it
   (cheap Set-of-Mark), plus the AX label + selected text already in the ContextPacket (those carry
   the *element identity* that does the actual disambiguation).
2. **Escalate to a downscaled, cursor-marked full frame on demand** — only for relational /
   multi-referent utterances. Pre-cache the downscaled frame between turns so escalation is just
   *selecting which cached payload to force-send at `activity_start`*, never a new capture on the
   711 ms hot path.
3. **Decouple workflow-mining entirely** onto a local, low-cadence sink built from cursor
   coordinates + AX element identity (no pixels) — off the hosted hot path, consistent with the
   local-first / redact-at-source posture.

---

## 2. The verified evidence

| # | Finding | Confidence | Implication for Aimer |
|---|---|---|---|
| 1 | Full cluttered screenshots are the dominant grounding failure mode; single-pass full-screen grounding is 6.96%–50.53% on dense/high-res GUIs. *(Strongest on hard 4K UIs; easy/low-res screens score 80%+.)* | high | Don't stream the full screen by default — worst exactly where deixis is hardest (small targets, dense screens). |
| 2 | Cropping to a cursor-centered ROI is the **single largest** accuracy lever (+28pp from ROI zoom alone; 18.9%→48.1% coarse-to-fine; Gemini-2.5-Pro 6.96%→46.86%). | high | Validates the cursor tile as the default deictic payload. |
| 3 | Targets average **0.07% of screen area**; accuracy degrades universally as the target shrinks. | high | A crop mechanically raises effective resolution on the referent. |
| 4 | Cropping is **not** unconditionally good — tight crops discard layout/text/icon-grouping context; smallest-crop mode is consistently worst; a minimum **context-floor** size is required. | high | Treat 256pt/512px as a calibrated context floor. Don't shrink further; possibly enlarge. |
| 5 | Relational / multi-referent deixis is materially harder; accuracy falls as the number of spatial relations rises. A single tight crop cannot contain two referents. | high | "Compare these two" / "move that there" need a wider or full-frame representation. |
| 6 | Strongest methods combine a high-res crop **with** full-image context, or global-scan-then-crop — not crop-alone (removing full-image context alongside a crop: 52→46). | high | Direct empirical case for a **tiered** design. |
| 7 | Marking the **full** frame (Set-of-Mark) beats cropping for disambiguation: GPT-4V 25.7%→86.4% on referring-expression comprehension, beating fully-finetuned SOTA; fixes the "multiple similar objects" problem while keeping context. *(Real SoM needs SEEM/SAM — too heavy for the hot path; a cursor marker is the cheap approximation.)* | high | Mechanism for the full-frame escalation tier; cursor-marker as lightweight stand-in. |
| 8 | Contrastive Region Guidance (CRG) beats unguided full-image input but **needs logit access**. Naive bbox-overlay *without* contrast hurts (54→40). | medium | Irrelevant to hosted Gemini Live; reserve for the **Week-5 local VLM** layer. |
| 9 | The cursor is a strong deictic anchor **at/just before a click** (74px at click) but weak during passive hover (233px inactive; cursor inactive 58.8% of the time). | high | Trust the tile most when actively pointing/clicking; during passive hover lean on AX/selected-text and/or widen the crop. |
| 10 | Workflow-mining needs **no pixels** — heatmaps need only (x,y) over time; recurring-workflow detection needs element/window identity sequences. Both already in the ContextPacket. | medium *(synthesized; no process-mining primary survived verification)* | The heatmap/workflow substrate can be a local, low-cadence, non-pixel sink — off the hot path. |

**Refuted (do not rely on):** the blanket "cursor is just noise / lags gaze 700px" claim (0-3); the
ZoomClick raw numbers (0-3); a single-vs-multi-instance 5.7% gap (0-3); specific region-prompt
11.1%/13% gain figures (1-2). The *directional* findings above survived; several precise numbers did
not.

---

## 3. Concrete design for Week 4

### Deixis hot path (sent to Gemini Live)
- **Default:** cursor tile @ 512px (context-floor crop, unshrunk) + **cursor marker** drawn on it →
  realtime-video channel. AX label + selected text → text rail *during the turn* (the
  element-identity signal that disambiguates). This rides the existing turn-gated
  `send_visual_context` path.
- **Escalation:** for relational / multi-referent utterances, force-send a **pre-cached, downscaled,
  cursor-marked full frame** at `activity_start`. Pre-caching between turns keeps the 711 ms floor
  intact (no new capture on the hot path).
- **Hover vs. action gating:** cursor settled/active → trust the tile; passive hover → weight
  AX/selected-text more and/or widen the crop.

### Workflow-mining substrate (decoupled, starts now — see §4)
- A **separate local, low-cadence persistence tier** keyed on `(timestamp, cursor x/y, focused
  window, AX element identity, selected-text)` — **not** the pixel stream.
- Heatmaps fall out of (x,y) over time; recurring-workflow mining out of element/window-identity
  sequences.
- Add *periodic local-only downscaled full frames* later **only if** mining proves it needs pixels
  for canvas/video/custom-drawn UIs the AX tree can't describe.

---

## 4. Timing decision: start the substrate now, defer the mining

**Start now (low-regret, and required early):** the capture substrate — a passive write-tap on the
existing ContextPacket stream. Workflow-mining is longitudinal; it mines *weeks* of history. If
retention doesn't start until Week 8, the mining milestone arrives with an empty database. Starting
the sink now is what makes post-Week-8 mining possible at all. It must **not** touch the Week-4
deixis hot path — separate tier, own cadence, append-only, cheap to delete (honors the privacy
posture).

**Defer to post-Week-8:** the mining + automation itself (heatmap analysis, recurring-workflow
detection, heatmap→cron suggestions). Unproven and expensive; nothing about it needs to exist now.

**Honest caveat:** finding #10 is *medium* confidence — the minimal substrate (AX-identity sequences
alone vs. needing periodic pixels) is unresolved. Mitigation: capture the cheap non-pixel floor now;
add pixels later if mining demands them.

---

## 5. Open questions to settle in the 50-task deictic eval

1. **Optimal context-floor crop size for Gemini Live on Retina** — 256pt/512px is *plausibly* near
   the training-resolution regime but unbenchmarked. Sweep it.
2. **Does a cursor marker measurably help Gemini Live** vs. an unmarked crop? A/B it.
3. **Relational-utterance detection** — speech-side trigger vs. always-cache-and-fallback. The
   pre-cache approach in §3 sidesteps the timing risk regardless.
4. **Minimal workflow-mining substrate** — AX-identity sequences vs. periodic pixels (for canvas /
   video / custom-drawn UIs). Unverified in the literature; settle empirically once the sink has
   data.

---

## 6. Source anchors

Grounding / crop-vs-full-frame: ScreenSpot-Pro (arXiv:2504.07981), MEGA-GUI (arXiv:2511.13087),
R-VLM (arXiv:2507.05673), "Zoom in, Click out" (arXiv:2512.05941), Zoom-Refine (arXiv:2506.01663),
CropVLM / "Zoom" (arXiv:2511.19820). Visual prompting / marks: Set-of-Mark (arXiv:2310.11441),
Contrastive Region Guidance (arXiv:2403.02325). Multi-referent deixis: Tumu et al.
(arXiv:2511.06146). Cursor-as-attention: Huang & White, CHI 2012. Systems/cost: Gemini Live API
capabilities + pricing; "the lethal trifecta" (simonwillison.net, 2025-06-16).
