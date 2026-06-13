# Week 4 — Deictic resolver acceptance

**Criterion:** "Fix this" / "summarize that" resolves correctly on ≥80% of the deictic eval.

**Result: ACCEPTED — 87.1% (production-faithful config), clears the 80% bar with ~7-pt margin on two independent runs.**

## Numbers (measured)

| Config | Run | single-judge (sonnet-4-5) | 3-vote (sonnet-4-6) |
|---|---|---|---|
| **Production-faithful** (AX on, tile + downscaled full-frame) | run1 | 75.9% | **86.2% (50/58)** |
| | run2 | 81.0% | **87.9% (51/58)** |
| | **mean** | 78.5% | **87.1%** (spread 1.7 pts) |
| Vision-only (AX off) | run1 | 70.7% | 79.3% (46/58) |

**AX effect: +7.8 pts** (79.3% → 87.1%), robust across judges. AX helps where populated
(links, table values, sub-parts of long text) and occasionally hurts (a "talk page"
hallucination on a heading); net clearly positive.

## What "production-faithful" means

The deployed system always sends, on every turn, on the never-interrupts video channel:
1. a **native-resolution 512×512 cursor tile** (the pointed-at element, full quality), plus
2. a **downscaled (~819×1024) full frame** for page-level context (research finding #6:
   high-res crop *with* full-image context beats crop-alone), plus
3. a text annotation carrying `app / title / cursor / selected / ax`
   (`accessibility_label` from macOS AX: `_first_present(AXDescription, AXTitle,
   AXRoleDescription, AXValue)` of the element under the cursor).

The eval mirrors this exactly. The **vision-only** arm (`--no-ax`) omits the AX fields and is
a strictly *harder* condition than production — useful as a lower bound, not the acceptance
number. Acceptance is measured on the config that actually ships (AX on).

## Methodology and integrity notes

- **Two noise sources were separated.** Initial single-run numbers swung ~8 pts
  (vision-only 70.7%↔79.3% across identical-config runs). Re-scoring the *same saved
  transcripts* with a stronger 3-vote judge moved both arms up ~9–10 pts and **tightened the
  ON-arm spread to 1.7 pts** — i.e. most of the apparent "model variance" was *judge* noise.
- **The judge upgrade is a fix, not goal-seeking.** The prior single sonnet-4-5 judge made
  objective errors (scored a correct link-identification as incorrect). The replacement
  (3-vote majority, sonnet-4-6) was validated by a **full independent hand-adjudication of
  all 58 ON-run1 tasks** against the written rubric — not just the swing tasks. The
  hand-count = **50/58 = 86.2%, exactly matching the 3-vote judge.** The judge has exactly
  one false-positive (`web_0023`, scored the "talk page" answer correct — it is wrong) and
  one false-negative (`web_0025`, scored a valid functional description wrong); they cancel.
  So 86% is verified by direct human reading, not leniency-inflated.
- **A failure mode was root-caused, not hand-waved.** Two tasks (`web_0023` "Internet",
  `web_0057` "Artificial intelligence") produced the *same* wrong answer — "this is the Talk
  page." Inspecting the tiles shows why: every Wikipedia `#firstHeading` crop also captures
  the "Article | Talk" nav bar ~20px below the heading, and "Talk" (blue, link-styled) is a
  salient distractor under the cursor. The model usually binds to the heading (e.g. `web_0001`
  "Coffee", identical layout, scored correct) but stochastically flips to "Talk". Notably
  `web_0057` carried the *correct* `ax='Artificial intelligence'` and the model chose the
  visually-prominent tab anyway — visual salience can override a correct AX label. This is
  partly a web-eval artifact (native macOS targets rarely have clickable chrome 20px away).
- **Remaining caveats (honest):** a few HN tasks have stale `expected_referent` (the front
  page is dynamic — those tasks are AX-empty and excluded from the AX delta); AX
  paragraph-derivation occasionally mismatches the exact cursor paragraph. Neither moves the
  acceptance conclusion.

## Reproduce

```bash
# 1. (optional) re-derive faithful AX labels for stable-page tasks
uv run --package duplex-bridge python scripts/bench/augment_ax.py

# 2. run the two arms on identical fixtures (only with_ax differs)
uv run --package duplex-bridge python scripts/bench/eval_deictic.py \
  --tasks scripts/bench/fixtures/deictic_tasks_web_ax.jsonl --escalate-full-frame --no-ax \
  --out scripts/bench/results/week4_ax_off.jsonl
uv run --package duplex-bridge python scripts/bench/eval_deictic.py \
  --tasks scripts/bench/fixtures/deictic_tasks_web_ax.jsonl --escalate-full-frame \
  --out scripts/bench/results/week4_ax_on.jsonl

# 3. stable 3-vote scoring (offline — no Gemini sessions)
uv run --package duplex-bridge python scripts/bench/rejudge.py \
  --results scripts/bench/results/week4_ax_on.jsonl \
  --fixtures scripts/bench/fixtures/deictic_tasks_web_ax.jsonl \
  --judge-model claude-sonnet-4-6 --votes 3
```

Raw per-task transcripts + verdicts: `scripts/bench/results/week4_ax_{off,on,on_run2}*.jsonl`.
