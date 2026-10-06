# docs/ — research documents in the order they were written

Every number quoted in these documents lives in `results/` (Hard Rule 4); the session-by-session
narrative is [`../CHANGELOG.md`](../CHANGELOG.md).

## Chronological index

| # | Version | Dates (2026) | Document | What it is |
|---|---|---|---|---|
| 1 | V1 | Aug – 4 Sep | [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md) | The original programme: phases, hard rules, primary metrics |
| 2 | V1 → V4 | 4 Sep – 17 Sep | [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | Session log from the single test pass (S9) to the V4 final audit |
| 3 | V4 | 17 Sep | [`../paper/v4/v4_findings.md`](../paper/v4/v4_findings.md), [`../paper/v4/model_card.md`](../paper/v4/model_card.md) | What V4 settled; model card for the deployed stack |
| 4 | V5 (frozen plan) | 18 Sep | [`v5_record/V5_MASTER_RESEARCH_PLAN_REVISED.md`](v5_record/V5_MASTER_RESEARCH_PLAN_REVISED.md), [`v5_record/V5_FINAL_RUNSHEET.md`](v5_record/V5_FINAL_RUNSHEET.md), [`v5_record/V5_SESSION_PLAN.md`](v5_record/V5_SESSION_PLAN.md) | The V5 pre-registration as first frozen |
| 5 | V5 (Amendment 01) | 23 Sep | [`v5_record/V5_PLAN_AMENDMENT_01.md`](v5_record/V5_PLAN_AMENDMENT_01.md) | Pre-S01 review; adopted before any V5 GPU run |
| 6 | V5 (design) | 23 – 29 Sep | [`v5_design/V5_IDEAS_BIOLOGY_FIRST.md`](v5_design/V5_IDEAS_BIOLOGY_FIRST.md) | Candidate biology-first mechanisms (idea pool, not pre-registered) |
| 7 | V5 (design) | 29 Sep | [`v5_design/V5_PLAN_AMENDMENT_02.md`](v5_design/V5_PLAN_AMENDMENT_02.md), [`v5_design/V5_DERM_REASONING_ENGINE.md`](v5_design/V5_DERM_REASONING_ENGINE.md), [`v5_design/V5_PLAN_AUDIT.md`](v5_design/V5_PLAN_AUDIT.md) | Amendment 02, the dermatologist reasoning engine, and its pre-adoption audit |
| 8 | V5 (execution) | adopted 30 Sep | [`V5_RUNSHEET.md`](V5_RUNSHEET.md) | The single V5 execution document: arms, gates, schedule, commands |
| 9 | V6 (draft) | 29 Sep – 4 Oct | [`V6_RUNSHEET.md`](V6_RUNSHEET.md) | V6 plan (Part A) and execution (Part B); runs after Review 2 |

## Rules

1. There is exactly **one runsheet per version**. If a design or record file disagrees with a
   runsheet, the runsheet wins (precedence: runsheet → Amendment 02 → DRE → Amendment 01 → frozen
   record).
2. `v5_record/` and the three adopted V5 design files are **byte-identical to their hashed
   versions**; the hashes are in `results/v5/v5_plan_freeze.json` and are re-checked by
   `python -m research.v5.adopt_e1 --verify`. Paths quoted inside them read `docs/<name>`; they refer
   to the same files, now in `v5_record/`.
3. Any change after a hash is a new amendment, labelled post-hoc if it follows any result.
