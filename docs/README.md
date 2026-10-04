# docs/ — which file to open

| To… | Open |
|---|---|
| **Run V5** (every rule, check, arm, parameter, night and command) | [`V5_RUNSHEET.md`](V5_RUNSHEET.md) |
| **Plan or run V6** (plan in Part A, execution in Part B) | [`V6_RUNSHEET.md`](V6_RUNSHEET.md) |
| Understand *why* V5 is designed this way | [`v5_design/`](v5_design/): Amendment 02 (biology, M1–M7), the Dermatologist Reasoning Engine, the audit, the idea pool |
| See the frozen V5 pre-registration | [`v5_record/`](v5_record/): master plan, frozen runsheet, session plan, Amendment 01. **Never edit.** Hashes are in `results/v5/v5_plan_freeze.json` |
| Read the programme history | [`RESEARCH_LOG.md`](RESEARCH_LOG.md) |

**Rules:**
1. There is exactly **one runsheet per version**. If a design or record file disagrees with a
   runsheet, the runsheet wins (precedence: runsheet → Amendment 02 → DRE → Amendment 01 → frozen
   record).
2. `v5_record/` files are byte-identical to their frozen versions. Paths quoted inside them still
   read `docs/<name>`; they refer to the same files, now in `v5_record/`.
3. After `V5_RUNSHEET.md` is hashed (with Amendment 02 and the DRE), any change is a new
   amendment, labelled post-hoc if it follows any V5 result.
