# GPT Pro: start here — 20-day scientific strategy review

Review snapshot: 2026-09-07. This is the current entry point. Older prompts and
historical decisions are evidence, not the current request to execute their plans.

## Current request

Read [the complete review prompt](GPTPRO_STRATEGY_PROMPT_20260907.md).
Return one self-contained Markdown document named
`GPTPRO_GREEN_20DAY_SCIENTIFIC_STRATEGY_20260907.md`.
The author has 20 days total, including the final five days for manuscript polishing.
The objective is a complete, credible, high-impact paper with an Oral ambition,
not an unsupported promise of an Oral outcome.

## Read in this order

1. [Current state, uncertainties, and corrections](review/20260907/CURRENT_STATE.md).
2. [Author-approved 20-day planning constraint and draft roadmap](analysis/GREEN_PAPER_20_DAY_DEPTH_FIRST_PLAN_20260907.md).
   This is a proposed allocation to critique, not a claim that the full route fits.
3. [Current manuscript](iclr2027/paper.tex) and [bibliography](iclr2027/references.bib).
   It is a draft, not a source of truth about completion or priority.
4. [Development outcomes and strong-baseline evidence](analysis/GREEN_V400_DEVELOPMENT_COMPLETENESS_DECISION_20260830.md).
5. [Corrected v3 diagnostic evidence](analysis/GREEN_V300_DEVELOPMENT_TERMINAL_20260826/POSTCORRIGENDUM_DIAGNOSTIC.md)
   and [its machine-readable summary](analysis/GREEN_V300_DEVELOPMENT_TERMINAL_20260826/POSTCORRIGENDUM_DIAGNOSTIC.json).
6. [Numerical repair handoff](analysis/GREEN_V410_P13_NUMERICAL_REPAIR_HANDOFF_20260907.md)
   and [existing 92-test report](analysis/GREEN_V410_REPAIR_REGRESSION_FINAL_20260907.xml).
7. [v4.1 scientific protocol](analysis/GPTPRO_GREEN_V4_SUCCESSOR_CERTIFICATE_PROTOCOL_DECISION_20260830.md),
   especially estimand/classifier, P13, resource policy, confirmation analysis, and endpoint separation.
8. [Execution scale and missing confirmation modules](analysis/GREEN_V410_LUNAMAX_MASTER_EXECUTION_PLAN_20260906.md).
   Its September 6 launch commands are superseded by the repair handoff.
9. [Earlier novelty/self-deception audit](analysis/CODEX_GREEN_V400_NOVELTY_COLLISION_AND_SELF_DECEPTION_AUDIT_20260828.md)
   and [original mission alignment](analysis/CODEX_GREEN_V400_ORIGINAL_MISSION_ALIGNMENT_CORRIGENDUM_20260828.md).
   Independently verify consequential literature claims; the audit is not authoritative.

## Inspect these implementations where claims require it

- [Complete certificate loop](src/green_v410_fixed_budget_certificate.py).
- [Actual P13 worker and record construction](analysis/green_v410_p13_certificate_worker.py).
- [Graph builder](src/green_bridge_v400_tensor_program.py).
- [Tensor executor](src/green_bridge_v400_mpfr_tensor_executor.py).
- [MPFR backend](native/green_v400_mpfr_backend.cpp).
- [Site classifier](src/green_v410_classifier.py) and [P13 rules](src/green_v410_p13.py).
- [Current engineering acceptance entry point](analysis/green_v410_certificate_repair_probe.py).

The full certificate evaluates five outputs together on each graph traversal.
Repeated node ordinals do not count completed outputs. At L4, each radius requires
six official interval evaluations, four audit interval evaluations, and six
endpoint evaluations: 48 evaluator calls over the three radii. Costs are nonuniform.

## Targeted historical evidence, only as needed

- [July core audit](ICLR_1_CORE_AUDIT_20260712.md).
- [IRS theory and scope](analysis/IRS_THEORY_P0.md).
- [IRS versus single direction](analysis/irs_vs_single_direction.md).
- [IRS stress summary](analysis/p0_irs_stress_summary.md).
- [Manuscript evidence ledger](iclr2027/REVISION_EVIDENCE_LEDGER.md).
- `iclr2026/` is the superseded manuscript and figures; it is not the current submission.
- `archive/prompts/` contains superseded early prompts.
- `analysis/` retains historical results and decisions to prevent selective erasure.

## What this publication is and is not

It contains the current local implementation snapshot, current draft, planning
documents, historical evidence, and an existing test receipt. It contains no new
formal result and no untouched confirmation endpoint payload. Large model,
tensor, environment, and run files remain outside this review publication.

The two engineering acceptance jobs were still running at the snapshot in
CURRENT_STATE.md. Successful unit tests do not establish successful real-world
certificates, useful contraction, predictive superiority, or semantic mechanism
identity.

See [publication scope and cleanup record](review/20260907/PUBLICATION_SCOPE.md).
