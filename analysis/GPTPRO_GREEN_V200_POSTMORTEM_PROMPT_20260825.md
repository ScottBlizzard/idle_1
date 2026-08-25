# Complete English prompt for GPT Pro — GREEN v2.0.0 postmortem

You are the senior scientific architect for an ICLR paper whose intended ceiling remains **Oral**. Please inspect the GitHub repository `https://github.com/ScottBlizzard/idle_1`, especially branch `codex/green-v200` and its latest postmortem commit. Do not answer from this prompt alone: read the repository evidence carefully.

Start with these files:

1. `analysis/GPTPRO_GREEN_V200_CORRIGENDUM_DECISION_20260825.md` — the binding v2.0.0 corrigendum and pre-launch protocol.
2. `analysis/GREEN_V200_DEVELOPMENT_TERMINAL_STOP_20260825.md` — the concise terminal execution report and read-only diagnosis.
3. `analysis/GREEN_V200_DEVELOPMENT_TERMINAL_DIAGNOSTIC_20260825.json` — machine-readable official counts, certificate audit, and clearly labeled non-protocol counterfactual.
4. `analysis/archive/green_v200_stop_20260825/` — selected official terminal artifacts copied from the server and verified against `sha256sums.txt`, including `result.json`, `dev_result.json`, `dev_cells.json`, `manifest.json`, `run_ledger.json`, the merged development Parquet files, AD route/theorem/enclosure audits, model-integrity audit, preflights, split, radii/noise/structural audits, and operation counts.
5. The v2.0.0 implementation and tests under `src/` and `tests/`. The immutable official execution commit is `e52e082296c33a10557636706e572147136fce34`.

The corrected one-shot completed prepare and development exactly once. Its immutable official result is:

```text
verdict = STOP_ORAL
phase = development
first_failed_gate = 12_DEVELOPMENT_SURVIVAL
n_surviving_cells = 0
n_conditioned_cells = 0
n_snr_cells = 0
confirmation_open = false
confirmation_started = false
```

All eight development workers completed; 64 tensor and 64 energy records were merged; all model-integrity checks passed. Across 1,280 gate-system classifications there were 7 `active-identified`, 1,262 `certified-target-null`, 11 `unresolved-bounded`, and zero numerical-invalid, structural-contradiction, route-failure, theorem-failure, or white-box-coordinate-failure cases. No system reached the required three active gates, so tensor set admissibility was 0/64. A deliberately permissive, read-only counterfactual that counted point-complete intervals as admissible still yielded 0/8 set-SNR cells; its mixed midpoint RMSE was about 0.00475 versus baseline LOOCV RMSE about 0.000811, with robust relative gain about -12.12. This counterfactual is diagnostic only and must never be relabeled as an official result.

The v2.0.0 result and formal server root are immutable. You must not authorize confirmation, a v2 retry, retrospective threshold lowering, development-set-driven cell redefinition, or any relabeling of the one-shot.

Please now make the next scientific decision while preserving the paper's theory-first, Oral-level ambition. Codex is the front-line executor and may make engineering adjustments, but needs your comprehensive scientific design at this point. In particular, distinguish among these competing explanations using the archived evidence:

A. The matched-bypass theory is genuinely falsified in the tested regime.

B. The theorem is correct, but the chosen contribution target or active-set identification rule is incompatible with the scale of the actual signal.

C. The interval construction is proof-valid but excessively conservative, especially in its endpoint/curvature remainder composition or SNR propagation.

D. The inferential target scale, normalization, or cell aggregation differs from the phenomenon that produced the earlier strong experiments.

E. The earlier evidence and v2.0.0 evidence concern distinct regimes, and the strongest defensible Oral-level contribution requires an explicitly scoped theorem plus a new falsifiable experiment.

For each explanation, state what the existing evidence supports or rejects and give the decisive analyses needed next. Prefer analyses that can be performed on already-observed development data when they are explicitly postmortem and cannot contaminate a future confirmatory set. If new data or a new split is required, specify a contamination firewall and a new protocol identity before any responses are computed.

Your decision must include:

1. **Scientific diagnosis:** the most likely cause of the v2 failure, with exact evidence anchors.
2. **Theory status:** which theorem, claim, or conceptual main line remains valid; what must be revised; and how to preserve or strengthen the Oral-level novelty without pretending the v2 result succeeded.
3. **Postmortem analyses:** exact read-only analyses Codex may run now on the development artifacts, including formulas, expected output schemas, and decision interpretations. Clearly mark exploratory versus proof-checking analyses.
4. **Successor protocol decision:** whether a new version is scientifically justified. If yes, assign a new identity (for example v2.1.0 or v3.0.0), specify every allowed change, define frozen inputs and hashes, and state what is inherited versus newly estimated.
5. **Anti-overfitting firewall:** prohibit tuning to these eight development cells; define how candidate rules are selected without development leakage; specify any nested split, synthetic calibration, theorem-only calibration, or held-out design.
6. **Exact gates:** preregistered survival, conditioning, SNR, baseline, robustness, and Oral/poster/stop criteria, including denominators and handling of null/unresolved points. Do not merely lower v2 thresholds after observing failure.
7. **Ablations and falsifiers:** tests that separately discriminate target mismatch, proof conservatism, aggregation mismatch, and true theory failure.
8. **Implementation contract:** files/functions to change, mandatory tests, artifact names/schemas, launch phases, one-shot locks, GPU plan, and stop conditions. Keep all server data/cache/log/output paths under `/mnt/sdb`, never the root disk.
9. **Paper strategy:** an honest assessment of whether an ICLR Oral trajectory remains plausible, the strongest defensible central claim, required figures/tables, and the minimum evidence package needed before writing that claim.
10. **Immediate next actions:** a numbered, dependency-ordered plan that Codex can execute without further interpretation, with explicit points at which it must return to you.

Do not give generic advice. Audit the actual code and artifacts, recompute any read-only aggregate you need from the archived Parquet/JSON files, and cite exact paths, fields, counts, and formulas. If you find an implementation error capable of invalidating the official scientific interpretation, distinguish it from harmless reporting defects and explain whether it changes the immutable v2 verdict or only motivates a new protocol.

Write your complete answer into exactly one repository-ready Markdown document named:

`analysis/GPTPRO_GREEN_V21_POSTMORTEM_DECISION_20260825.md`

The document should be self-contained, binding, and executable by Codex. End with one unambiguous directive chosen from:

- `AUTHORIZE_READ_ONLY_POSTMORTEM_ONLY`
- `AUTHORIZE_NEW_VERSION_PREPARE_ONLY`
- `STOP_GREEN_AND_REDIRECT_THEORY`

If you authorize a new version, do not authorize development or confirmation in the same step; require a clean implementation, exact tests, hashes, and a prepare audit first.
