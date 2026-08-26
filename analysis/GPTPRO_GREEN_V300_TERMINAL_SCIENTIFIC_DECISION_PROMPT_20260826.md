# Prompt for GPTPro: GREEN v3.0.0 Terminal Scientific Decision

You are acting as the senior scientific architect and protocol authority for an
ICLR Oral-ambition mechanistic interpretability paper. Carefully inspect the
entire GitHub repository before answering:

- Repository: `https://github.com/ScottBlizzard/idle_1.git`
- Branch: `codex/green-v300`
- The branch-tip commit stated by Codex in the accompanying message is binding.

Do not answer from this prompt alone. Read the theory, all prior GPTPro
decisions, the implementation, the formal prepare artifacts, both implementation
corrigenda, the terminal evidence package, and the read-only diagnostic. At
minimum inspect:

- `analysis/GPTPRO_GREEN_V21_POSTMORTEM_DECISION_20260825.md`
- `analysis/CODEX_GREEN_V300_DEVELOPMENT_AUTHORIZATION_20260826.md`
- `analysis/CODEX_GREEN_V300_FLOAT64_RESPONSE_CORRIGENDUM_20260826.md`
- `analysis/CODEX_GREEN_V300_DEVELOPMENT_EXECUTION_INCIDENT_20260826.md`
- `analysis/CODEX_GREEN_V300_DEVELOPMENT_BOUND_AND_STABILITY_CORRIGENDUM_20260826.md`
- `analysis/GREEN_V300_FORMAL_PREPARE_20260826/`
- `analysis/GREEN_V300_DEVELOPMENT_TERMINAL_20260826/`
- `analysis/green_v300_postcorrigendum_diagnostic.py`
- all relevant `src/green_bridge_v300_*` and inherited v2 source files.

## Binding factual state

GREEN v3.0.0 formal prepare passed. The authorized one-shot development run
completed on 8 RTX 4090 GPUs. Confirmation has never been opened, inspected,
started, or authorized.

An initial implementation incident occurred before any result artifact was
written: the model-integrity context-manager verdict was read before `__exit__`.
This was repaired without seeing metric output. The formal development then ran
successfully. A later read-only audit found two implementation discrepancies:

1. the recoverable joint scalar bound omitted the frozen projected-direction
   and response-envelope terms;
2. the v3 coarse/fine stability merge omitted the inherited v2 denominator
   floor of `0.05`.

The original `POSTER_ONLY` evidence was independently archived before a scoped
corrigendum. The 80 transport records were reused byte-for-byte; only the 80
joint records were recomputed. The corrected projection/envelope contraction
changed joint bounds by at most `1.23046e-7` relatively, because the physical
joint direction already lies almost entirely in the probe frames. Restoring the
inherited stability definition changed the median symmetric change from
`0.936902` to `0.00182003`, so that gate now passes.

The corrected terminal verdict remains `POSTER_ONLY`. Exactly three gates fail:

- set-SNR-qualified cells: 0/10, required at least 6;
- detectability Spearman: `0.0161269`;
- detectability cluster-bootstrap 95% LCB: `-0.0457267`.

Everything else passes, often by a very large margin:

- 80 transport and 80 joint development records; 10/10 cells survive;
- 1,599/1,600 gate-system units recoverable, one unresolved;
- resolved coverage `0.9996875`; no numerical invalidity, structural
  contradiction, or theorem-bound failure;
- nonnull recoverability `0.9993746`;
- direct error median `3.00235e-6`, p90 `1.09067e-5`;
- joint error median `3.34602e-7`, p90 `1.04629e-6`;
- matched group-balanced RMSE `1.68485e-5` versus best frozen baseline
  `0.197069`, with gain `0.9999145` and bootstrap LCB `0.9999025`;
- null leakage median `8.64e-11`, p95 `8.23e-10`;
- coarse/fine cell Spearman `0.987879` and corrected median change `0.00182003`.

The point estimator is extraordinarily accurate, but every one of the 80
record-level signed joint confidence intervals crosses zero. Median record-level
set SNR is `0.102241`, maximum `0.652`, and median bound is `9.79144` times the
absolute target (worst `503.108`). Cell-level SNR ranges only from about `0.0445`
to `0.1594`. The present certificate composes ten gate contributions for each of
PAT and TAR using worst-case triangle addition, while signed signal components
cancel to a much smaller joint target.

The detectability panel is also saturated: curvature SNR already ranges from
about `9` to `6385`; 1,599/1,600 units are recoverable; 99.25% of direct errors
are at most `1e-4`. Thus the panel does not visibly span a recovery transition,
even though recovery performance itself is nearly perfect.

## Non-negotiable scientific constraints

1. Do not downgrade or remove the central theory/identification contribution.
   The target remains a defensible ICLR Oral-level paper.
2. Do not lower a failed threshold on the already observed v3 outcome, relabel
   the current verdict, open the sealed v3 confirmation under `POSTER_ONLY`, or
   use confirmation data for diagnosis.
3. Do not suggest cosmetic writing changes as the main rescue. The required
   decision is theoretical and experimental.
4. Distinguish a valid theorem/certificate improvement from invalid post-hoc
   threshold tuning. Any successor must receive a new protocol identity and
   fresh outcome-blind preregistration.
5. Codex is the front-line executor and may solve ordinary engineering details.
   Your task is to make the comprehensive scientific decision that genuinely
   needs senior theoretical judgment.

## Required analyses and decision

### A. Independent audit

Independently verify from repository evidence that the corrected v3 result and
the interpretation above are accurate. Check especially:

- the exact recoverable operator and joint-functional certificates;
- whether any still-unnoticed implementation error could explain the set-SNR
  width without changing the frozen estimand;
- whether PAT/TAR or ten-gate errors share structure that makes the current
  triangle-sum certificate unnecessarily loose;
- whether the detectability correlation gate is theoretically appropriate in a
  panel entirely inside the high-identifiability plateau.

If you identify an implementation error, give the exact corrected formula,
code-level location, required replay scope, and why the correction is not
outcome adaptation.

### B. Joint-certificate theory

Determine whether a proof-valid correlation-aware certificate can contract the
joint functional directly instead of summing twenty independent worst-case
scalar widths. Investigate, with explicit mathematics, possibilities such as:

- a single block-linear functional over all selected gates;
- a joint support function or ellipsoidal/Frobenius uncertainty set;
- shared finite-response remainder structure across gates and PAT/TAR;
- paired cancellation that can be certified rather than assumed;
- simultaneous or vector-valued concentration/rounding bounds;
- a direct joint finite-response/AD route that preserves the response-only point
  estimator and white-box use only as a certificate.

State the assumptions needed, prove or outline the key theorem rigorously, and
say whether existing saved v3 artifacts suffice for a read-only counterfactual
or whether a fresh development execution is required. Do not claim cancellation
unless it is certified uniformly.

### C. Detectability claim and boundary-spanning experiment

Decide how to preserve and strengthen the detectability theorem when the frozen
panel samples only a saturated recovery plateau. Consider whether the correct
successor claim is a two-regime theorem—boundary transition plus high-SNR
plateau—and design an outcome-blind experiment that deliberately spans the
identification boundary without tuning to observed target errors. Possible
exogenous axes include preregistered radius, controlled response attenuation,
probe-frame geometry/conditioning, or synthetic-to-natural bridge conditions.

Specify exactly which axis is scientifically legitimate, how it is generated,
what data split remains untouched, what statistic tests the transition, and how
the plateau result becomes positive evidence rather than a failed correlation.
Do not redefine the current v3 verdict post hoc.

### D. Current literature and novelty

Search and verify the most relevant literature available through 2026-08-26,
prioritizing primary papers and official proceedings. Map the proposed theorem
and experiments against the closest work on causal/mechanistic transport,
activation/path patching, local linearization, Jacobian or response
identification, uncertainty sets, and detectability/phase transitions. Provide
links/identifiers and a concrete novelty table. Explain what would make the
successor contribution credible at ICLR Oral level rather than merely a strong
poster.

### E. Binding successor protocol

Make one clear decision, not a menu of vague possibilities. If a defensible
Oral-level path exists, assign a new protocol identity (for example v3.1.0 or
v4.0.0) and provide a complete binding protocol:

- exact estimands and theorem statements;
- exact estimator and certificate formulas;
- exact data and split policy, including treatment of the still-sealed v3
  confirmation set;
- exact prepare/development/confirmation stages and firewalls;
- exact sample sizes, grouping, baselines, bootstrap units, metrics, thresholds,
  and multiplicity rules with outcome-blind justification;
- exact synthetic tests and implementation contracts;
- exact 8-GPU execution plan and expected artifacts;
- exact stop/open-confirmation criteria;
- what Codex may implement autonomously and the next point that requires GPTPro.

If no proof-valid Oral-level successor is defensible, state that explicitly and
explain why. Do not silently fall back to a weaker main line.

## Required output format

Return one complete, self-contained Markdown document named:

`GPTPRO_GREEN_V300_TERMINAL_SCIENTIFIC_DECISION_20260826.md`

The document must contain the independent audit, derivations, literature audit,
single binding decision, complete successor protocol, and an executable Codex
handoff checklist. Do not scatter required instructions across chat messages.
