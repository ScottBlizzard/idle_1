# GPT Pro review entry point

This repository is a frozen P0 research snapshot for an intended ICLR oral-level
paper on neural-intervention validity and mechanism restoration.  The July audit
falsified the previous IVS-centered causal-validity claim.  The current candidate
main line studies a three-level intervention-equivalence hierarchy:

1. zero-order behavioral restoration;
2. local functional response agreement under an explicit probe law;
3. structural circuit/path restoration.

The review must decide whether this rebuild is genuinely oral-caliber or merely
a careful synthesis of known non-identifiability results.  Negative results and
the frozen novelty gate are part of the evidence and must not be softened.

## Read in this order

1. [`analysis/GPTPRO_REDTEAM_PACKET_20260805.md`](analysis/GPTPRO_REDTEAM_PACKET_20260805.md)
   — self-contained claims, results, negative evidence, collision set, and the
   six decisions requested from GPT Pro.
2. [`analysis/P0_NOVELTY_GATE_20260805.md`](analysis/P0_NOVELTY_GATE_20260805.md)
   — thresholds frozen before the stress tests and the resulting negative gate.
3. [`analysis/IRS_THEORY_P0.md`](analysis/IRS_THEORY_P0.md)
   — formal P0 theory, scope limitations, and claim--evidence map.
4. [`analysis/P0_EXECUTION_20260804.md`](analysis/P0_EXECUTION_20260804.md)
   — July-audit execution record and the evidence that invalidated the old line.
5. [`analysis/p0_irs_gpt2_aggregate.md`](analysis/p0_irs_gpt2_aggregate.md),
   [`analysis/irs_vs_single_direction.md`](analysis/irs_vs_single_direction.md),
   and [`analysis/p0_irs_stress_summary.md`](analysis/p0_irs_stress_summary.md)
   — compact empirical summaries.
6. [`analysis/P0_GOAL_COMPLETION_AUDIT.md`](analysis/P0_GOAL_COMPLETION_AUDIT.md)
   — requirement-by-requirement verification of the P0 stopping condition.

Then inspect the implementation and tests:

- `src/validity_crossfit.py`, `src/test_validity_crossfit.py`;
- `src/interventional_response.py`, `src/test_interventional_response.py`;
- `src/run_irs_analytic_synthetic.py`;
- `src/exp_p0_irs_gpt2.py`;
- `src/exp_p0_single_direction_int_gpt2.py`;
- `src/exp_p0_irs_probe_sweep_gpt2.py`;
- `src/exp_p0_irs_corruption_shift_gpt2.py`;
- `src/analyze_p0_irs.py`, `src/analyze_irs_vs_single_direction.py`, and
  `src/analyze_p0_irs_stress.py`.

Selected raw JSON evidence is intentionally committed under `outputs/` even
though the directory is normally ignored.  It includes the five analytic seeds,
three GPT-2 IRS seeds, the fair single-direction baseline, probe sweep,
corruption shift, reference-distribution and temporally eligible NMH audits, and
the four independently initialized trained synthetic conformal runs.

## Output contract

Return one self-contained Markdown document intended to be saved as:

`analysis/GPTPRO_REVIEW_20260805.md`

Do not distribute the final verdict across multiple messages.  Follow the exact
review scope and output structure in `GPTPRO_PROMPT_20260805.md`.
