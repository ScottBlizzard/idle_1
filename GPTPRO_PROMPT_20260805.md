# Prompt to submit to GPT Pro

You are acting as a hostile but constructive ICLR senior reviewer and research
strategist. Review the complete repository at:

**https://github.com/ScottBlizzard/idle_1** (branch: `main`)

Start with `GPTPRO_START_HERE.md`, then read every file it routes you to. In
particular, read `analysis/GPTPRO_REDTEAM_PACKET_20260805.md` in full before
forming a verdict. Inspect the actual theory, implementation, tests, aggregate
reports, and selected raw JSON evidence; do not assess the project from the old
manuscript title or a summary alone.

## Context and non-negotiable standard

This work is intended to compete for an **ICLR oral**, not merely achieve
acceptance. The July audit falsified the previous IVS-centered claim that
activation overlap is a general causal-validity certificate. The current
candidate rebuild distinguishes:

1. zero-order behavioral restoration;
2. local functional response agreement under an explicit intervention/probe law;
3. structural circuit or path restoration.

The proposed Interventional Response Signature (IRS) operationalizes level 2,
while composite split conformal audits compatibility with a declared target
reference distribution. The repository contains negative results: IRS does not
clearly outperform a fair single clean--corrupt interaction direction on the
standard IOI layer-stage diagnostic; the preregistered strict robustness gate
fails; and pABC corruption separates low IRS from strongly failed NMH recovery.
Do not hide, reinterpret, or excuse these results. A stronger replacement story
is allowed only if it is genuinely more important and defensible than the
falsified story.

## Required independent verification

Use current primary sources available as of **5 August 2026** to verify novelty,
especially the collision set listed in the red-team packet. Search for any
additional 2025--2026 work on mechanistic-intervention identifiability,
multi-mediator interactions, causal abstraction/scrubbing, representation
equivalence, off-manifold interventions, local functional equivalence, and
finite-difference or Jacobian-based mechanism diagnostics. Distinguish what you
verified from sources from what you infer.

Audit the mathematics rather than trusting theorem labels. Check:

- whether the zero-order non-identification and local transport statements are
  correctly scoped;
- whether the finite-radius forward/symmetric IRS estimands, normalizations, and
  bias claims align exactly with the code;
- whether the probe-law trace identity and concentration statement are correct
  and sufficiently nontrivial;
- whether the composite conformal implementation satisfies the stated
  exchangeability and split-independence requirements;
- whether context-matched probe construction or endpoint selection leaks
  information or weakens conformal coverage;
- whether Name Mover Head recovery is a valid independent structural witness in
  each corruption condition;
- whether any claim relies on layer-level pseudo-replication, post-selection, or
  an invalid statistical unit.

## Decisions you must make

1. Give probabilities for: **oral-worthy after a focused rebuild**, **poster-level
   contribution only**, and **no-go without a new central result**. Explain the
   dominant uncertainty.
2. Decide whether the behavioral/local-functional/structural equivalence
   hierarchy is materially new relative to the closest literature or merely a
   predictable synthesis.
3. State the strongest precise theorem that could make the hierarchy a genuine
   contribution. Include assumptions, estimand, conclusion, and why it is not
   already implied by standard Taylor/random-projection/non-identifiability
   arguments. If no such plausible theorem exists, say so explicitly.
4. Decide whether IRS should remain a headline named method, become only an
   operational witness inside a broader impossibility/equivalence paper, or be
   removed from the main contribution.
5. Select exactly **one** next pretrained-model experiment with the highest
   information value. Specify task, model(s), intervention unit, corruption,
   target probe law, structural ground truth/readout, fair baselines, statistical
   unit, decisive success threshold, and decisive failure threshold. Do not
   recommend generic scaling, more seeds, or several loosely related tasks.
6. Identify the three most likely reviewer attacks that would kill an oral bid,
   including any flaw you find in theory, normalization, conformal logic, or the
   interpretation of the negative results.
7. Give a concrete go/no-go decision tree for the next **7--10 GPU-days**. Every
   branch must resolve a named ambiguity; include explicit stop conditions.
8. List claims that are currently supported, claims that require one more result,
   and claims that must be prohibited.
9. Recommend the best final paper framing, title pattern, and a one-paragraph
   oral-level contribution statement. Do this only after the go/no-go analysis;
   do not use wording to compensate for missing novelty.

## Output format

Your entire final answer must be one self-contained Markdown document intended
to be saved verbatim as:

`analysis/GPTPRO_REVIEW_20260805.md`

Use exactly these top-level sections:

1. `# Executive Verdict`
2. `# Novelty Collision Audit`
3. `# Formal Theory Audit`
4. `# Empirical and Statistical Audit`
5. `# Main-Line Decision`
6. `# Single Highest-Value Next Experiment`
7. `# Seven-to-Ten GPU-Day Decision Tree`
8. `# Allowed, Conditional, and Prohibited Claims`
9. `# Recommended Framing`
10. `# Final Go/No-Go Checklist`

Include direct links to every external primary source used. Be decisive,
quantitative where possible, and adversarial. Do not praise the project, write a
generic review, or suggest manuscript polishing before resolving the scientific
go/no-go question.
