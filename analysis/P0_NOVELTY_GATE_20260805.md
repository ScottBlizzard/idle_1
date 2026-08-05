# P0 novelty gate: IRS versus single-direction interaction

## Frozen decision rule

The paper may claim that IRS is a useful multi-direction mechanism witness only
if it contributes evidence unavailable from the ordinary clean--corrupt
interaction direction under matched prompts, sites, outcomes, and compute.  A
broader definition alone is not sufficient.  The empirical distinction must be
either greater stability under corruption/probe-law shift, reliable detection
of a discrepancy missed by a single direction, or added predictive information
for an independent mechanism readout.

## First fair baseline (GPT-2 IOI, seed 20260712)

| Quantity | IRS | Single clean--corrupt direction |
|:--|--:|--:|
| Layer Spearman versus NMH recovery | -0.900 | -0.967 |
| Divergent/aligned layer ratio | 5.674 | 4.991 |
| Minimum target endpoint acceptance | 0.997 | 0.750 |
| Leave-one-context-out prompt RMSE when added to controls | 0.1673 | 0.1620 |

Control-only prompt RMSE was 0.1657; adding both scores gave 0.1634.  Therefore,
this experiment falsifies a claim of clear IRS superiority on the standard IOI
layer-stage diagnostic.  It does not establish equivalence in general because a
single corruption supplies only one direction and its endpoint is appreciably
less compatible with the declared clean target reference.

## Current evidence-safe interpretation

1. IRS is a distribution-indexed local response-field object; the ordinary
   interaction baseline is one member of that family.
2. On IOI, both detect the layer-level separation between behavioral restoration
   and downstream Name Mover Head recovery.
3. The presently demonstrated IRS advantage is target-admissible directional
   coverage, not better prompt-level prediction or better layer ranking.
4. Method novelty remains gated on the preregistered probe robustness and
   corruption-shift stress tests.  If neither separates IRS from the ordinary
   interaction baseline, the current method claim is not oral-level and must be
   redesigned before manuscript promotion.

The controlled analytic construction now verifies the rank-one blindness in
Theorem 4 over five seeds: the fixed single direction is orthogonal to the known
gradient discrepancy and has RMSE 0, whereas isotropic IRS recovers RMSE
\(\sqrt{2}\) exactly.  This establishes possibility/identifiability, not a
pretrained-model advantage; the latter remains gated below.

## Running stress tests

- Probe robustness: interpolation 0.1/0.25/0.5/1.0 and nested probe counts
  2/4/8/16, with the primary 0.25 and 8-probe setting frozen.
- Corruption shift: identical clean computation and clean-reference IRS probe
  law under duplicate-S and independent-third-name corruptions.  The
  single-direction baseline is recomputed from each corruption displacement.

No main-paper superiority statement is allowed until these gates are resolved.

## Quantitative gates frozen before inspecting outputs

Probe robustness passes only if every setting with at least four probes and
minimum endpoint acceptance at least 0.90 has (i) Spearman rank correlation at
least 0.90 with the primary layer ordering and (ii) Spearman correlation at most
-0.75 with NMH recovery.  Settings failing endpoint acceptance are evidence
about the admissible radius boundary, not robustness failures of the in-scope
estimand.

Across corruptions, IRS is considered non-inferior in stability only if its
cross-corruption rank correlation is no more than 0.05 below the single-direction
baseline and its relative L2 change is no more than 1.25 times the baseline.
Clear stability superiority requires either a rank-correlation gain of at least
0.10 or a relative-L2-change ratio no greater than 0.75, while retaining at least
0.90 endpoint acceptance.  Better admissible coverage is reported separately
and is not, by itself, labeled better mechanism prediction.

## Observed frozen-gate results

### Probe robustness

All settings at interpolation 0.10 and 0.25 with 4, 8, or 16 probes passed:
rank correlation with the primary layer ordering was 0.967--1.000, correlation
with NMH recovery was -0.950 to -0.983, and minimum endpoint acceptance was
0.966--0.998.  Interpolation 0.50 remained associated with NMH recovery
(-0.833 to -0.850) but failed the frozen primary-ordering requirement
(0.767--0.817).  Interpolation 1.0 had minimum acceptance only 0.132--0.345 and
is outside the declared admissible radius.  The strict all-admissible-settings
probe gate therefore **failed**.

### Corruption stability

| Quantity | IRS | Single direction |
|:--|--:|--:|
| Cross-corruption layer-rank correlation | 0.783 | 0.817 |
| Relative L2 change | 0.804 | 0.858 |
| Minimum endpoint acceptance | 0.997 | 0.613 |

The IRS/single relative-change ratio was 0.937.  IRS passed the frozen
non-inferiority rule but failed the clear-superiority rule.  Its admissible
coverage advantage is large and must remain separately labeled.

### Additional falsification exposed by pABC corruption

At L4--L8 under independent-third-name corruption, mean restoration remained
0.920--0.939 and IRS normalized RMSE was only 0.149--0.176, yet normalized NMH
recovery ranged from -1.403 to -2.311.  Thus local response-field agreement did
not imply restoration of the selected Name Mover Head path.  This does not
invalidate IRS as a local functional estimand; it does invalidate any wording
that treats IRS as a structural-circuit certificate.

## Final P0 gate verdict

**The current evidence does not establish an oral-level empirical advantage for
IRS over a fair single-direction interaction baseline.**  It does establish:

1. an exact rank-one blind-spot construction;
2. stable stage-level signal over a useful, conformal-admissible radius range;
3. substantially broader admissible probe coverage; and
4. empirical separation between behavioral, local-functional, and selected
   structural-path agreement.

The next decision is no longer a routine hyperparameter or replication choice.
It is whether to rebuild the paper around an intervention-equivalence hierarchy
with two non-implication results, or to seek a different pretrained task where
multi-direction coverage yields a genuine empirical advantage.  This is the
point at which independent GPT Pro red-team judgment is required.
