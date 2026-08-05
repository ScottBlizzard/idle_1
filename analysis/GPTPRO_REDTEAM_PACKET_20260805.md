# GPT Pro red-team packet: preserving an ICLR-oral-level main line

Date: 2026-08-05  
Project: intervention validity / mechanism restoration in transformers  
Requested role: skeptical ICLR oral-area reviewer and theory strategist

## 1. Non-negotiable objective

The paper must not be rescued by weakening its central theoretical or scientific
claim.  We may replace a falsified claim with a stronger and more defensible
one, but we must not hide negative results, relabel distributional overlap as
causal validity, or claim structural circuit identification from local
functional evidence.

Please decide whether the evidence below supports a genuinely oral-caliber
rebuild, and if so specify the narrowest theorem/experiment package that would
make it credible.  A blunt no-go verdict is preferable to optimistic wording.

## 2. What the July audit falsified

The old headline treated an activation-overlap score (IVS) as a general causal
intervention-validity certificate and suggested that low overlap explained
mechanism bypass.  Reanalysis falsified this:

- overlap labels changed under plausible corrupt, clean, mixture, and
  context-matched references;
- historical million-scale reconstruction z-scores were calibration artifacts;
- the pooled Name Mover Head AUROC was confounded by patch time and position;
- at a fixed IO position with all measured NMHs downstream, GPT-2 L4--L8 had
  high behavioral restoration but low NMH recovery, while overlap stayed near
  0.5 and had negligible prompt-level residual association with NMH recovery;
- supervised baselines did not support a special high-dimensional IVS advantage.

The safe retained claim is only that cross-fitted, conformal overlap can test
compatibility with an explicitly declared target reference distribution.

## 3. Current replacement theory

Cut a network at activation \(h\) and write the downstream computation as
\(F(h,b)\), where \(b\) contains unpatched bypass/downstream context.

1. **Zero-order non-identification.** Exact equality
   \(F(h_p,b_p)=F(h_\star,b_\star)\) places no upper bound on
   \(\|J_hF(h_p,b_p)-J_hF(h_\star,b_\star)\|\).  A smooth two-dimensional
   construction makes the gap arbitrarily large.
2. **Local transport.** If Jacobians differ by at most \(\epsilon_J\) and the
   two downstream maps have bounded Hessians, intervention-effect mismatch at
   radius \(r\) is at most
   \(\epsilon_J r + (H_p+H_\star)r^2/2\).
3. **IRS.** Interventional Response Signature compares paired forward or
   symmetric finite differences under an explicit probe law.  GPT-2 uses
   clean-reference forward chords and audits every endpoint with composite
   split conformal.
4. **Probe coverage.** With \(A=J_p-J_\star\) and
   \(M_Q=\mathbb E[uu^T]\), first-order IRS energy equals
   \(\mathrm{tr}(A^TA M_Q)\).  A single direction is rank one and has unavoidable
   blind spots; a multi-direction law only covers the subspace on which
   \(M_Q\) has positive spectrum.  A Hoeffding finite-sketch bound is included.
5. **Composite conformal validity.** Geometry fit, component normalization, and
   final scalar calibration use disjoint splits, giving standard marginal
   finite-sample coverage conditional on the declared reference law.

All three outputs remain separate: behavioral restoration \(R\), reference
admissibility \(S_Q\), and local response discrepancy \(G_Q\).  No universal
product score is claimed.

## 4. Positive evidence

### Controlled ground truth

- Analytic high/low restoration × high/low support × high/low IRS quadrants pass
  in 5/5 seeds.
- With a known gradient gap 4 in dimension 8, symmetric IRS recovers directional
  MSE 2.0 to numerical precision in 5/5 seeds.
- In an exact orthogonal blind-direction construction, the fixed single
  direction has RMSE 0 while isotropic IRS has RMSE \(\sqrt{2}\), in 5/5 seeds.
- A trained gated-transformer with explicit target support gives composite
  conformal donor-detection AUROC mean/min 0.9997/0.9993 over four independently
  initialized checkpoints.

### GPT-2 IOI, three prompt seeds

Frozen site: IO-token `resid_post`, L0--L8; both selected Name Mover Heads are
strictly downstream.  Each layer/seed has 80 prompts and eight identical paired
clean-reference probes.

- L4--L7 satisfy high restoration (>0.80 in every seed), low NMH recovery
  (<0.5), and endpoint acceptance >0.995.
- Mean normalized IRS is 0.866 in divergent L4--L7 versus 0.165 in aligned
  L0--L2 (about 5.3×).
- Fixed-effect layer-level residual Spearman IRS versus NMH recovery is -0.762,
  p=3.89e-6.
- Prompt-level residual association is weak: rho=-0.060, p=0.005.  IRS is a
  stage diagnostic here, not a prompt classifier.

## 5. Fair baseline and negative evidence

### Ordinary single clean--corrupt interaction direction

On the standard duplicate-S IOI corruption (seed 20260712):

| Quantity | IRS | Single direction |
|:--|--:|--:|
| Layer Spearman versus NMH recovery | -0.900 | -0.967 |
| Divergent/aligned layer ratio | 5.674 | 4.991 |
| Minimum target endpoint acceptance | 0.997 | 0.750 |
| Leave-one-context-out prompt RMSE when added to controls | 0.1673 | 0.1620 |

The single direction is at least as good for standard IOI stage ranking and
prompt prediction.  IRS has much broader target-admissible coverage, but that is
not labeled better mechanism prediction.

### Frozen probe sweep

- Interpolation 0.10 and 0.25, with 4/8/16 probes: all pass the preregistered
  gate (rank versus primary 0.967--1.000; rank versus NMH -0.950 to -0.983;
  endpoint acceptance 0.966--0.998).
- Interpolation 0.50: still tracks NMH (-0.833 to -0.850) but fails the frozen
  fine-ranking threshold (0.767--0.817 < 0.90).
- Interpolation 1.0 leaves support (minimum acceptance 0.132--0.345).
- Strict all-admissible-settings robustness verdict: false.

### Frozen corruption shift

Keep the clean target and clean-reference IRS law fixed; replace the clean IO
token either with repeated S or with an independent third name.  Recompute the
single clean--corrupt direction under each corruption.

| Quantity | IRS | Single direction |
|:--|--:|--:|
| Cross-corruption layer-rank correlation | 0.783 | 0.817 |
| Relative L2 change | 0.804 | 0.858 |
| Minimum endpoint acceptance | 0.997 | 0.613 |

IRS is non-inferior by the frozen rule but not clearly superior.  More
importantly, under third-name corruption, L4--L8 have restoration 0.920--0.939
and low IRS 0.149--0.176, while normalized NMH recovery is -1.403 to -2.311.
Thus local functional response agreement does not imply recovery of this
selected structural path.

## 6. Current honest verdict

The present experiments do **not** establish an oral-level empirical advantage
for IRS over the fair single-direction baseline.  They do establish two
separations:

1. behavioral equality does not imply local response-field equality;
2. local response-field equality does not imply restoration of a selected
   structural circuit path.

This suggests a possible stronger headline: an intervention-equivalence
hierarchy (behavioral / local-functional / structural) with strict
non-implications, rather than “IRS is a better mechanism metric.”  We are unsure
whether this is a genuine oral-caliber conceptual contribution or merely a
careful taxonomy around known non-identifiability.

## 7. Closest/latest collision set

- Grant et al., *Mechanistic Interpretability Needs Philosophy* / divergent
  internal representations, ICLR 2026: https://arxiv.org/abs/2511.04638
- Vaidyanathan et al., *The Curse of Multiple Mediators*, June 2026:
  https://arxiv.org/abs/2606.27510
- Sutter et al., nonlinear representation dilemma, NeurIPS 2025:
  https://papers.nips.cc/paper_files/paper/2025/hash/dbb98528c9870377f3f0d133aae6050b-Abstract-Conference.html
- Guo et al., IGSD/off-manifold intervention diagnostic, June 2026:
  https://arxiv.org/abs/2606.20678
- Lin and Liu, identification assumptions for mechanistic interpretability,
  May 2026: https://arxiv.org/abs/2605.08012

## 8. Decisions requested from GPT Pro

Please answer as a hostile but constructive ICLR senior reviewer:

1. Is the three-level equivalence hierarchy materially new relative to the
   collision set, or is it predictable synthesis?  Give a no-go probability.
2. What is the strongest theorem that would turn the hierarchy into a real
   contribution?  State it precisely enough that we can attempt a proof.
3. Which **one** next pretrained experiment has the highest information value:
   Greater-Than, factual recall, multi-site interventions, causal scrubbing, or
   something else?  Specify outcome, intervention, structural readout, and
   decisive result.
4. Should IRS remain a named method, become only an operational witness inside
   the hierarchy, or be removed from the headline?
5. Identify the three reviewer attacks most likely to kill an oral bid, including
   any flaw in the current theory/normalization/conformal logic.
6. Give a concrete go/no-go decision tree for the next 7--10 GPU-days.  Do not
   recommend generic scale-up or more seeds unless it resolves a named ambiguity.

The desired output is a decisive research strategy, not manuscript polishing.
