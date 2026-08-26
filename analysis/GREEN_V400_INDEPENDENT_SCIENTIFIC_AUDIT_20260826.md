# GREEN v4 Independent Scientific Audit — 2026-08-26

## Review setup

- Input scope: current manuscript, July/post-July audit trail, GREEN v2/v3/v4 decisions and artifacts, current source/tests, and server-side reproducibility checks.
- Audited baseline: clean commit `30e6b0fb9c0904142649b7d766f74ca9a30c7911`.
- Assessment boundary: read-only and outcome-blind. No real v4 endpoint, derivative, P13, development, or confirmation outcome was opened.
- Shared claim under audit: Transformer mechanism evidence should be organized as an identifiability hierarchy, culminating in a matched-bypass, four-branch Joint Witness whose finite-radius error is certified and whose detectability boundary is tested prospectively.
- Visible evidence: IVS/P0 results, restricted matched-bypass theory, v2/v3 frozen results, v4 scalar theorem and synthetic infrastructure, resource counts, code and tests.
- Missing evidence affecting confidence: real GPT-2 TensorProgram extraction/replay, graph-to-TransformerLens parity, non-vacuous real Joint Witness certificates, Boundary Transition results, sealed confirmation, and an independent current literature-priority search.

## Reviewer 1 — technical soundness emphasis

- Overall assessment: the four-branch mathematics and scalar certificate skeleton are credible, but the theorem is not yet connected to the repository's actual GPT-2 intervention.
- Who would be interested, and why: mechanistic-interpretability, causal-mediation, and verified-numerics researchers would care about a method that distinguishes output manipulation from identifiable path evidence.
- Major strengths:
  - the official scalar is now unambiguous: `PAT_J - PAT_B - TAR_J + TAR_B`;
  - J keeps selected-gate posts live, B freezes them to the same anchor, and both retain the residual bypass;
  - the toy nonlinear graph verifies `Psi'(0)=theta` and correctly shows that the legacy internal-subtraction curve can agree at first order while differing at second order;
  - interval jets, signed curvature, multi-radius intersection, dyadic partitions, and 384/512 nesting form a substantial scalar proof skeleton;
  - the one-shot scientific budget remains intact.
- Major concerns:
  - current prepare code does not construct the real PAT anchor by patching TAR block-8 `hook_mlp_out` into the corrupt prompt;
  - the real intervention requires block-10 `resid_mid`, selected block-10 MLP post activations, block 11, final LayerNorm, and contrast, but the current manifest records different/coarser hooks;
  - the current graph manifest is still a hand-written template, not a replayable graph;
  - Tensor-SSA is a schema scaffold: there is no production GPT-2 generator, deserializer/executor, semantic shape/type verifier, scalarization-root recomputation, or graph-to-model parity;
  - the real-model `Psi'(0)=theta` bridge remains untested.
- Technical failings that must be addressed before the case is established:
  1. build actual PAT/TAR × J/B programs from the repository hook semantics;
  2. serialize the frozen direction, selected gates, live/frozen policy, branch roots, final-token contrast, and exact tensor-store closure;
  3. replay the program independently and compare all four branches plus their derivative against TransformerLens and the component estimator;
  4. enforce operation counting and full 384/512 certificate evidence in the production executor.
- Assessment against review criteria:
  - originality: potentially strong as a combination; external priority is not assessable from local evidence alone;
  - scientific importance: high if the certificate changes mechanistic conclusions on real models;
  - interdisciplinary readership: plausible across interpretability, causality, and verification;
  - technical soundness: scalar layer supported, real-model closure unsupported;
  - nonspecialist readability: the branch semantics are clearer, but protocol/version status remains difficult to follow.
- Recommendation posture: GO for outcome-blind remediation; BLOCK formal outcome, development, and confirmation.

## Reviewer 2 — originality and scientific-importance emphasis

- Overall assessment: the strongest paper is no longer an IVS-score paper. Its high-upside contribution is an identifiability hierarchy for mechanistic evidence with a prospective certified boundary.
- Who would be interested, and why: researchers using patching, path interventions, JVP/AtP-style attribution, causal representation methods, or neural verification would care about knowing when an internal intervention is evidence for a mechanism rather than merely an effective edit.
- Major strengths:
  - v3 contains a striking empirical fact: the response-only joint estimator agrees with an independent AD target near numerical precision (median absolute error about `9.12e-9`), while all 80 componentwise box intervals still cross zero;
  - this cleanly separates estimator accuracy from set identification and motivates a direct relational Joint Witness rather than post-hoc covariance tuning;
  - matched bypass, response-only estimation, independent target evaluation, finite-radius certification, and a prospective detectability boundary create a conceptually stronger story than ordinary patch validation.
- Major concerns:
  - all individual ingredients have nearby precedents: path patching/mediation, JVP and integrated-gradient methods, relational verification, interval/Taylor methods, curvature bounds, and detectability analysis;
  - the claim that their exact combination is first is not assessable without a fresh external literature audit;
  - a certificate that merely adds an expensive error bar without changing any scientific conclusion will not sustain an Oral claim;
  - `basis-free`, `path-specific causal effect`, and `phase transition` can all be overstated unless tightly qualified and empirically established.
- Technical failings that must be addressed before the case is established:
  1. show real cases where patching/JVP/finite differences look decisive but the prospective certificate correctly marks them unresolved or wrong;
  2. validate those calls with an independent held-out causal target;
  3. compare against strong attribution and certified baselines;
  4. freeze and test the Boundary Transition without outcome-aware tuning;
  5. reproduce on sealed confirmation and preferably a second model or architecture.
- Assessment against review criteria:
  - originality: conditional high-upside, priority not yet independently established;
  - scientific importance: potentially outstanding if it changes what counts as mechanism evidence;
  - interdisciplinary readership: meaningful if framed as evidence/identifiability rather than a GPT-2 engineering artifact;
  - technical soundness: v3 point-estimation evidence is strong, v4 certificate evidence is absent;
  - nonspecialist readability: the central distinction between accuracy and identification can be made accessible and is the best narrative anchor.
- Recommendation posture: core theory direction GO; Oral-level claim CONDITIONAL; current evidence not yet sufficient.

## Reviewer 3 — numerical, protocol, and reproducibility emphasis

- Overall assessment: current tests establish implementation self-consistency, not formal-prepare completion or scientific success. The frozen v4.0 resource cap is impossible for the specified dense tail, and the v4.0.1 document remains a non-effective draft.
- Who would be interested, and why: verified-computation and reproducible-ML readers would care if exact graph provenance, directed rounding, fixed subdivision, and one-shot scientific firewalls are all enforced end to end.
- Major strengths:
  - exact IEEE-to-MPFR import, directed endpoints, domain guards, balanced reductions, partition-cover checks, multi-radius logic, and precision nesting are implemented at the scalar level;
  - row roles and authorization firewalls remain separated;
  - no real v4 scientific outcome was found, so remediation has not spent the one-shot budget;
  - the code now stops on shape-derived infeasibility instead of reporting the old hand-written 75M estimate.
- Major concerns:
  - independent resource count at sequence length 12 is `7,112,448` coefficient terms/branch/cell, `28,449,792` four-branch terms/cell, `341,397,504` conservative MPFR operations/cell, and `682,795,008` for the mandatory two-cell start of one radius;
  - block-11 dense computation is about 99.514% of the term count; exact sharing, pairwise reduction, or tensorization cannot legally reduce the frozen scalar count below 100M;
  - the proposed `64 full-tail-equivalent passes` is not defined or frozen in the current draft and cannot be approved without a synthetic tail benchmark;
  - before remediation, the width tolerance used `max(abs, rel*scale)` even though both tolerances must pass, the splitter used left-to-right DFS instead of `w(J)*wid(Q)` priority, and `_coverage_manifest()` returned `None` because its return block was mis-indented;
  - no compiled/reference bit-identity report, peak-memory report, benchmark-derived cap, or real TensorProgram exists.
- Technical failings that must be addressed before the case is established:
  1. finish the deterministic certificate-policy repairs and counterexample tests;
  2. generate the real TensorProgram and static counters;
  3. implement Python reference and compiled MPFR executors with bit-identical 384/512 outputs on outcome-free synthetic tail programs;
  4. pre-register and run a tail-shaped throughput/peak-memory benchmark;
  5. derive, audit, and explicitly approve the replacement cap before any real certificate;
  6. regenerate formal prepare in a new empty output root and re-audit it.
- Assessment against review criteria:
  - originality: numerical machinery alone is not the novelty; its mechanistic target may be;
  - scientific importance: depends on non-vacuity and prospective validation;
  - interdisciplinary readership: possible if the certificate is presented as scientific evidence control, not infrastructure;
  - technical soundness: current formal authorization is BLOCK;
  - nonspecialist readability: test counts and archived PASS labels can mislead unless engineering, formal, and scientific states are explicitly separated.
- Recommendation posture: GO only for local/outcome-blind remediation and synthetic benchmarking; BLOCK all real-row stages.

## Cross-review synthesis

- Consensus strengths:
  1. the theory has not been falsified or downgraded;
  2. v3 provides a rare and valuable accuracy-versus-identification gap;
  3. direct four-branch Joint Witness is a principled response to the failed componentwise box;
  4. protocol discipline preserved the scientific one-shot.
- Consensus technical risks:
  1. no real GPT-2 intervention program or replay executor;
  2. no TensorProgram-to-TransformerLens parity or actual-model composition identity;
  3. the current 100M resource contract is mathematically infeasible;
  4. a real certificate may still be vacuous or too slow after engineering repair;
  5. current manuscript text does not contain the GREEN v3/v4 story and cannot inherit its evidence.
- Where emphasis differs:
  - Reviewer 1 places greatest weight on the theorem-to-model bridge;
  - Reviewer 2 places greatest weight on whether the certificate changes a scientific conclusion and sustains Oral-level novelty;
  - Reviewer 3 places greatest weight on exact execution, frozen policy, and one-shot reproducibility.
- Broad-interest/significance readout: high-upside but not established. The best claim is not “a better validity score”; it is “mechanistic interpretation should progress from behavioral restoration to identifiable and prospectively certifiable path evidence.”
- Most important issues before the strong case is established:
  1. real replayable GPT-2 Joint Witness;
  2. non-vacuous certificate that prospectively distinguishes reliable from misleading ordinary evidence;
  3. frozen Boundary Transition and controls;
  4. sealed confirmation and broader replication;
  5. fresh prior-art audit and a full manuscript rewrite around the new mainline.

## Risk / unsupported claims

- Unsupported: IVS is a general counterfactual-validity measure.
- Unsupported: IVS invalidity proves mechanism bypass.
- Unsupported: a universal or sharp phase transition has already been observed.
- Unsupported: a real GPT-2 exact Joint Witness certificate already exists.
- Unsupported: Joint Witness is tighter than strong generic certified baselines.
- Unsupported: the boundary generalizes across tasks, models, or architectures.
- Not assessable from local materials: worldwide literature priority as of 2026.
- Unsupported at present: the project already has ICLR Oral-level evidence.

## Post-audit remediation record

The following outcome-blind defects found by the reviewers were fixed immediately after the audited baseline:

1. width stopping now requires both absolute and relative tolerances;
2. adaptive subdivision now uses curvature-weighted proof-width priority with frozen tie breaks;
3. the certificate-plan split-policy string now matches that rule;
4. `_coverage_manifest()` now returns from the correct function;
5. counterexample regression tests were added for the tolerance and subdivision-order defects.

Server verification after these repairs: `400 passed`. This is engineering evidence only and does not change the scientific BLOCK status.

The isolated compiled runtime under `/mnt/sdb` now contains GMP 6.3.0 and MPFR 4.2.1 built from hash-verified sources; the MPFR upstream check suite completed successfully. No scientific row was used.

## Final gate decision

- Core theory direction: **GO**.
- ICLR Oral narrative potential: **CONDITIONAL GO, high upside**.
- Outcome-blind graph/backend remediation: **GO**.
- Resource-only corrigendum in principle: **GO without theory downgrade**, but the current draft remains non-effective.
- Current formal prepare PASS: **BLOCK**.
- Real-row certificate: **BLOCK**.
- Development: **BLOCK**.
- Confirmation: **BLOCK**.
- Current IVS manuscript submission: **BLOCK; major rewrite or GREEN-centered replacement required**.

