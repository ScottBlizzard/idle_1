# GREEN v4 Grant capture-semantics audit

Date: 2026-08-29  
Status: prepare-only scientific blocker; no untouched outcome opened

## Primary-source finding

Grant et al. define divergence between natural representations and the
corresponding intervened representations.  Their Figure 2 transformer
comparison takes both vectors from the residual stream at the intervention
position.  Appendix A.1.2 further requires a corresponding natural
ground-truth vector for each intervention, compares the full-dimensional
natural and intervened distributions with Sinkhorn EMD, and uses a
natural--natural comparison as the sampling baseline.

The pinned author implementation is consistent with that description:
`filter_by_layer_and_position` accepts tensors with shape
`[batch, layer, position, feature]`, selects an explicit layer and position,
and `collect_divergences` computes paired MSE plus distributional metrics and
their natural--natural controls.

Primary sources:

- Grant et al., *Addressing divergent representations from causal
  interventions on neural networks*, arXiv:2511.04638v5, especially Figure 2
  and Appendix A.1.2:
  https://arxiv.org/html/2511.04638v5
- Author repository, pinned commit
  `f2548d2ea9b4f4b87a87ba5d53db43838d15c521`, especially
  `divergence/divergence_utils.py`:
  https://github.com/grantsrb/rep_divergence/blob/f2548d2ea9b4f4b87a87ba5d53db43838d15c521/divergence/divergence_utils.py

## Consequence for the frozen GREEN estimand

GREEN v4 patches an entire `resid_post` vector.  At the patch site, the
ordinary clean-to-corrupt patch therefore installs an exactly natural clean
activation.  A literal Grant-at-the-intervention-position diagnostic on that
ordinary patch is degenerate by construction: the intervened vector is the
corresponding natural vector.  This differs materially from the coordinate,
subspace, SAE-reconstruction, and mean-direction interventions studied by
Grant et al.

GREEN's local response probes instead evaluate `center + direction` states.
Those states are nontrivial interventions, but the currently sealed Grant
cohort jobs do not bind whether their intervened cohort contains:

1. the ordinary full-vector patched centers;
2. all eight `center + green_direction` probe states per row;
3. one deterministic aggregation of those eight probe states; or
4. representations captured at a strictly downstream site after the patch.

Choices 2--4 change the scientific estimand.  Choice 4 is useful for contextual
transport but is not the literal intervention-position diagnostic described
for Figure 2 and risks becoming a second endpoint-like measurement.  The
existing queue records only a phase-by-layer cohort and does not bind any of
these choices, a capture position, or a natural-ground-truth pairing rule.

## Fail-closed decision

The Grant metric implementation and isolated-runtime parity remain valid, but
the untouched activation-collection route is not scientifically sealed.
`grant_divergence` must therefore remain
`SCIENTIFIC_BINDING_REQUIRED`, and the plan must remain blocked by baselines,
until one later binding decision chooses one of the following:

- **N/A / literal-source option:** report that full-residual patching is
  exactly natural at the intervention site, do not claim a Grant baseline win,
  and remove it from required execution; or
- **GREEN-probe extension:** explicitly define the intervened cohort,
  natural-ground-truth pairing, measurement hook/position, aggregation over
  directions, prediction-only firewall, and reviewer-facing label as an
  extension rather than an exact replication.

No result-dependent choice is allowed.  This blocker must be resolved before
development outcomes, together with the separate development-authorization
barrier already imposed by the binding corrigendum.
