# GREEN v4 Grant capture-semantics audit

Date: 2026-08-29  
Status: scientific binding resolved and implementation-ready; no untouched
outcome opened

## Primary-source finding

Grant et al. compare natural representations with corresponding intervened
representations. Their Figure 2 transformer comparison uses residual-stream
vectors at an explicitly selected layer and position. Appendix A.1.2 pairs
each intervention with its natural ground-truth vector, compares the
full-dimensional cohorts with Sinkhorn EMD, and uses natural--natural subsets
as the sampling baseline. The pinned author implementation is consistent with
those requirements.

Primary sources:

- Grant et al., *Addressing divergent representations from causal
  interventions on neural networks*, arXiv:2511.04638v5, Figure 2 and Appendix
  A.1.2: https://arxiv.org/html/2511.04638v5
- Author repository at commit
  `f2548d2ea9b4f4b87a87ba5d53db43838d15c521`, especially
  `divergence/divergence_utils.py`:
  https://github.com/grantsrb/rep_divergence/blob/f2548d2ea9b4f4b87a87ba5d53db43838d15c521/divergence/divergence_utils.py

## Applicability correction

GREEN installs the complete paired clean `resid_post` vector at the candidate
site. At that exact layer and token, paired MSE and full empirical-distribution
difference are therefore zero by construction. A finite split-sample EMD may
be nonzero, but it has the same sampling interpretation as the
natural--natural control and is not an informative intervention effect.

A downstream layer at the same token is not universally degenerate: it can
read unpatched earlier positions. It can nevertheless collapse in tasks where
the corruption is token-local and patching that token restores the entire
relevant prefix. Because one frozen definition must apply to both IOI and
Greater-Than, same-token downstream capture is not a robust common estimand.

## Frozen extension

The accepted diagnostic is a **Grant-style downstream contextual-divergence
extension**, not an exact replication and not an off-manifold proof:

- intervention: corrupt run with the complete clean center patched at candidate
  `blocks.{0..8}.hook_resid_post` and the task-defined candidate position;
- natural reference: the paired clean run for the same sealed prompt;
- measurement: `blocks.10.hook_resid_post` at the final prompt position, which
  must be strictly after the candidate position and strictly downstream in
  layer;
- contextual control: paired clean versus unpatched-corrupt states at the same
  measurement site;
- cohort: exactly one vector per planned prompt/site row, sorted by
  `site_row_id`, separately for every task, phase, and candidate layer;
- estimator: the pinned Grant metric panel using all rows in an even-sized
  cohort, with one deterministic disjoint half split. The patched and natural
  control comparisons use the same second-half prompt identities;
- reporting: preserve signed `emd - base_emd`, report paired MSE and companion
  costs, and keep the unpatched contextual control alongside it;
- role: descriptive cohort diagnostic only. It is never broadcast to rows,
  used for threshold fitting or row selection, or counted as a GREEN win;
- firewall: prediction route only, with no GREEN directions, endpoint
  directions, endpoint outcomes, or serialized raw activations.

Measuring at the final prompt position does not redefine the paper's held-out
transport endpoint: the Grant packet sees no held-out direction and produces
only cohort distribution diagnostics. The purpose is to ask whether an exactly
natural local state remains contextually natural after the corrupted
computation continues.

## Verification completed

The binding is machine-readable in
`configs/green_v400_grant_capture_spec.json`. The plan compiler hashes it into
every Grant job; the formal worker reconstructs each cohort from the sealed
queue; typed receipts and the append-only phase ledger require every phase's
Grant commitments before endpoint authorization.

Unit and firewall tests pass locally and in the isolated server runtime. A
frozen GPT-2 historical-prompt smoke on GPU 4 exercised 12 captures over layers
0, 4, and 8. Every clean, patched, and unpatched-control vector was finite with
width 768; the route was nondegenerate, used no untouched v4 row or direction,
and serialized no activation. Evidence is in
`analysis/green_v400_grant_capture_historical_smoke_20260829_v1.json`.

This resolves the former Grant scientific-binding blocker. It does not, by
itself, authorize development or confirmation outcomes.
