# GREEN v4 primary-source baseline pin

Date: 2026-08-29  
Status: implementation evidence only; no scientific outcome authorization

## AtP*

- Primary paper: Kramár, Lieberum, Shah, and Nanda, *AtP*: An efficient and scalable method for localizing LLM behaviour to components*, arXiv:2403.00745.
- Equation (4): ordinary AtP is the clean-state gradient dotted with the noise-minus-clean activation change.
- Q/K fix: for query and key nodes, recompute the finite attention-probability change exactly and linearize only the computation after the softmax.
- GradDrop, equation (11): run one backward pass per layer with that layer's residual contribution gradient zeroed, average the absolute estimates, and multiply by `L/(L-1)`.
- Applicability constraint stated by the paper: the method targets large sweeps of fine-grained nodes. For full residual-stream sites, the authors warn that linearization can be poor and say ordinary activation patching is often cheap enough; Appendix C.1 treats a stop-gradient layer-normalization variant as future empirical work.
- Consequence for GREEN v4: the current IOI challenge patches one full `resid_post` vector, so finite activation patching is the fair primary comparator under the frozen coarse-site estimand. This is a scope decision, not a claim that finite patching executes, supersedes, or beats AtP*. A component-level AtP* result may be added only under its own node semantics; a residual-vector first-order score must not be relabeled as complete AtP*.
- No author-maintained implementation was linked by the paper. `koayon/atp_star` at commit `07e323537ad8a55f0b69f73533f26dab15d69836` is explicitly an unofficial, incomplete replication and is not parity authority.

## Grant et al. divergence

- Primary paper: Grant, Han, Tartaglini, and Potts, *Addressing Divergent Representations from Causal Interventions on Neural Networks*, ICLR 2026 Oral, arXiv:2511.04638 / OpenReview `cZrTMqYVL6`.
- Author repository: `https://github.com/grantsrb/rep_divergence`, pinned at `f2548d2ea9b4f4b87a87ba5d53db43838d15c521`.
- Authority files: `divergence/divergence_utils.py` and the experiment notebooks under `divergence/`.
- Capture semantics: the paper's transformer comparison uses residual-stream vectors at the intervention position; Appendix A.1.2 pairs each intervened vector with the natural ground-truth vector it is meant to approximate. The author utility accepts `[batch, layer, position, feature]` tensors and explicitly filters layer and position before analysis.
- Primary cohort metric: featurewise-standardize natural and intervened cohorts separately using the repository's `torch.std` convention, compute `geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.05)`, and divide by `sqrt(d)`.
- Companion metrics: paired MSE, optimal and nearest-neighbor cosine/correlation cost, and optimal and nearest-neighbor MSE cost.
- Required control: repeat distribution metrics between two natural-data subsets (`base_*`).
- Applicability constraint: this is a cohort-level divergence diagnostic, not a per-row certificate and not by itself a test of whether divergence is harmless or pernicious.
- GREEN firewall binding: only development/prediction activations may enter this baseline. Endpoint directions, endpoint activations, transport outcomes, and NMH outcomes remain unavailable.
- Full-vector caveat: GREEN's ordinary clean-to-corrupt `resid_post` patch is exactly natural at the intervention site by construction. The frozen informative comparison is explicitly labelled a Grant-style extension: capture at `blocks.10.hook_resid_post` at the final prompt position, with paired clean and unpatched-corrupt contextual controls; see `analysis/CODEX_GREEN_V400_GRANT_CAPTURE_SEMANTICS_AUDIT_20260829.md`.

## Readiness rule

Source pinning does not make a baseline `READY`. Readiness additionally requires task serialization, deterministic tests, historical-row resource measurement, and isolated-runtime parity. No untouched outcome may be opened on the strength of this note.
