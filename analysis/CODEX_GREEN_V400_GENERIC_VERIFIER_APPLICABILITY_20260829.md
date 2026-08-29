# GREEN v4 generic-verifier applicability audit

Date: 2026-08-29  
Status: documented generic-tail applicability failure; not the same GREEN estimand; no scientific outcomes opened

## Authority and pin

- Verifier: official `Verified-Intelligence/auto_LiRPA` source, commit `5a098e8f9fb5786a428a024981d833d303921f2d` (June 2026 release), package version `0.7.2`.
- The official project supports PyTorch 2.x and contains operators for GELU, softmax, and LayerNorm. Its language Transformer example nevertheless defaults to a `no_var` LayerNorm variant.
- Model: `openai-community/gpt2` at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Historical-only input: the pre-existing P0 grammar from 2026-07-12. The untouched v4 manifest was not loaded.

## Generic single-tail adapter (not the GREEN four-branch estimand)

The adapter starts at the full IO-position `resid_post` vector after layer 8, retains every other sequence-position activation as a constant, evaluates blocks 9--11, applies the original final LayerNorm and tied LM head, and returns the Mary-minus-John logit contrast. Evaluation-only dropout operators were removed, but no active mathematical operation or weight was changed. This reproduces one ordinary response tail only. It does **not** implement the scalar-control matched-bypass functional `PAT_J-PAT_B-TAR_J+TAR_B`, so it is not a same-estimand GREEN comparator.

The reconstructed single tail and full Hugging Face model agree to absolute error `6.866455078125e-05` in float32, below the frozen `1e-4` trace-equivalence threshold. Therefore the following failures are valid evidence about generic verification of that tail, but not evidence that a generic verifier failed on GREEN's four-branch scalar-control estimand.

## Reproduced failures

1. `IBP`, `epsilon=1e-5`: exact standard LayerNorm is decomposed into mean, square, variance, square root, and reciprocal nodes. Dependency loss makes the propagated variance lower bound non-positive; `BoundReciprocal` stops with `Only positive values are supported in BoundReciprocal`.
2. `backward`/CROWN, `epsilon=1e-7`: the verifier warns that it is creating `12288 x 12288` identity matrices for perturbed intermediate nodes, then fails with a `196608` versus `12288` intermediate-bound shape mismatch.

The failures are recorded verbatim in:

- `analysis/green_v400_auto_lirpa_layer8_ibp_eps1e-5_20260829.json`
- `analysis/green_v400_auto_lirpa_layer8_crown_eps1e-7_20260829.json`

## Binding decision

This is an honest N/A record for the current generic single-tail adapter. It does **not** satisfy a same-estimand generic-verifier comparison, does **not** produce a prediction, and must never be counted as a GREEN win. Replacing standard GPT-2 LayerNorm with the example's `no_var` approximation would change the model and is forbidden. A generic verifier becomes a primary comparator only after a scalar-`t`, four-branch `Psi` wrapper is implemented and passes trace equivalence; otherwise it remains N/A.
