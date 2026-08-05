# GREEN Bridge Server Gate-04 Report

Date: 2026-08-05  
Theory base commit: `126556f`  
Execution commit: `98a64fef31f39ad23beb7278400cf3b1d3056d7d`  
Server: RTX 4090, physical GPU 4  
Terminal verdict: `STOP` at `04_HF_TL`

## Executive finding

The frozen GREEN bridge implementation reached the preregistered Hugging
Face-versus-TransformerLens equivalence gate and stopped mechanically. Across
the 32 hash-selected donor prompts, the maximum absolute discrepancy among the
100 final-position year logits was

```text
1.526e-04
```

against the frozen maximum of

```text
2.000e-05
```

No development or confirmation model responses were computed. Confirmation
remained locked. This is therefore a pre-response numerical-contract failure,
not an empirical failure of the matched-bypass bridge or the theory.

## Frozen environment actually observed

```text
Python              3.11.13
PyTorch             2.7.1+cu126
CUDA runtime        12.6
TransformerLens     3.6.0
transformers        5.13.0
NumPy               2.2.6
SciPy               1.15.3
pandas               2.2.3
pyarrow              19.0.1
GPU                  NVIDIA GeForce RTX 4090
model                openai-community/gpt2
model revision       607a30d783dfa663caf39e06633721c8d4cfcd7e
```

The repository was clean at launch. Model configuration matched the manifest:
12 layers, 768 residual width, 12 heads, 3072 MLP width, `LN`, `gelu_new`, and
LayerNorm epsilon `1e-5`.

## TransformerLens source identity

The installed PyPI wheel was not merely accepted by its version string. The
four implementation-critical source files were hashed and all hashes exactly
matched the sources at frozen commit `4a4dc26`:

```text
HookedTransformer.py       f80ee1ec42039a287a2b9366c75f98eec23ff33c6e941ffeee03f0374eb20af3
HookedRootModule.py        e7144971a973ec2d63bf7400db6443caba5d03f22f310f6789d52fa4a56ad245
components/mlps/mlp.py     615cb178d3ce65d8784af18dec86fbfe2b3957ddc02d3b99bdd2d45aa6759b32
utilities/addmm.py         f9e72f6a3d6c508814fa8e69918c20e1cb72cbc9ae7bcb1a1abb2476e246bc38
```

Thus the failure is not explained by a TransformerLens commit mismatch.

## Additional diagnostics

All diagnostics used only the same donor audit prompt family. They did not open
development or confirmation responses.

### Forced Hugging Face eager attention on GPU

For one representative donor prompt, forcing
`attn_implementation="eager"` produced:

```text
full-vocabulary max abs difference     9.1552734375e-05
100-year-logit max abs difference      7.62939453125e-05
```

This improves over the 32-prompt maximum but still fails `2e-5`.

### Forced eager execution on CPU

The same representative prompt on CPU float32 produced:

```text
full-vocabulary max abs difference     2.593994140625e-04
100-year-logit max abs difference      1.983642578125e-04
```

Moving the audit to CPU therefore does not solve the gate.

### Layerwise deterministic GPU comparison

With TF32 disabled, deterministic algorithms enabled, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, representative Hugging Face versus
TransformerLens residual differences after layers 0 through 10 were:

```text
3.052e-05, 6.104e-05, 1.335e-05, 1.144e-05, 9.537e-06,
2.289e-05, 3.052e-05, 2.289e-05, 2.861e-05, 3.433e-05,
6.104e-05
```

The corresponding full-vocabulary logit maximum was `1.373291015625e-04`.
The non-monotone, low-magnitude layerwise discrepancies are consistent with
different float32 attention/matmul accumulation orders rather than different
weights, hooks, model revision, or topology.

## What passed before the stop

- exact environment versions and CUDA runtime;
- exact TransformerLens source hashes;
- deterministic split materialization and hashes;
- tokenizer contract and all 100 unique suffix tokens;
- GPT-2 architecture/configuration checks;
- all 19 GREEN-specific CPU contract tests;
- clean repository and frozen source/requirements hashes.

## Frozen terminal artifact

```json
{
  "detail": "max error 1.526e-04",
  "first_failed_gate": "04_HF_TL",
  "schema_version": "green-bridge-terminal-v1",
  "time_utc": "2026-08-05T09:48:08Z",
  "verdict": "STOP"
}
```

## Decision required from GPT Pro

The executor will not silently relax `2e-5`, change versions, change the model,
or bypass the gate. Please determine the binding next action while preserving
the theorem and the full GREEN scientific design:

1. Is there an exact documented backend/configuration under the frozen package
   versions that should make native Hugging Face and TransformerLens float32
   year logits agree within `2e-5` on RTX 4090?
2. If the threshold is numerically unattainable because the two faithful
   implementations use different float32 accumulation orders, may the
   preregistration be amended *before any scientific responses* to a justified
   equivalence contract? Specify the exact metric and exact frozen threshold.
3. Should equivalence instead be checked at weights plus selected intermediate
   anchors and a task-relevant logit-contrast error, while retaining the
   existing strict manual-tail-versus-full-TransformerLens gate?
4. If none of those is defensible, confirm that the GREEN experiment remains a
   mandatory STOP and state whether a new theory-preserving bridge design is
   required.

Any authorization must give exact executable changes. Development and
confirmation will remain unopened until that decision is incorporated into a
new frozen manifest and committed.
