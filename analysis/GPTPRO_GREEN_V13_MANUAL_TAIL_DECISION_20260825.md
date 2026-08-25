<!-- filename: analysis/GPTPRO_GREEN_V13_MANUAL_TAIL_DECISION_20260825.md -->

# GPTPRO GREEN v1.3 Manual-Tail Binding Scientific and Protocol Decision — 2026-08-25

## Document status

| Field                                                   | Binding value                                                                  |
| ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Repository                                              | `ScottBlizzard/idle_1`                                                         |
| Reviewed repository commit                              | `b8b5f7b509abda310f3e2414e0552a644a12b4bf`                                     |
| Stopped server execution commit                         | `ed4b3b4c55ba2c7acfda1291b4814957ce90c845`                                     |
| Stopped protocol identity                               | `green-bridge-v1.3`                                                            |
| Stopped attempt                                         | attempt 1, retry forbidden                                                     |
| Existing terminal result                                | valid and immutable `STOP` for v1.3                                            |
| Scientific-response exposure                            | none                                                                           |
| Binding classification under the requested alternatives | **Option 2: pre-scientific technical implementation/equivalence-audit defect** |
| Fresh corrected identity                                | `green-bridge-v1.3.1`                                                          |
| Fresh attempt                                           | attempt 1 under v1.3.1, retry forbidden                                        |
| New GPU execution                                       | **authorized exactly once, subject to every prerequisite in this document**    |
| Structural-envelope theory                              | unchanged                                                                      |
| Basis-free ambient rank-one operator                    | unchanged                                                                      |
| Fixed-rank donor PCA                                    | permanently terminated; restoration forbidden                                  |

---

# 1. Binding executive verdict

The existing v1.3 output root and its `STOP` are valid, terminal, and immutable for the identity `green-bridge-v1.3`, attempt 1. They must not be resumed, overwritten, deleted, reinterpreted as a pass, or treated as a failed trial that may simply be repeated. The server report records a clean launch, attempt index one, retry disabled, the first failure at `06_MANUAL_TAIL`, and no development or confirmation response. ([GitHub][1])

The v1.3 `STOP` is **not**, however, a scientific failure of:

* the matched-bypass identification theorem;
* the basis-free ambient rank-one path operator;
* the exact LayerNorm structural envelope;
* the gate set;
* the intervention sites;
* the radii;
* the independent residual-bypass-subtracted target;
* the preregistered estimator;
* the development or confirmation thresholds.

The failure occurred in a pre-scientific executable-equivalence audit. The audited “manual tail” and the independent full-model hook path did not execute the same final unembedding operation:

1. the full-model reference ran the pinned TransformerLens final endpoint over the full sequence and full vocabulary;
2. the manual tail first selected the final token and 100 output columns, then used a separate matrix-multiplication expression;
3. the manual tail also omitted the TransformerLens output-softcap call from its executable endpoint.

These expressions are algebraically equivalent only in exact arithmetic when softcapping is inactive. They are not the same IEEE-float32 operation graph. TransformerLens explicitly uses `F.linear(residual, W_U.T.contiguous(), b_U)` for its unembedding, and the full model applies the output softcap before returning logits. ([GitHub][2])

The observed errors,

[
6.103515625\times 10^{-5}=2^{-14}
]

and

[
7.62939453125\times 10^{-5}=5\cdot 2^{-16},
]

are quantized float32-scale discrepancies consistent with differing GEMM shape, layout, primitive, and accumulation order. They were measured only after conversion of already-computed outputs to float64, so the conversion used by the audit did not create them.

The large reported “derivative-relative” values are not evidence of a derivative mismatch. The v1.3 code did not compute derivatives in that check. It computed an endpoint-effect relative error whose numerator algebraically reduces to the same raw manual-versus-full logit discrepancy. At the center condition, the reference effect is zero and the denominator is replaced by the hard floor (10^{-5}), causing the raw logit discrepancy to be amplified into the reported value (25.895). This metric is mathematically ill-conditioned near zero and incorrectly named and interpreted as a derivative audit. The old failure nevertheless remains valid: neither its raw-logit failure nor its malformed relative check may be retroactively waived.

A fresh, separately versioned, one-shot v1.3.1 attempt is authorized solely to:

* replace the faulty manual and target endpoints with the exact pinned TransformerLens unembedding endpoint;
* retain raw-logit equivalence as the primary binding audit;
* replace the incorrectly named endpoint-relative quantity with an actual central-finite-difference derivative comparison;
* handle near-zero derivative reference norms through a proof-derived absolute bound, without deleting the center condition or relaxing either frozen threshold;
* establish the root cause through stagewise GPU traces before any scientific response.

No scientific design change is authorized.

---

# 2. Evidence reviewed

The following files at reviewed commit `b8b5f7b509abda310f3e2414e0552a644a12b4bf` were reviewed as the binding evidence boundary:

1. `analysis/GPTPRO_GREEN_GATE08_V12_DECISION_20260805.md`
2. `analysis/GREEN_SERVER_V13_PREPARE_STOP_20260825.md`
3. `src/exp_green_bridge_gpt2.py`
4. `src/green_bridge_tail.py`
5. `src/green_bridge_path_target.py`
6. `src/green_bridge_structural_frame.py`
7. `src/green_bridge_whitebox_audit.py`
8. `src/green_bridge_numerics.py`
9. `src/matched_bypass_gate.py`
10. `src/green_bridge_spec.py`
11. `src/test_green_bridge_contract.py`
12. `analysis/GPTPRO_GREEN_GATE04_DECISION_20260805.md`
13. `analysis/GREEN_SERVER_GATE04_20260805.md`
14. `analysis/GREEN_SERVER_GATE08_V12_20260805.md`

The v1.2 binding decision permanently terminated fixed-rank donor PCA while preserving and strengthening the main theory into a basis-invariant ambient rank-one path operator, made probe-complete by an exact five-vector LayerNorm structural envelope. That decision explicitly preserved the actual MLP gate coordinates, block-10 `resid_mid` intervention, matched control, independent target, residual-bypass subtraction, phase separation, confirmatory statistics, and ICLR Oral-level ambition. ([GitHub][3])

The v1.3 server report establishes that:

* the server execution was clean;
* attempt index was one;
* retry was false;
* Gate-04 passed;
* the same-TransformerLens no-op maximum error was exactly zero;
* the full structural preflight passed;
* the first failure was `06_MANUAL_TAIL`;
* neither development nor confirmation started;
* no scientific response was observed;
* a durable `STOP` result was written. ([GitHub][1])

The active specification freezes float32 execution, the model and TransformerLens revisions, the intervention geometry, the ten gate indices, the (4/5/14) structural-frame dimensions, the (0.20) radii, the raw tail threshold (2\times10^{-5}), and the derivative-relative threshold (10^{-4}). Its donor-PCA material is explicitly historical and inactive. ([GitHub][4])

---

# 3. Exact computation trace

## 3.1 `capture_tail_anchor`

For a token tensor with shape ([B,P]), `capture_tail_anchor`:

1. defines every final position as (P-1);
2. optionally patches only the final-position block-8 MLP output;
3. runs the complete pinned TransformerLens model through `model.run_with_cache`;
4. caches:

   * `blocks.8.hook_mlp_out`;
   * `blocks.10.hook_resid_mid`;
   * `blocks.10.mlp.hook_pre`;
   * `blocks.10.mlp.hook_post`;
   * `blocks.10.hook_resid_post`;
5. receives the model’s full output logits;
6. calls `gather_year_logits` to select the final position and the 100 year-suffix output coordinates;
7. stores those raw year logits as `anchor.year_logits`.

The anchor logits therefore come from the full TransformerLens endpoint:

[
\operatorname{ln_final}
\rightarrow
\operatorname{Unembed}
\rightarrow
\operatorname{apply_softcap}
\rightarrow
\text{final-position gather}
\rightarrow
\text{100-year-logit gather}.
]

The code performs the full forward before gathering. ([GitHub][5])

## 3.2 `gather_year_logits`

`gather_year_logits` does not perform any model arithmetic. It:

[
L\in\mathbb R^{B\times P\times V}
\mapsto
L[\text{rows},\text{final positions},:]
\mapsto
L_{\text{year}}\in\mathbb R^{B\times100}.
]

The unused `model` argument has no numerical effect. The function selects the final token first and then selects the 100 suffix-token coordinates from an already-computed logit tensor. ([GitHub][5])

`gather_year_logits` is not the source of the mismatch. Both the anchor path and `full_hook_endpoint` use it after the full model has produced logits.

## 3.3 `GreenBridgeTail.evaluate_physical`

For physical residual intervention (\delta r), selected-gate intervention (z), and an anchor, `evaluate_physical` executes:

1. cast (\delta r) to the anchor residual dtype;
2. clone the cached block-10 `resid_mid`;
3. add (\delta r) only at the final position;
4. call the actual block-10 `ln2`;
5. compute block-10 MLP preactivation with the same pinned TransformerLens `batch_addmm`;
6. apply the requested `path`, `control`, or `joint` (z)-intervention;
7. apply the actual GPT-2 activation function;
8. anchor every omitted final-position gate to the cached postactivation;
9. leave only the authorized live gate or gate set;
10. compute the MLP output with the same pinned `batch_addmm`;
11. form `resid_post = resid_mid + mlp_out`;
12. optionally subtract the direct residual bypass;
13. execute the complete block 11;
14. execute the model’s final LayerNorm;
15. **diverge from TransformerLens at the unembedding endpoint**.

The divergent statements are:

```python
final = normalized_final[rows, positions, :]
W_selected = self.model.W_U.index_select(1, self.suffix_ids)
logits = final @ W_selected
if getattr(self.model, "b_U", None) is not None:
    logits = logits + self.model.b_U.index_select(0, self.suffix_ids)
return logits
```

Thus the manual path slices the sequence and vocabulary before the matrix multiplication and uses `@` against the selected (768\times100) matrix. It does not call `model.unembed`, and it does not apply `apply_softcap`. ([GitHub][5])

The preceding block-10 and block-11 computations use the same actual TransformerLens modules and the same `batch_addmm` implementation used in the pinned model. TransformerLens itself computes MLP input and output projections with `batch_addmm`, applies `ln2`, and forms `resid_post = resid_mid + mlp_out`. ([GitHub][6])

## 3.4 `full_hook_endpoint`

`full_hook_endpoint` executes the complete model from tokens and installs independent hooks that:

* add (x) at `blocks.10.hook_resid_mid`;
* add (z) at `blocks.10.mlp.hook_pre`;
* anchor omitted gates at `blocks.10.mlp.hook_post`;
* optionally subtract the direct residual bypass at `blocks.10.hook_resid_post`;
* verify that no undeclared tensor entries changed;
* verify exact hook invocation counts.

It then calls:

```python
logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
return gather_year_logits(model, logits, positions, suffix_tensor)
```

Consequently, its endpoint is the pinned TransformerLens endpoint, not the selected-column manual expression. ([GitHub][7])

The pinned model’s final forward path is:

```python
residual = self.ln_final(residual)
logits = self.unembed(residual)
logits = apply_softcap(logits, self.cfg.output_logits_soft_cap)
```

and `Unembed.forward` is:

```python
result = F.linear(residual, self.W_U.T.contiguous(), self.b_U)
```

over the full ([B,P,768]) tensor and full vocabulary. ([GitHub][2])

## 3.5 `_tail_preflight_v13`

The v1.3 structural preflight constructs the first gate’s five-dimensional structural frame and evaluates these five frozen raw conditions:

[
\begin{aligned}
C_0&=(\mathrm{path},x=0,z=0),\
C_1&=(\mathrm{path},x=h_xe_0,z=0),\
C_2&=(\mathrm{path},x=0,z=h_z),\
C_3&=(\mathrm{path},x=h_xe_1,z=h_z),\
C_4&=(\mathrm{control},x=h_xe_2,z=h_z),
\end{aligned}
]

where (h_z=0.20) and (h_x) is the frozen structural residual radius. ([GitHub][7])

For every condition it computes:

[
E_{\infty,i}
============

|M_i-F_i|_\infty,
]

where (M_i) is the manual tail and (F_i) is the full-hook endpoint.

The raw gate is validly defined as:

[
\max_i E_{\infty,i}
\le 2\times10^{-5}.
]

It failed with:

[
\max_i E_{\infty,i}
===================

7.62939453125\times10^{-5}.
]

The function also computes:

```python
manual_delta = manual - anchor.year_logits
full_delta = full - anchor.year_logits
relative = norm(manual_delta - full_delta) / max(norm(full_delta), 1e-5)
```

and labels the result a derivative-relative error. ([GitHub][7])

This second quantity is not a derivative.

---

# 4. Exact root cause of the raw-logit discrepancy

## 4.1 Reference and manual maps

Let:

* (T(x,z)\in\mathbb R^{B\times P\times768}) be the tail computation through final LayerNorm;
* (W_U\in\mathbb R^{768\times V});
* (b_U\in\mathbb R^V);
* (S) be the final-position and 100-year-coordinate gather;
* (\mathcal C) be TransformerLens output softcapping;
* (\operatorname{FL}) be the pinned `F.linear` operation.

The full-hook reference is:

[
Y_F(x,z)
========

S,
\mathcal C
\left(
\operatorname{FL}
\left(
T_F(x,z),
W_U^\top_{\mathrm{contiguous}},
b_U
\right)
\right).
]

The old manual endpoint is:

[
Y_M^{\mathrm{old}}(x,z)
=======================

T_M(x,z)*{\mathrm{final}}
,W*{U,S}
+
b_{U,S}.
]

In real arithmetic, and only when (\mathcal C) is the identity,

[
S(TW_U+b_U)=T_{\mathrm{final}}W_{U,S}+b_{U,S}.
]

IEEE float32 does not guarantee equality between those executable programs. The following are changed simultaneously by the old manual expression:

* matrix-multiplication primitive;
* right-hand operand layout;
* explicit contiguous transpose;
* output width (V) versus (100);
* flattened leading dimension (B\cdot P) versus (B);
* sequence slicing before versus after the GEMM;
* vocabulary slicing before versus after the GEMM;
* softcap inclusion versus omission.

TransformerLens’ source explicitly documents that its `F.linear` and contiguous transpose are chosen to preserve the intended linear-layer memory layout and accumulation behavior. ([GitHub][2])

## 4.2 Statement-level defect

The exact source-code defect is therefore:

> `GreenBridgeTail.evaluate_physical` and `evaluate_joint_target` implemented the final unembedding through an independently rearranged selected-column matrix multiplication rather than executing the frozen TransformerLens endpoint and gathering afterward.

The raw discrepancy is generated at this endpoint boundary. The full-hook path and anchor path both use the full TransformerLens endpoint; the manual and target paths do not.

The independent target contains the same defective statements:

```python
final = normalized_final[rows, positions, :]
W_selected = model.W_U.index_select(1, suffix_token_ids)
logits = final @ W_selected
if getattr(model, "b_U", None) is not None:
    logits = logits + model.b_U.index_select(0, suffix_token_ids)
return logits
```

after otherwise executing the actual block-10 MLP, residual-bypass subtraction, block 11, and final LayerNorm. ([GitHub][8])

## 4.3 Why the discrepancy has the observed scale

The values:

[
6.103515625\times10^{-5}
\quad\text{and}\quad
7.62939453125\times10^{-5}
]

lie exactly on binary float increments. They are consistent with a small number of float32 last-place differences in raw GPT-2 logits produced by two GEMM realizations of the same real-valued dot products.

The runtime was frozen to float32 with:

* `torch.set_float32_matmul_precision("highest")`;
* TF32 disabled;
* deterministic algorithms enabled;
* the pinned TransformerLens model in float32. ([GitHub][7])

No change to numerical precision is required or authorized.

## 4.4 Root-cause proof condition

Although the source-level defect is unambiguous, the fresh prepare must make the attribution empirically complete. Before any scientific response, it must demonstrate:

1. bitwise equality of the manual and full-hook tensors at every stage through final LayerNorm;
2. reproduction of the archived discrepancy by an isolated, diagnostic-only copy of the legacy selected-column expression;
3. removal of that discrepancy when the identical full TransformerLens unembedding endpoint is used;
4. no other first-diverging upstream tensor.

Failure of any of these assertions terminates v1.3.1 at `06B_MANUAL_TAIL_STAGE_TRACE`. It does not authorize another correction or run.

---

# 5. Classification of alternative explanations

| Candidate explanation                       | Binding finding                                                     | Reason                                                                                                                                   |
| ------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Genuine derivative mismatch                 | **Not established**                                                 | The failed “derivative” check computes no finite difference, Jacobian-vector product, or derivative.                                     |
| Ill-conditioned relative metric near zero   | **Yes**                                                             | The denominator is the norm of an endpoint response and is replaced by (10^{-5}) when the response is zero or tiny.                      |
| Incorrect denominator or comparison target  | **Yes, for a derivative audit**                                     | It compares endpoint deviations from the anchor, not derivatives.                                                                        |
| Full-hook/manual-tail anchoring mismatch    | **No evidence; ruled out as the reported ratio’s numerator source** | The common anchor cancels exactly from the numerator; same-TransformerLens no-op error was (0.0).                                        |
| TransformerLens arithmetic-order difference | **Yes**                                                             | The manual path rearranges the final linear operation instead of invoking the pinned unembedding endpoint.                               |
| Dtype or casting mismatch                   | **No**                                                              | Both paths execute in frozen float32; conversion to float64 occurs only after outputs are produced for measurement.                      |
| Omitted unembedding bias                    | **No**                                                              | The manual code adds the selected `b_U`; Gate-04 also audits the mapped bias.                                                            |
| Omitted block-10 or block-11 operation      | **No code evidence**                                                | The manual path calls the actual `ln2`, activation, two pinned `batch_addmm` operations, block 11, and final LayerNorm.                  |
| Omitted output softcap                      | **Yes, as a separate endpoint-completeness defect**                 | Full TransformerLens applies it; manual and target code do not. Its numerical contribution must be reported explicitly in the new trace. |
| Wrong gates, sites, frame, or radius        | **No**                                                              | These passed their independent structural and hook preflights.                                                                           |
| Theory failure                              | **No**                                                              | The failure is upstream of any development or confirmation scientific response.                                                          |

---

# 6. Analysis of the reported derivative-relative values

## 6.1 Algebraic cancellation

The old metric is:

[
R_i
===

\frac{
\left|
(M_i-A)-(F_i-A)
\right|_2
}{
\max(|F_i-A|_2,10^{-5})
},
]

where (A) is `anchor.year_logits`.

The numerator simplifies exactly:

[
(M_i-A)-(F_i-A)=M_i-F_i.
]

Therefore:

[
\boxed{
R_i
===

\frac{|M_i-F_i|_2}
{\max(|F_i-A|_2,10^{-5})}
}
]

It is the same raw implementation discrepancy normalized by the size of the full endpoint effect. It is not a derivative error.

## 6.2 Center condition

At (C_0=(x=0,z=0)), the full-hook computation is the same no-op TransformerLens endpoint as the anchor. The server recorded same-TransformerLens no-op maximum error (0.0). ([GitHub][1])

Thus:

[
F_0-A=0,
]

and:

[
R_0
===

# \frac{|M_0-F_0|_2}{10^{-5}}

25.89502372509329.
]

The corresponding numerator is:

[
|M_0-F_0|_2
===========

# 25.89502372509329\times10^{-5}

2.589502372509329\times10^{-4}.
]

This is fully compatible with 100 coordinates each differing by at most approximately (7.63\times10^{-5}). It is not evidence of an enormous derivative disagreement. It is a nonzero raw endpoint discrepancy divided by an arbitrary conditioning floor.

The center condition remains binding. It must not be deleted, hidden, converted to a centered-logit check, or reclassified as irrelevant.

## 6.3 Near-zero noncenter condition

The second value,

[
1.3051775251256144,
]

arises where the full endpoint response (|F_i-A|_2) is small enough that the same (O(10^{-4})) vector discrepancy is comparable to or larger than the true response.

This is a genuine failure of the old relative endpoint-effect metric as a stable derivative audit. It is not harmless: the underlying raw endpoint discrepancy also independently exceeds the frozen (2\times10^{-5}) maximum-absolute threshold.

## 6.4 Larger-response conditions

The remaining values,

[
3.8560\times10^{-4},\quad
3.0074\times10^{-4},\quad
3.2303\times10^{-4},
]

are smaller because their denominators are larger. They still exceed (10^{-4}). This is again expected when a roughly fixed implementation-level raw error is divided by endpoint effects of different magnitudes.

No archived derivative claim may be made from these values.

---

# 7. Correct mathematical derivative audit

## 7.1 Required quantity

For a frozen coordinate direction (v) and frozen radius (h>0), define the central finite differences:

[
D_M(v;h)
========

\frac{M(+hv)-M(-hv)}{2h},
]

[
D_F(v;h)
========

\frac{F(+hv)-F(-hv)}{2h}.
]

Define:

[
E_D
===

|D_M-D_F|_2,
\qquad
N_D
===

|D_F|_2.
]

Only this or an exact JVP comparison constitutes a derivative-equivalence audit. The fresh preflight shall use the central finite differences above because both implementations are already endpoint functions and the required signed endpoints are easy to audit in raw-logit space.

## 7.2 Frozen stencils

The active derivative-equivalence panel shall use the same first gate and the already-frozen radii:

1. `path_dx_e0_at_z0`:
   [
   x=\pm h_xe_0,\quad z=0;
   ]
2. `path_dx_e1_at_z0`:
   [
   x=\pm h_xe_1,\quad z=0;
   ]
3. `path_dx_e2_at_z0`:
   [
   x=\pm h_xe_2,\quad z=0;
   ]
4. `path_dz_at_x0`:
   [
   x=0,\quad z=\pm h_z;
   ]
5. `control_dx_e2_at_z0`:
   [
   x=\pm h_xe_2,\quad z=0;
   ]
6. `control_dz_at_x0`:
   [
   x=0,\quad z=\pm h_z.
   ]

Here:

[
h_z=0.20
]

and (h_x) is the unmodified structural radius produced from the frozen preflight anchors.

The original five raw conditions remain present and binding. The signed derivative stencils are additional technical-equivalence assertions, not new scientific intervention conditions.

## 7.3 Non-near-zero branch

Freeze:

```python
TAIL_DERIVATIVE_REFERENCE_NORM_FLOOR = 1.0e-5
```

This constant is a conditioning classifier, not a relaxed pass threshold.

When:

[
N_D>10^{-5},
]

require:

[
\boxed{
\frac{E_D}{N_D}\le10^{-4}
}
]

using the existing frozen derivative-relative threshold unchanged.

## 7.4 Near-zero branch

When:

[
N_D\le10^{-5},
]

the relative ratio is classified as:

```text
NOT_APPLICABLE_NEAR_ZERO
```

It must not be assigned zero, silently omitted, or counted as an automatic pass.

Let:

[
\epsilon=2\times10^{-5}
]

be the unchanged raw endpoint maximum-absolute threshold, and let (m=100) be the output dimension.

Every signed endpoint must first independently satisfy:

[
|M(+hv)-F(+hv)|_\infty\le\epsilon,
]

[
|M(-hv)-F(-hv)|_\infty\le\epsilon.
]

Then, coordinate-wise:

[
\left|
[D_M-D_F]_j
\right|
=

\left|
\frac{
(M_+-F_+)*j-(M*--F_-)_j
}{2h}
\right|
\le
\frac{\epsilon}{h}.
]

Therefore the proof-derived bounds are:

[
\boxed{
|D_M-D_F|_\infty
\le
\frac{\epsilon}{h}
}
]

and

[
\boxed{
|D_M-D_F|_2
\le
\sqrt{100}\frac{\epsilon}{h}
============================

10\frac{\epsilon}{h}.
}
]

Both bounds shall be asserted. They introduce no discretionary tolerance and no threshold relaxation; they are direct consequences of the unchanged raw endpoint threshold.

---

# 8. Quantity that must be compared

## 8.1 Primary binding quantity

The manual-tail and path-target executable-equivalence gates must compare:

[
\boxed{\text{raw 100-dimensional year logits}}
]

coordinate by coordinate.

The binding raw condition remains:

[
\boxed{
\max |Y_M-Y_F|
\le
2\times10^{-5}.
}
]

## 8.2 Why raw logits are required

Raw logits are the strongest endpoint-equivalence object. They detect:

* common offsets;
* coordinate-dependent offsets;
* wrong biases;
* wrong unembedding behavior;
* wrong softcapping;
* incorrect vocabulary selection;
* hidden changes that a margin or contrast could cancel.

Centered logits, logit deltas, and margins can remove common components and therefore conceal an endpoint implementation defect. They may be serialized as diagnostics only after the raw gate passes.

## 8.3 Diagnostic-only quantities

The following may be reported but cannot substitute for the raw gate:

* centered year logits;
* logit deltas from the anchor;
* task margins;
* contrast projections;
* RMS error;
* cosine similarity.

No centered-only, delta-only, or margin-only equivalence audit is authorized.

---

# 9. Binding classification of the v1.3 STOP

The requested alternatives are resolved as follows.

## 9.1 Existing v1.3 identity

For:

```text
schema_version = green-bridge-v1.3
protocol_run_id = green-bridge-v1.3-one-shot
attempt_index = 1
```

the `STOP` is terminal and valid.

It establishes the narrow factual result:

> The one authorized v1.3 prepare execution failed its frozen manual-tail executable-equivalence gate before any development or confirmation response.

It does not establish a scientific failure of the structural-envelope matched-bypass hypothesis.

## 9.2 Correction classification

The defect is a technical implementation/equivalence-audit defect because:

1. it lies in the manual endpoint implementation;
2. it occurs before development;
3. it occurs before confirmation;
4. no scientific score was observed;
5. the correction leaves the mathematical endpoint unchanged and instead makes the executable endpoint faithful to the frozen model;
6. no target, gate, site, radius, frame, estimator, dataset, or statistical rule needs to change;
7. no precision change or threshold relaxation is needed.

Therefore this decision selects:

[
\boxed{\text{Option 2}}
]

and authorizes a separately versioned one-shot attempt.

---

# 10. Theory and scientific design that remain frozen

The following are immutable under v1.3.1:

* basis-free ambient rank-one matched-bypass path operator;
* exact LayerNorm structural envelope;
* common-frame dimension (4);
* per-gate frame dimension (5);
* all-gate frame dimension (14);
* all ten selected MLP-10 gate indices;
* block-8 patch site;
* block-10 `resid_mid` (x)-intervention site;
* block-10 MLP preactivation (z)-intervention site;
* postactivation anchoring semantics;
* matched control;
* direct residual-bypass preservation in path/control;
* independent joint-gate target;
* residual-bypass subtraction in the target;
* evaluation population;
* deterministic first-order directions;
* all radii;
* finite-difference estimators used for scientific scores;
* numerical inverse;
* no pseudoinverse or ridge substitution;
* development/confirmation split;
* development thresholds;
* confirmation lock;
* confirmatory thresholds;
* bootstrap rules;
* forward-count accounting;
* ICLR Oral-level central claim.

The v1.2 decision explicitly states that fixed-rank donor PCA was the wrong scientific object and that the replacement is the basis-invariant ambient operator with exact architecture-derived probe completeness. That upgrade remains the main line. ([GitHub][3])

Fixed-rank donor PCA, rank search, eigengap search, donor replacement, spectral filtering, and learned alignment remain forbidden.

---

# 11. Exact source-code changes

## 11.1 `src/green_bridge_tail.py`

### 11.1.1 Add the exact endpoint helper

Add, immediately after `gather_year_logits`:

```python
def full_transformerlens_year_logits(
    model,
    normalized_final,
    final_positions,
    suffix_token_ids,
):
    """Apply the exact pinned TransformerLens output endpoint, then gather.

    The full sequence and full vocabulary must be passed through model.unembed
    before either the sequence position or year-token coordinates are selected.
    """
    try:
        from transformer_lens.utilities import apply_softcap
    except ImportError as exc:  # pragma: no cover - server environment
        raise RuntimeError("pinned TransformerLens is required") from exc

    full_logits = model.unembed(normalized_final)
    full_logits = apply_softcap(
        full_logits,
        model.cfg.output_logits_soft_cap,
    )
    return gather_year_logits(
        model,
        full_logits,
        final_positions,
        suffix_token_ids,
    )
```

The implementation must use `model.unembed`. It must not reimplement `F.linear`, because the model module and its frozen source are the executable reference.

### 11.1.2 Replace the defective endpoint

In `GreenBridgeTail.evaluate_physical`, delete:

```python
final = normalized_final[rows, positions, :]
W_selected = self.model.W_U.index_select(1, self.suffix_ids)
logits = final @ W_selected
if getattr(self.model, "b_U", None) is not None:
    logits = logits + self.model.b_U.index_select(0, self.suffix_ids)
return logits
```

Replace it exactly with:

```python
return full_transformerlens_year_logits(
    self.model,
    normalized_final,
    positions,
    self.suffix_ids,
)
```

No position slicing and no vocabulary slicing may occur before `model.unembed`.

### 11.1.3 Add a shared traceable core

Refactor without changing normal return values:

```python
def _evaluate_physical_core(..., return_trace: bool):
    ...
```

`evaluate_physical` must call:

```python
logits, _ = self._evaluate_physical_core(..., return_trace=False)
return logits
```

Add:

```python
def evaluate_physical_with_trace(...):
    return self._evaluate_physical_core(..., return_trace=True)
```

For prepare-only single-record audits, the trace must expose detached tensors under exactly these keys:

```text
resid_mid_after_x
ln2_output
pre_after_z
live_post
anchored_post
mlp_out
resid_post_before_subtraction
resid_post_after_subtraction
block11_resid_post
ln_final_output
unembed_pre_softcap_full
unembed_post_softcap_full
year_logits
```

The ordinary scientific path must not serialize or retain these tensors.

## 11.2 `src/green_bridge_path_target.py`

The target module must remain code-isolated. It must not import:

* `green_bridge_tail`;
* predictor code;
* `matched_bypass_gate`;
* baseline code.

### 11.2.1 Add a local full-endpoint helper

Add a local helper with the same operation order:

```python
def _full_transformerlens_year_logits(
    model,
    normalized_final,
    final_positions,
    suffix_token_ids,
):
    try:
        from transformer_lens.utilities import apply_softcap
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("path target requires pinned TransformerLens") from exc

    full_logits = model.unembed(normalized_final)
    full_logits = apply_softcap(
        full_logits,
        model.cfg.output_logits_soft_cap,
    )

    torch = _torch()
    rows = torch.arange(
        full_logits.shape[0],
        device=full_logits.device,
    )
    final = full_logits[rows, final_positions]
    return final.index_select(-1, suffix_token_ids)
```

The helper is intentionally duplicated to preserve target independence.

### 11.2.2 Replace the target endpoint

Delete the final-position/selected-column `@` expression and replace it with:

```python
return _full_transformerlens_year_logits(
    model,
    normalized_final,
    positions,
    suffix_token_ids,
)
```

No active target code may contain:

```python
model.W_U.index_select(...)
final @ W_selected
```

## 11.3 `src/exp_green_bridge_gpt2.py`

### 11.3.1 Reviewed ancestor

Set:

```python
REVIEW_COMMIT = "b8b5f7b509abda310f3e2414e0552a644a12b4bf"
```

The new execution commit must be a clean descendant of this commit.

### 11.3.2 Source and protocol files

`SOURCE_FILES` must include both launchers:

```text
src/launch_green_bridge.sh
src/launch_green_bridge_v131.sh
```

The v1.3 launcher remains historical and must not be modified.

Add to `PROTOCOL_FILES`:

```text
analysis/GREEN_SERVER_V13_PREPARE_STOP_20260825.md
analysis/GPTPRO_GREEN_V13_MANUAL_TAIL_DECISION_20260825.md
analysis/archive/green_v13_stop_20260825/archive_manifest.json
analysis/archive/green_v13_stop_20260825/green_bridge_v13_prepare.log
```

### 11.3.3 Predecessor verification

Add:

```python
def verify_v13_terminal_archive() -> dict:
    ...
```

It must verify every frozen hash in Section 17 before creating or writing the v1.3.1 output root.

It must also assert:

```text
v1.3 result verdict == STOP
v1.3 first_failed_gate == 06_MANUAL_TAIL
v1.3 attempt_index == 1
v1.3 retry_allowed == false
v1.3 development_started == false
v1.3 confirmation_started == false
v1.3 development artifacts absent
v1.3 confirmation artifacts absent
```

Failure is:

```text
00A_PREDECESSOR_ARCHIVE
```

### 11.3.4 Physical full-hook reference

Refactor:

```python
def full_hook_endpoint(...)
```

so that coordinate conversion is separate from the hook implementation.

Add:

```python
def full_hook_endpoint_physical(
    model,
    tokens,
    suffix_ids,
    anchor,
    residual_delta,
    z,
    *,
    mode,
    gate_slot=None,
    block8_patch=None,
    subtract_residual_bypass=False,
    return_trace=False,
):
    ...
```

The existing `full_hook_endpoint` must compute:

```python
residual_delta = x @ U.T
```

and delegate to `full_hook_endpoint_physical`.

This creates an exact independent reference for the physical-vector path target without manufacturing an auxiliary coordinate basis.

### 11.3.5 Full-hook trace

When `return_trace=True`, add read-only hooks or capture points for exactly:

```text
resid_mid_after_x
pre_after_z
post_after_anchor
mlp_out
resid_post_after_subtraction
block11_resid_post
ln_final_output
unembed_pre_softcap_full
unembed_post_softcap_full
year_logits
```

Read-only trace hooks must return their input unchanged.

### 11.3.6 Legacy diagnostic expression

Add a private prepare-only helper:

```python
def _legacy_selected_projection_year_logits(
    model,
    normalized_final,
    final_positions,
    suffix_ids,
):
    torch = torch_module()
    rows = torch.arange(
        normalized_final.shape[0],
        device=normalized_final.device,
    )
    final = normalized_final[rows, final_positions, :]
    selected = model.W_U.index_select(1, suffix_ids)
    logits = final @ selected
    if getattr(model, "b_U", None) is not None:
        logits = logits + model.b_U.index_select(0, suffix_ids)
    return logits
```

Restrictions:

* this helper may be called only from `_tail_preflight_v131`;
* it may not be imported by the manual tail or path target;
* it may not generate scientific outputs;
* it may not be selected as a performance fallback;
* it must be labelled `diagnostic_only` in artifacts.

### 11.3.7 Replace `_tail_preflight_v13`

Add:

```python
def _tail_preflight_v131(...):
    ...
```

The prepare phase must call only `_tail_preflight_v131`.

The new preflight order is binding:

1. reproduce the legacy raw mismatch;
2. compare stagewise manual and full-hook tensors;
3. compare corrected raw logits on the original five conditions;
4. compare raw logits on every signed derivative endpoint;
5. perform true derivative checks;
6. perform batch-shape checks;
7. audit the independent target;
8. run prepare-only memory and throughput checks;
9. serialize all audit artifacts;
10. only then mark prepare complete.

### 11.3.8 Legacy reproduction assertion

On the same deterministically selected record and in the same condition order, the diagnostic legacy expression must reproduce:

```python
[
    7.62939453125e-05,
    6.103515625e-05,
    6.103515625e-05,
    7.62939453125e-05,
    6.103515625e-05,
]
```

The values must match exactly as Python floats under the frozen server environment.

Failure is:

```text
06A_LEGACY_ROOT_CAUSE_REPRODUCTION
```

No scientific response may follow.

### 11.3.9 Stagewise assertion

For the original five conditions, compare the manual and full-hook traces.

Require bitwise equality for:

```text
resid_mid_after_x
ln2_output
pre_after_z
anchored_post
mlp_out
resid_post_after_subtraction
block11_resid_post
ln_final_output
```

Require bitwise equality for the corrected pre-softcap and post-softcap full-vocabulary tensors when both paths use `model.unembed` on tensors of the same shape.

If any tensor first diverges before unembedding, stop at:

```text
06B_MANUAL_TAIL_STAGE_TRACE
```

The artifact must record the first divergent stage and every preceding equality result.

### 11.3.10 Corrected raw gate

For every original and signed condition require:

```python
max_abs(manual_raw_year_logits - full_raw_year_logits) <= 2.0e-5
```

Failure is:

```text
06C_MANUAL_TAIL_RAW
```

### 11.3.11 True derivative gate

Implement:

```python
def derivative_equivalence_record(
    manual_plus,
    manual_minus,
    full_plus,
    full_minus,
    *,
    step,
):
    manual_derivative = (
        manual_plus.double() - manual_minus.double()
    ) / (2.0 * step)

    full_derivative = (
        full_plus.double() - full_minus.double()
    ) / (2.0 * step)

    difference = manual_derivative - full_derivative
    absolute_l2 = float(
        torch.linalg.vector_norm(difference).item()
    )
    absolute_max = float(
        difference.abs().max().item()
    )
    reference_l2 = float(
        torch.linalg.vector_norm(full_derivative).item()
    )

    if reference_l2 > TAIL_DERIVATIVE_REFERENCE_NORM_FLOOR:
        relative = absolute_l2 / reference_l2
        passed = (
            relative
            <= THRESHOLDS.tail_derivative_relative
        )
        status = "RELATIVE_APPLICABLE"
    else:
        max_bound = (
            THRESHOLDS.tail_max_abs / step
        )
        l2_bound = (
            DIMENSIONS.output_dimension ** 0.5
            * THRESHOLDS.tail_max_abs
            / step
        )
        relative = None
        passed = (
            absolute_max <= max_bound
            and absolute_l2 <= l2_bound
        )
        status = "NOT_APPLICABLE_NEAR_ZERO"

    return {...}
```

Every raw signed endpoint must already have passed before this function is called.

Failure is:

```text
06D_MANUAL_TAIL_DERIVATIVE
```

### 11.3.12 Batch-shape equivalence

The actual tail and target batch sizes must be frozen during prepare.

For physical RTX 4090, test candidate manual-tail batch sizes in this exact order:

```python
[512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
```

Test candidate full-model and JVP batch sizes in this exact order:

```python
[64, 32, 16, 8, 4, 2, 1]
```

The first candidate satisfying the memory contract is selected.

This is an explicitly authorized hardware-only choice. It does not alter examples, intervention coordinates, precision, estimators, or statistics.

For every selected batch size, require:

1. batched manual output versus same-shape batched full-hook output:
   [
   \max|\Delta|\le2\times10^{-5};
   ]
2. batched manual output versus concatenated batch-one manual outputs:
   [
   \max|\Delta|\le2\times10^{-5};
   ]
3. identical condition order and repeated anchor values;
4. peak allocated memory no greater than 20 GB.

Write the choice before development to:

```text
hardware_batch_plan.json
```

No batch-size change is allowed after the first development endpoint.

Failure is:

```text
06E_BATCH_SHAPE_EQUIVALENCE
```

### 11.3.13 Independent target audit

Using `full_hook_endpoint_physical` with:

```text
mode = joint
subtract_residual_bypass = true
```

compare `evaluate_joint_target` against the full-hook reference at:

```text
physical_delta = 0
physical_delta = +v
physical_delta = -v
physical_delta = +0.5 v
physical_delta = -0.5 v
```

where (v) is the frozen preflight target physical vector.

For every raw 100-dimensional logit vector require:

[
\max|\Delta|\le2\times10^{-5}.
]

Failure is:

```text
06F_PATH_TARGET_RAW
```

### 11.3.14 Prepare throughput

Because full unembedding is now executed faithfully, add a prepare-only technical throughput projection using the preflight record and the frozen operation mixture.

Requirements:

* no development or confirmation records may be loaded;
* operation counts remain unchanged;
* peak memory remains at most 20 GB;
* extrapolated RTX 4090 runtime remains at most 24 GPU hours;
* the full-vocabulary endpoint may not be replaced by selected projection for speed.

Failure is:

```text
06G_PREPARE_THROUGHPUT
```

The original protocol’s 24-hour hard cap remains binding. ([GitHub][3])

### 11.3.15 Prepare artifacts

Write all files listed in Section 18 before `prepare_complete=true`.

At the end of prepare:

1. write `prepare_result.json`;
2. write `manifest.json`;
3. call `finalize_hashes(output_root)`;
4. fsync the files and parent directories;
5. exit successfully.

### 11.3.16 Phase schemas

Update every active schema literal from v1.3 to v1.3.1:

```text
green-bridge-manifest-v1.3.1
green-bridge-terminal-v1.3.1
green-bridge-prepare-v1.3.1
```

Historical inactive v1.2 code and archived v1.3 files must not be rewritten.

## 11.4 `src/green_bridge_spec.py`

Set:

```python
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "green_bridge_v131"

SCHEMA_VERSION = "green-bridge-v1.3.1"
PROTOCOL_ID = "structural-envelope-matched-bypass-v1.3.1"
PARENT_PROTOCOL_ID = "structural-envelope-matched-bypass-v1"

AMENDMENT_ID = (
    "GPTPRO-GREEN-V13-MANUAL-TAIL-EQUIVALENCE-v1-20260825"
)

PROTOCOL_RUN_ID = "green-bridge-v1.3.1-one-shot"

TAIL_DERIVATIVE_REFERENCE_NORM_FLOOR = 1.0e-5
TAIL_EQUIVALENCE_OUTPUT_DIM = 100
```

Add:

```python
PREDECESSOR_RUN = {
    "schema_version": "green-bridge-v1.3",
    "protocol_id": "structural-envelope-matched-bypass-v1",
    "protocol_run_id": "green-bridge-v1.3-one-shot",
    "attempt_index": 1,
    "retry_allowed": False,
    "execution_commit":
        "ed4b3b4c55ba2c7acfda1291b4814957ce90c845",
    "first_failed_gate": "06_MANUAL_TAIL",
    "result_sha256":
        "6f61c77b262eee821970dc19ff98f3baaf78e0aa9a65135bed343ed54ac7445d",
}
```

The following values remain unchanged:

```python
MODEL_ID
MODEL_REVISION
TRANSFORMER_LENS_COMMIT
SELECTED_GATES
PROBE_FRAME_DIM
COMMON_FRAME_DIM
ALL_GATE_FRAME_DIM
FIRST_ORDER_RESIDUAL_DIRECTIONS
RESIDUAL_RADIUS_MULTIPLIER
GATE_RADIUS
HALF_RADIUS_MULTIPLIER
all scientific thresholds
all bootstrap settings
all development rules
all confirmation rules
all forward counts
```

The values:

```python
THRESHOLDS.tail_max_abs == 2e-5
THRESHOLDS.tail_derivative_relative == 1e-4
```

must not change.

Add the pinned TransformerLens file:

```text
components/unembed.py
```

to the source-hash contract. Its expected SHA-256 must be computed from the exact pinned commit `4a4dc26c750475b29e6f54b362c2aab988702c9c`, checked into the implementation, and verified by both launcher and runner.

## 11.5 `src/launch_green_bridge_v131.sh`

Create a new launcher by copying the frozen v1.3 launcher. Do not modify `src/launch_green_bridge.sh`.

The only authorized launcher differences are:

```text
comment/version:
  v1.3.1

test log:
  /tmp/green_bridge_v131_contract_${PHASE}.log

execution log:
  /tmp/green_bridge_v131_${PHASE}.log

output root:
  outputs/green_bridge_v131

source hash panel:
  add transformer_lens/components/unembed.py
```

Preserve exactly:

* conda environment name;
* Python version;
* package versions;
* PyTorch CUDA wheel;
* pinned TransformerLens commit;
* GPU exposure;
* `PYTHONHASHSEED`;
* `CUBLAS_WORKSPACE_CONFIG`;
* all one-thread environment variables;
* float32 execution;
* phase set limited to `prepare`, `development`, and `confirmation`.

The existing launcher confirms the frozen environment variables and invokes the CPU contract before the phase runner. ([GitHub][9])

---

# 12. Required CPU contract tests

All 124 existing tests and assertions must be retained. The following existing tests are revised only where required by the new identity:

1. `HistoricalAndTerminationTests.test_prior_stop_reports_are_protocol_hashed`
2. `SerializationAndOneRunTests.test_attempt_index_is_one`
3. `SerializationAndOneRunTests.test_retry_allowed_is_false`
4. `SerializationAndOneRunTests.test_phase_all_is_rejected`
5. `SerializationAndOneRunTests.test_stopped_v13_root_cannot_resume`
6. `FrozenCoreTests.test_schema_and_protocol`

The existing test suite currently freezes the no-PCA contract, one-run semantics, phase firewall, schema, review ancestor, and STOP non-resumability. ([GitHub][10])

Add exactly these 26 tests:

1. `ManualTailEndpointContractTests.test_manual_tail_uses_full_transformerlens_unembed`
2. `ManualTailEndpointContractTests.test_manual_tail_applies_output_softcap_before_gather`
3. `ManualTailEndpointContractTests.test_manual_tail_does_not_index_wu_before_unembed`
4. `ManualTailEndpointContractTests.test_manual_tail_does_not_slice_final_position_before_unembed`
5. `PathTargetEndpointContractTests.test_path_target_uses_full_transformerlens_unembed`
6. `PathTargetEndpointContractTests.test_path_target_applies_output_softcap_before_gather`
7. `PathTargetEndpointContractTests.test_path_target_does_not_index_wu_before_unembed`
8. `PathTargetEndpointContractTests.test_path_target_remains_code_isolated`
9. `FullHookReferenceContractTests.test_full_hook_endpoint_remains_independent_reference`
10. `TailAuditMetricContractTests.test_tail_raw_gate_compares_raw_year_logits`
11. `TailAuditMetricContractTests.test_tail_raw_gate_threshold_is_two_e_minus_five`
12. `TailAuditMetricContractTests.test_tail_center_condition_is_binding`
13. `TailAuditMetricContractTests.test_tail_derivative_gate_uses_central_difference`
14. `TailAuditMetricContractTests.test_tail_nonzero_derivative_relative_threshold_is_one_e_minus_four`
15. `TailAuditMetricContractTests.test_tail_near_zero_derivative_uses_propagated_absolute_bound`
16. `TailAuditMetricContractTests.test_tail_near_zero_derivative_is_not_silently_dropped`
17. `ProtocolIdentityV131Tests.test_v131_identity_is_not_v13_attempt_two`
18. `ProtocolIdentityV131Tests.test_v131_output_root_is_distinct_from_v13_root`
19. `ProtocolIdentityV131Tests.test_v131_attempt_index_is_one`
20. `ProtocolIdentityV131Tests.test_v131_retry_is_false`
21. `PredecessorArchiveContractTests.test_v13_stop_hashes_are_frozen_and_verified`
22. `PredecessorArchiveContractTests.test_v13_external_log_hash_is_frozen`
23. `PrepareArtifactContractTests.test_root_cause_reproduction_written_before_equivalence_pass`
24. `PrepareArtifactContractTests.test_stage_trace_written_before_equivalence_pass`
25. `PrepareArtifactContractTests.test_path_target_equivalence_written_before_manifest`
26. `TheoryPreservationContractTests.test_fixed_rank_donor_pca_remains_terminated`

After these additions:

```text
Ran 150 tests
OK
```

is required on both the implementation host and the server immediately before every phase.

No test deletion, skip, expected failure, or threshold mocking is authorized.

---

# 13. Required GPU prepare assertions

Before `prepare_complete=true`, every assertion below must pass.

## 13.1 Repository and predecessor

```text
branch == main
reviewed commit b8b5f7... is an ancestor
worktree clean
new output root absent
old v1.3 root present
old v1.3 frozen hashes exact
old v1.3 external log hash exact
```

## 13.2 Attempt identity

```text
schema_version == green-bridge-v1.3.1
protocol_id == structural-envelope-matched-bypass-v1.3.1
parent_protocol_id == structural-envelope-matched-bypass-v1
protocol_run_id == green-bridge-v1.3.1-one-shot
attempt_index == 1
retry_allowed == false
prepare_restart_allowed == false
development_restart_allowed == false
confirmation_restart_allowed == false
phase_all_allowed == false
```

## 13.3 Environment

```text
physical GPU == RTX 4090 GPU 4
CUDA_VISIBLE_DEVICES == 4
process-visible device == cuda:0
dtype == float32
TF32 disabled
float32 matmul precision == highest
deterministic algorithms enabled
CUBLAS_WORKSPACE_CONFIG == :4096:8
all frozen package versions exact
all frozen TransformerLens source hashes exact
```

## 13.4 Gate-04

```text
ordered prompt SHA-256 exact
HF attention backend == eager
batch size == 1
mapped weight mismatch count == 0
all Gate-04 thresholds pass
HF/TL error enters epsilon_y == false
same-TransformerLens no-op max <= 2e-5
```

The previously observed no-op value was `0.0`; the threshold remains (2\times10^{-5}). ([GitHub][1])

## 13.5 Structural frame

Require the original unchanged assertions:

```text
common frame dimension == 4
gate frame dimension == 5
all-gate frame dimension == 14
max orthogonality error <= 5e-13
max raw-atom residual <= 1e-12
max analytic-gradient envelope residual <= 1e-10
max formula/autograd absolute error <= 1e-10
max formula/autograd relative error <= 1e-9
max shift-null metric <= 1e-12
repeated frames bitwise equal == true
```

## 13.6 Root-cause reproduction

```text
legacy diagnostic errors exactly equal archived five-value vector
legacy diagnostic max_abs > 2e-5
legacy diagnostic marked non-scientific
```

## 13.7 Stagewise trace

For each original condition:

```text
all pre-unembed stages bitwise equal
corrected full unembed operation graph identical
corrected post-softcap full logits bitwise equal
corrected gathered year logits max_abs <= 2e-5
first divergent legacy stage == unembedding endpoint
```

## 13.8 Raw conditions

```text
all original five raw conditions <= 2e-5
all signed derivative endpoints <= 2e-5
center condition retained and passed
```

## 13.9 Derivatives

```text
true central finite differences used
relative branch only when reference norm > 1e-5
relative threshold unchanged at 1e-4
near-zero branch explicitly labelled
near-zero max-norm propagated bound passed
near-zero L2 propagated bound passed
no stencil omitted
```

## 13.10 Batch shape

```text
hardware batch plan durably written
selected batch fits <= 20 GB
same-shape manual/full raw equivalence passes
batched/concatenated-batch-one equivalence passes
batch plan frozen before development
```

## 13.11 Independent target

```text
zero and ±1, ±0.5 frozen target-vector endpoints pass raw <= 2e-5
joint ten-gate semantics preserved
residual bypass subtraction preserved
target implementation remains code-isolated
```

## 13.12 Throughput

```text
prepare-only extrapolated RTX 4090 runtime <= 24 GPU hours
scientific call counts unchanged
no selected-projection performance fallback
```

## 13.13 Phase firewall

After prepare, require absence of:

```text
development_anchor_cache.pt
development_structural_inputs.npz
development_frames.npz
development_radii.json
development_target_vectors.npz
noise_audit_dev.json
dev_tensor_scores.parquet
dev_energy_targets.parquet
dev_cells.json
dev_result.json
frozen_analysis.json
confirmation_anchor_cache.pt
confirm_tensor_scores.parquet
confirm_energy_targets.parquet
confirm_cells.json
```

---

# 14. New protocol identity

The fresh run is not “attempt 2” under v1.3.

| Field                 | Required value                                         |
| --------------------- | ------------------------------------------------------ |
| `SCHEMA_VERSION`      | `green-bridge-v1.3.1`                                  |
| manifest schema       | `green-bridge-manifest-v1.3.1`                         |
| terminal schema       | `green-bridge-terminal-v1.3.1`                         |
| prepare-result schema | `green-bridge-prepare-v1.3.1`                          |
| `PROTOCOL_ID`         | `structural-envelope-matched-bypass-v1.3.1`            |
| `PARENT_PROTOCOL_ID`  | `structural-envelope-matched-bypass-v1`                |
| `AMENDMENT_ID`        | `GPTPRO-GREEN-V13-MANUAL-TAIL-EQUIVALENCE-v1-20260825` |
| `PROTOCOL_RUN_ID`     | `green-bridge-v1.3.1-one-shot`                         |
| attempt index         | `1`                                                    |
| retry allowed         | `false`                                                |
| prepare restart       | `false`                                                |
| development restart   | `false`                                                |
| confirmation restart  | `false`                                                |
| `phase all`           | forbidden                                              |
| output root           | `outputs/green_bridge_v131`                            |
| predecessor           | immutable v1.3 attempt-1 STOP                          |

A failed v1.3.1 prepare does not authorize v1.3.2. It requires a new binding decision.

---

# 15. Scientific-invariance record

Before launch, create:

```text
analysis/archive/green_v13_stop_20260825/frozen_scientific_spec_v13.json
```

It must contain the canonical v1.3 scientific values for:

* model and revision;
* tokenizer rules;
* evaluation plan;
* selected gates;
* sites;
* structural dimensions;
* frame construction;
* radii;
* first-order direction seed and hash;
* scientific estimator;
* numerical inverse;
* all scientific thresholds;
* bootstrap rules;
* development rules;
* confirmation rules;
* forward counts;
* prohibition of donor PCA.

Prepare must create:

```text
scientific_invariance_v131.json
```

with:

```json
{
  "parent_schema": "green-bridge-v1.3",
  "current_schema": "green-bridge-v1.3.1",
  "scientific_payload_equal": true,
  "parent_scientific_sha256": "...",
  "current_scientific_sha256": "...",
  "allowed_differences": [
    "protocol identity",
    "output root",
    "manual-tail executable endpoint",
    "target executable endpoint",
    "equivalence-audit metric implementation",
    "equivalence-audit artifacts",
    "predecessor archival metadata"
  ]
}
```

The parent and current scientific hashes must be equal.

Any other difference is:

```text
00B_V131_IDENTITY
```

and terminates prepare.

---

# 16. Authorized actions

The executor is authorized to perform only the following:

1. preserve and archive the existing v1.3 terminal evidence;
2. implement the exact code changes in Section 11;
3. add or revise only the tests named in Section 12;
4. add the v1.3.1 launcher;
5. update protocol identity and provenance constants;
6. run all CPU tests;
7. create one clean implementation commit descending from the reviewed commit;
8. launch v1.3.1 prepare exactly once on physical RTX 4090 GPU 4;
9. proceed to development only after a complete prepare pass;
10. proceed to confirmation only after the frozen development rule opens confirmation;
11. run the exact same scientific experiment after the technical endpoint is repaired;
12. lower hardware batch size only through the deterministic prepare-only sequence in Section 11.3.12;
13. record diagnostic centered logits, deltas, and margins in addition to, never instead of, raw logits.

---

# 17. Forbidden actions

The following remain forbidden:

* rerunning `outputs/green_bridge`;
* invoking the v1.3 launcher on the old root;
* deleting the v1.3 root;
* renaming the v1.3 root to reuse its protocol identity;
* overwriting any v1.3 artifact;
* modifying the old external log;
* treating v1.3.1 as attempt 2 under v1.3;
* relaxing (2\times10^{-5});
* relaxing (10^{-4});
* changing precision;
* enabling TF32;
* changing the CUDA arithmetic contract merely to pass;
* switching to float64, float16, or bfloat16;
* using a selected-column projection as the active endpoint;
* changing the target;
* changing gate indices;
* changing intervention sites;
* changing radii;
* changing frame construction;
* changing frame dimensions;
* changing structural atoms;
* changing the estimator;
* changing numerical inverse rules;
* changing development thresholds;
* changing confirmation thresholds;
* changing bootstrap rules;
* changing the evaluation population;
* accessing development outcomes during prepare;
* accessing confirmation outcomes before the confirmation lock opens;
* silently deleting the center;
* converting the raw-logit audit into a centered-logit audit;
* converting it into a delta-only audit;
* converting it into a margin-only audit;
* declaring a near-zero derivative comparison passed without the propagated absolute bound;
* restoring donor PCA;
* running a rank sweep;
* introducing a donor eigengap;
* using a learned alignment;
* adding a pseudoinverse or ridge to rescue a gate;
* retrying after any v1.3.1 phase has been claimed;
* changing batch size after scientific endpoints begin;
* resubmitting a failed endpoint batch;
* accessing a partial development or confirmation result after a crash and then continuing;
* creating another fresh identity without a new binding decision.

---

# 18. Existing v1.3 archival and hash contract

The directory:

```text
outputs/green_bridge
```

must remain in place and byte-for-byte unchanged.

The following hashes must pass before implementation commit, before v1.3.1 prepare, before development, and before confirmation:

```text
outputs/green_bridge/result.json
6f61c77b262eee821970dc19ff98f3baaf78e0aa9a65135bed343ed54ac7445d

outputs/green_bridge/run_ledger.json
a4c21ea2bea3e42de13bd7789a17db849290556147250ba6f284b3aefa51172c

outputs/green_bridge/hook_audit.json
49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99

outputs/green_bridge/structural_frame_preflight.json
e0f65f22d29fb8db891094c407c25234f7f8f9f19738d4edaf3fb2ed5a19a05a

outputs/green_bridge/first_order_coefficients.npy
d9305194f8d026ddde1a1d9084dd74409eae21e25b0b7600ca51f8887ff7b926

outputs/green_bridge/splits.json
0490113fbfe66bcab1fba924896f832fac4668f2566402aa0107ed4fa43ed0ca

outputs/green_bridge/development_splits.json
7fb05a1bf83d0083c622630694df09485dbaf18f4caaf6f5614200e0d8d2baf0

outputs/green_bridge/model_fingerprint.json
fb9bd5a686d1bb09fa31c4cc308ff51f26c1d64075feb57d5a330db8fcaa6cb0

outputs/green_bridge/gate04_legacy_panel.json
646d2ebcf1229645c83ebadea7f39d782e12152a8248dbd122f8c11e58c83df1

/tmp/green_bridge_v13_prepare.log
28c2788da0477b5c95c4498d70a9a4183f2188419dbca6e5f4725b60d5dc8e52
```

These are the frozen server-report values. ([GitHub][1])

Create:

```text
analysis/archive/green_v13_stop_20260825/
```

containing:

```text
archive_manifest.json
green_bridge_v13_prepare.log
frozen_scientific_spec_v13.json
```

The external log must be copied, not moved, and its copied hash must remain:

```text
28c2788da0477b5c95c4498d70a9a4183f2188419dbca6e5f4725b60d5dc8e52
```

`archive_manifest.json` must record:

* all paths and hashes above;
* reviewed commit;
* server execution commit;
* branch;
* clean launch;
* GPU mapping;
* attempt index;
* retry policy;
* first failed gate;
* development/confirmation firewall state;
* copy timestamp;
* statement that no old file was modified.

---

# 19. Required v1.3.1 prepare artifacts

The new prepare must durably produce:

```text
outputs/green_bridge_v131/run_ledger.json
outputs/green_bridge_v131/model_fingerprint.json
outputs/green_bridge_v131/splits.json
outputs/green_bridge_v131/development_splits.json
outputs/green_bridge_v131/gate04_legacy_panel.json
outputs/green_bridge_v131/hook_audit.json
outputs/green_bridge_v131/structural_frame_preflight.json
outputs/green_bridge_v131/first_order_coefficients.npy
outputs/green_bridge_v131/scientific_invariance_v131.json
outputs/green_bridge_v131/manual_tail_root_cause_reproduction_v131.json
outputs/green_bridge_v131/manual_tail_stage_trace_v131.json
outputs/green_bridge_v131/manual_tail_equivalence_v131.json
outputs/green_bridge_v131/manual_tail_derivative_v131.json
outputs/green_bridge_v131/manual_tail_batch_equivalence_v131.json
outputs/green_bridge_v131/path_target_equivalence_v131.json
outputs/green_bridge_v131/hardware_batch_plan.json
outputs/green_bridge_v131/manual_tail_throughput_v131.json
outputs/green_bridge_v131/tail_audit.json
outputs/green_bridge_v131/prepare_result.json
outputs/green_bridge_v131/manifest.json
outputs/green_bridge_v131/sha256sums.txt
```

`tail_audit.json` is an aggregate index. It must not replace the detailed files.

## 19.1 `prepare_result.json`

It must contain:

```json
{
  "schema_version": "green-bridge-prepare-v1.3.1",
  "verdict": "PREPARE_PASS",
  "first_failed_gate": null,
  "development_started": false,
  "confirmation_started": false
}
```

## 19.2 Manifest hashes

`manifest.json` must contain:

```text
source_sha256
protocol_sha256
requirements_sha256
artifact_sha256
frozen_spec_sha256
scientific_spec_sha256
predecessor_archive_sha256
execution_commit
review_commit
```

Every prepare artifact except `manifest.json` and `sha256sums.txt` must appear in `artifact_sha256`.

`sha256sums.txt` must hash every regular file in the root except itself, including `manifest.json`.

New hash values are generated from the actual immutable bytes and therefore cannot be named before execution. They must be computed atomically, serialized, and reverified before each continuation phase.

---

# 20. Exact implementation and archival commands

Run these commands from the server repository.

## 20.1 Establish the reviewed base

```bash
set -euo pipefail

REPO=/home/ccj/workspace_1/idle_1_green_bridge
cd "$REPO"

git switch main

test "$(git rev-parse HEAD)" = \
  "b8b5f7b509abda310f3e2414e0552a644a12b4bf"

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"
```

Do not merge unrelated later commits into the implementation.

## 20.2 Verify the old terminal evidence

```bash
sha256sum -c <<'EOF'
6f61c77b262eee821970dc19ff98f3baaf78e0aa9a65135bed343ed54ac7445d  outputs/green_bridge/result.json
a4c21ea2bea3e42de13bd7789a17db849290556147250ba6f284b3aefa51172c  outputs/green_bridge/run_ledger.json
49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99  outputs/green_bridge/hook_audit.json
e0f65f22d29fb8db891094c407c25234f7f8f9f19738d4edaf3fb2ed5a19a05a  outputs/green_bridge/structural_frame_preflight.json
d9305194f8d026ddde1a1d9084dd74409eae21e25b0b7600ca51f8887ff7b926  outputs/green_bridge/first_order_coefficients.npy
0490113fbfe66bcab1fba924896f832fac4668f2566402aa0107ed4fa43ed0ca  outputs/green_bridge/splits.json
7fb05a1bf83d0083c622630694df09485dbaf18f4caaf6f5614200e0d8d2baf0  outputs/green_bridge/development_splits.json
fb9bd5a686d1bb09fa31c4cc308ff51f26c1d64075feb57d5a330db8fcaa6cb0  outputs/green_bridge/model_fingerprint.json
646d2ebcf1229645c83ebadea7f39d782e12152a8248dbd122f8c11e58c83df1  outputs/green_bridge/gate04_legacy_panel.json
EOF

test "$(
  sha256sum /tmp/green_bridge_v13_prepare.log |
  awk '{print $1}'
)" = \
"28c2788da0477b5c95c4498d70a9a4183f2188419dbca6e5f4725b60d5dc8e52"
```

Verify phase state:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/green_bridge")

result = json.loads(
    (root / "result.json").read_text(encoding="utf-8")
)
ledger = json.loads(
    (root / "run_ledger.json").read_text(encoding="utf-8")
)

assert result["verdict"] == "STOP"
assert result["first_failed_gate"] == "06_MANUAL_TAIL"
assert ledger["attempt_index"] == 1
assert ledger["retry_allowed"] is False
assert ledger["development_started"] is False
assert ledger["confirmation_started"] is False

for name in (
    "frozen_analysis.json",
    "dev_tensor_scores.parquet",
    "dev_energy_targets.parquet",
    "confirm_tensor_scores.parquet",
    "confirm_energy_targets.parquet",
):
    assert not (root / name).exists(), name
PY
```

## 20.3 Create the archive

```bash
ARCH=analysis/archive/green_v13_stop_20260825

test ! -e "$ARCH"
mkdir -p "$ARCH"

cp -p \
  /tmp/green_bridge_v13_prepare.log \
  "$ARCH/green_bridge_v13_prepare.log"

test "$(
  sha256sum "$ARCH/green_bridge_v13_prepare.log" |
  awk '{print $1}'
)" = \
"28c2788da0477b5c95c4498d70a9a4183f2188419dbca6e5f4725b60d5dc8e52"
```

Generate `archive_manifest.json` and `frozen_scientific_spec_v13.json` from the exact values in Sections 15 and 18.

## 20.4 Implement and test

Apply only the source changes in Section 11 and the tests in Section 12.

Then run:

```bash
python src/test_green_bridge_contract.py \
  2>&1 |
  tee /tmp/green_bridge_v131_contract_precommit.log

grep -F "Ran 150 tests" \
  /tmp/green_bridge_v131_contract_precommit.log

grep -F "OK" \
  /tmp/green_bridge_v131_contract_precommit.log
```

Check the patch:

```bash
git diff --check

git diff -- \
  src/green_bridge_spec.py \
  src/green_bridge_tail.py \
  src/green_bridge_path_target.py \
  src/exp_green_bridge_gpt2.py \
  src/test_green_bridge_contract.py \
  src/launch_green_bridge_v131.sh \
  analysis/GPTPRO_GREEN_V13_MANUAL_TAIL_DECISION_20260825.md \
  analysis/archive/green_v13_stop_20260825
```

Confirm that the old launcher is unchanged:

```bash
git diff --exit-code \
  b8b5f7b509abda310f3e2414e0552a644a12b4bf \
  -- src/launch_green_bridge.sh
```

## 20.5 Commit the correction

```bash
git add \
  analysis/GPTPRO_GREEN_V13_MANUAL_TAIL_DECISION_20260825.md \
  analysis/archive/green_v13_stop_20260825 \
  src/green_bridge_spec.py \
  src/green_bridge_tail.py \
  src/green_bridge_path_target.py \
  src/exp_green_bridge_gpt2.py \
  src/test_green_bridge_contract.py \
  src/launch_green_bridge_v131.sh

git diff --cached --check

git commit -m \
  "Correct GREEN v1.3 manual-tail endpoint as v1.3.1"

EXECUTION_COMMIT="$(git rev-parse HEAD)"
printf '%s\n' "$EXECUTION_COMMIT"

git merge-base --is-ancestor \
  b8b5f7b509abda310f3e2414e0552a644a12b4bf \
  "$EXECUTION_COMMIT"

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"

test ! -e outputs/green_bridge_v131
```

The printed commit is the only authorized v1.3.1 execution commit. It must be recorded in the new ledger and manifest.

---

# 21. Exact prepare command

After all Section 20 commands pass:

```bash
set -euo pipefail
cd /home/ccj/workspace_1/idle_1_green_bridge

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"

test ! -e outputs/green_bridge_v131

bash src/launch_green_bridge_v131.sh 4 prepare
```

This command may be issued exactly once.

If it exits nonzero, is interrupted after ledger creation, writes `STOP`, or leaves a partial root, it must not be rerun.

---

# 22. Mechanical prepare-pass verification

Run only after the prepare command exits successfully:

```bash
set -euo pipefail
cd /home/ccj/workspace_1/idle_1_green_bridge

python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("outputs/green_bridge_v131")

manifest = json.loads(
    (root / "manifest.json").read_text(encoding="utf-8")
)
ledger = json.loads(
    (root / "run_ledger.json").read_text(encoding="utf-8")
)
prepare = json.loads(
    (root / "prepare_result.json").read_text(encoding="utf-8")
)

assert manifest["schema_version"] == \
    "green-bridge-manifest-v1.3.1"
assert manifest["run"]["protocol_id"] == \
    "structural-envelope-matched-bypass-v1.3.1"
assert manifest["run"]["parent_protocol_id"] == \
    "structural-envelope-matched-bypass-v1"
assert manifest["run"]["attempt_index"] == 1
assert manifest["run"]["retry_allowed"] is False
assert manifest["prepare_complete"] is True
assert manifest["confirmation_open"] is False

assert ledger["protocol_run_id"] == \
    "green-bridge-v1.3.1-one-shot"
assert ledger["attempt_index"] == 1
assert ledger["retry_allowed"] is False
assert ledger["development_started"] is False
assert ledger["confirmation_started"] is False

assert prepare["verdict"] == "PREPARE_PASS"
assert prepare["first_failed_gate"] is None

assert manifest["binding_decision"][
    "fixed_rank_donor_pca_terminated"
] is True
assert manifest["structural_estimand"][
    "donor_pca_used"
] is False

invariance = json.loads(
    (root / "scientific_invariance_v131.json").read_text(
        encoding="utf-8"
    )
)
assert invariance["scientific_payload_equal"] is True
assert (
    invariance["parent_scientific_sha256"]
    ==
    invariance["current_scientific_sha256"]
)

for name, expected in manifest["artifact_sha256"].items():
    path = root / name
    assert path.is_file(), name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (name, actual, expected)

for forbidden in (
    "development_anchor_cache.pt",
    "development_structural_inputs.npz",
    "development_frames.npz",
    "development_radii.json",
    "development_target_vectors.npz",
    "noise_audit_dev.json",
    "dev_tensor_scores.parquet",
    "dev_energy_targets.parquet",
    "dev_cells.json",
    "dev_result.json",
    "frozen_analysis.json",
    "confirmation_anchor_cache.pt",
    "confirm_tensor_scores.parquet",
    "confirm_energy_targets.parquet",
):
    assert not (root / forbidden).exists(), forbidden

result_path = root / "result.json"
if result_path.exists():
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result.get("verdict") != "STOP"
PY

(
  cd outputs/green_bridge_v131
  sha256sum -c sha256sums.txt
)
```

Reverify all predecessor hashes from Section 20.2 before development.

---

# 23. Exact development command

Development is authorized only after every assertion in Section 22 passes.

```bash
set -euo pipefail
cd /home/ccj/workspace_1/idle_1_green_bridge

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"

bash src/launch_green_bridge_v131.sh 4 development
```

The development command may be issued exactly once.

If development does not produce the frozen `OPEN_CONFIRMATION` verdict, confirmation is forbidden.

No threshold, condition, batch plan, or source file may be changed after development is claimed.

---

# 24. Mechanical development-pass verification

Run only after development exits successfully:

```bash
set -euo pipefail
cd /home/ccj/workspace_1/idle_1_green_bridge

python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("outputs/green_bridge_v131")

manifest = json.loads(
    (root / "manifest.json").read_text(encoding="utf-8")
)
ledger = json.loads(
    (root / "run_ledger.json").read_text(encoding="utf-8")
)
development = json.loads(
    (root / "dev_result.json").read_text(encoding="utf-8")
)
frozen = json.loads(
    (root / "frozen_analysis.json").read_text(encoding="utf-8")
)

assert ledger["development_started"] is True
assert ledger["confirmation_started"] is False
assert development["verdict"] == "OPEN_CONFIRMATION"
assert manifest["development_complete"] is True
assert manifest["confirmation_open"] is True

expected = manifest["frozen_analysis_sha256"]
actual = hashlib.sha256(
    (root / "frozen_analysis.json").read_bytes()
).hexdigest()
assert actual == expected

assert frozen["source_sha256"] == manifest["source_sha256"]
PY

(
  cd outputs/green_bridge_v131
  sha256sum -c sha256sums.txt
)
```

Reverify every old v1.3 hash before confirmation.

---

# 25. Exact confirmation command

Confirmation is authorized only after Section 24 passes:

```bash
set -euo pipefail
cd /home/ccj/workspace_1/idle_1_green_bridge

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"

bash src/launch_green_bridge_v131.sh 4 confirmation
```

Confirmation may be issued exactly once.

Its success or failure is terminal. No rerun is authorized.

---

# 26. Explicit STOP conditions

The runner must write a durable v1.3.1 terminal `result.json` with `verdict="STOP"` before exiting whenever controlled execution reaches any of these conditions:

1. predecessor hash mismatch;
2. predecessor phase state mismatch;
3. wrong branch;
4. dirty worktree;
5. reviewed commit not an ancestor;
6. new output root already exists;
7. wrong attempt identity;
8. retry enabled;
9. environment mismatch;
10. package mismatch;
11. TransformerLens source mismatch;
12. model revision mismatch;
13. tokenizer mismatch;
14. Gate-04 failure;
15. same-TransformerLens no-op failure;
16. structural-frame failure;
17. legacy discrepancy not reproduced exactly;
18. any pre-unembed stagewise difference;
19. corrected raw manual-tail error above (2\times10^{-5});
20. any signed raw endpoint above (2\times10^{-5});
21. meaningful derivative relative error above (10^{-4});
22. near-zero absolute derivative bound failure;
23. batch-shape equivalence failure;
24. memory above 20 GB;
25. throughput projection above 24 GPU hours;
26. independent target raw error above (2\times10^{-5});
27. missing prepare artifact;
28. artifact hash mismatch;
29. scientific-invariance mismatch;
30. development artifact observed during prepare;
31. confirmation artifact observed before the lock opens;
32. uncommitted endpoint batch;
33. crash after a phase is claimed;
34. source or manifest mutation between phases;
35. development does not satisfy its frozen gates;
36. confirmation noise exceeds its frozen rule;
37. confirmation does not satisfy its frozen gates;
38. any proposal to modify scientific design in response to a result.

An uncontrolled process termination after ledger creation is also terminal even when `result.json` could not be written. It must be reported rather than retried.

---

# 27. Executor checklist

## 27.1 Before implementation

* [ ] Repository is exactly at reviewed commit `b8b5f7b509abda310f3e2414e0552a644a12b4bf`.
* [ ] Branch is `main`.
* [ ] Worktree is clean.
* [ ] All old v1.3 hashes pass.
* [ ] Old v1.3 ledger records attempt one and no retry.
* [ ] Old v1.3 development and confirmation flags are false.
* [ ] No old scientific response artifacts exist.
* [ ] Old external log hash passes.
* [ ] Old root has not been modified.

## 27.2 Archive

* [ ] Archive directory is newly created.
* [ ] External log is copied, not moved.
* [ ] Copied log hash is exact.
* [ ] Archive manifest contains all predecessor hashes.
* [ ] Frozen v1.3 scientific payload is serialized.
* [ ] Existing v1.3 root remains untouched.

## 27.3 Source correction

* [ ] Manual tail calls `model.unembed` on the full normalized sequence.
* [ ] Manual tail applies `apply_softcap`.
* [ ] Manual tail gathers only after full unembedding.
* [ ] Active manual tail contains no selected `W_U` multiplication.
* [ ] Independent target makes the same endpoint correction.
* [ ] Target remains code-isolated.
* [ ] Full-hook reference remains independently implemented.
* [ ] Legacy selected expression exists only as a prepare diagnostic.
* [ ] True central finite-difference derivative audit is implemented.
* [ ] Near-zero proof-derived bounds are implemented.
* [ ] Original center condition remains binding.
* [ ] Raw logits remain the primary quantity.
* [ ] Thresholds remain (2\times10^{-5}) and (10^{-4}).
* [ ] Precision remains float32.
* [ ] Structural theory and science code remain unchanged.
* [ ] Donor PCA remains absent.

## 27.4 Tests and commit

* [ ] All 150 CPU tests pass.
* [ ] No test is skipped.
* [ ] Old launcher is unchanged.
* [ ] New v1.3.1 launcher is present.
* [ ] Git diff contains only authorized files.
* [ ] Execution commit descends from reviewed commit.
* [ ] Worktree is clean after commit.
* [ ] New output root is absent.

## 27.5 Prepare

* [ ] Physical GPU 4 is exposed as `cuda:0`.
* [ ] Prepare command is issued exactly once.
* [ ] Legacy discrepancy is reproduced exactly.
* [ ] All upstream trace tensors are bitwise equal.
* [ ] Corrected raw logits pass.
* [ ] Signed raw endpoints pass.
* [ ] True derivative checks pass.
* [ ] Near-zero branches are explicit.
* [ ] Batch-shape equivalence passes.
* [ ] Path-target equivalence passes.
* [ ] Memory contract passes.
* [ ] Throughput hard cap passes.
* [ ] Scientific-invariance hashes match.
* [ ] All prepare artifacts are present and hashed.
* [ ] No development or confirmation artifact exists.
* [ ] `prepare_result.json` says `PREPARE_PASS`.
* [ ] `sha256sums.txt` verifies.

## 27.6 Development

* [ ] Prepare pass is mechanically verified.
* [ ] Old v1.3 hashes still pass.
* [ ] Source and worktree remain frozen.
* [ ] Development command is issued exactly once.
* [ ] No endpoint batch is repeated.
* [ ] Batch plan is unchanged.
* [ ] Development result is checked mechanically.
* [ ] Confirmation proceeds only for `OPEN_CONFIRMATION`.

## 27.7 Confirmation

* [ ] Confirmation lock is open.
* [ ] Frozen-analysis hash passes.
* [ ] Old v1.3 hashes still pass.
* [ ] Confirmation command is issued exactly once.
* [ ] Final result is treated as terminal.
* [ ] No retry or redesign occurs.

---

# 28. Final binding determination

The v1.3 `STOP` remains valid and terminal for its original identity. It may not be erased, retried, or recast as a successful audit.

The failure is not a scientific rejection of the structural-envelope matched-bypass theory. It is a pre-scientific executable-equivalence defect caused by replacing the frozen TransformerLens unembedding endpoint with a differently ordered selected-column matrix multiplication and by omitting the model’s output-softcap operation.

The large reported derivative-relative values do not measure derivatives. Their numerator is exactly the raw endpoint discrepancy, and their center denominator is the (10^{-5}) floor applied to a zero reference response. The old check must be replaced by actual central finite differences, with the unchanged (10^{-4}) relative threshold used only when the reference derivative is well-conditioned and a proof-derived absolute bound used otherwise.

Exactly one fresh v1.3.1 GPU attempt is authorized after the correction, tests, archive, clean commit, and all prerequisites above are complete.

The central theory remains:

> matched-bypass derivatives identify a basis-invariant ambient rank-one path operator, made probe-complete by the exact LayerNorm structural envelope.

Fixed-rank donor PCA remains permanently terminated.

# BINDING VERDICT — V1.3 STOP IMMUTABLE; PRE-SCIENTIFIC MANUAL-TAIL ENDPOINT DEFECT IDENTIFIED; ONE FRESH STRUCTURAL-ENVELOPE V1.3.1 ATTEMPT AUTHORIZED UNDER THE EXACT CORRECTION AND ONE-SHOT CONTRACT ABOVE

[1]: https://raw.githubusercontent.com/ScottBlizzard/idle_1/b8b5f7b509abda310f3e2414e0552a644a12b4bf/analysis/GREEN_SERVER_V13_PREPARE_STOP_20260825.md "https://raw.githubusercontent.com/ScottBlizzard/idle_1/b8b5f7b509abda310f3e2414e0552a644a12b4bf/analysis/GREEN_SERVER_V13_PREPARE_STOP_20260825.md"
[2]: https://raw.githubusercontent.com/TransformerLensOrg/TransformerLens/4a4dc26c750475b29e6f54b362c2aab988702c9c/transformer_lens/components/unembed.py "https://raw.githubusercontent.com/TransformerLensOrg/TransformerLens/4a4dc26c750475b29e6f54b362c2aab988702c9c/transformer_lens/components/unembed.py"
[3]: https://raw.githubusercontent.com/ScottBlizzard/idle_1/b8b5f7b509abda310f3e2414e0552a644a12b4bf/analysis/GPTPRO_GREEN_GATE08_V12_DECISION_20260805.md "https://raw.githubusercontent.com/ScottBlizzard/idle_1/b8b5f7b509abda310f3e2414e0552a644a12b4bf/analysis/GPTPRO_GREEN_GATE08_V12_DECISION_20260805.md"
[4]: https://raw.githubusercontent.com/ScottBlizzard/idle_1/b8b5f7b509abda310f3e2414e0552a644a12b4bf/src/green_bridge_spec.py "https://raw.githubusercontent.com/ScottBlizzard/idle_1/b8b5f7b509abda310f3e2414e0552a644a12b4bf/src/green_bridge_spec.py"
[5]: https://github.com/ScottBlizzard/idle_1/blob/b8b5f7b509abda310f3e2414e0552a644a12b4bf/src/green_bridge_tail.py?plain=1 "https://github.com/ScottBlizzard/idle_1/blob/b8b5f7b509abda310f3e2414e0552a644a12b4bf/src/green_bridge_tail.py?plain=1"
[6]: https://raw.githubusercontent.com/TransformerLensOrg/TransformerLens/4a4dc26c750475b29e6f54b362c2aab988702c9c/transformer_lens/components/mlps/mlp.py "https://raw.githubusercontent.com/TransformerLensOrg/TransformerLens/4a4dc26c750475b29e6f54b362c2aab988702c9c/transformer_lens/components/mlps/mlp.py"
[7]: https://raw.githubusercontent.com/ScottBlizzard/idle_1/b8b5f7b509abda310f3e2414e0552a644a12b4bf/src/exp_green_bridge_gpt2.py "https://raw.githubusercontent.com/ScottBlizzard/idle_1/b8b5f7b509abda310f3e2414e0552a644a12b4bf/src/exp_green_bridge_gpt2.py"
[8]: https://github.com/ScottBlizzard/idle_1/blob/b8b5f7b509abda310f3e2414e0552a644a12b4bf/src/green_bridge_path_target.py?plain=1 "https://github.com/ScottBlizzard/idle_1/blob/b8b5f7b509abda310f3e2414e0552a644a12b4bf/src/green_bridge_path_target.py?plain=1"
[9]: https://raw.githubusercontent.com/ScottBlizzard/idle_1/b8b5f7b509abda310f3e2414e0552a644a12b4bf/src/launch_green_bridge.sh "https://raw.githubusercontent.com/ScottBlizzard/idle_1/b8b5f7b509abda310f3e2414e0552a644a12b4bf/src/launch_green_bridge.sh"
[10]: https://raw.githubusercontent.com/ScottBlizzard/idle_1/b8b5f7b509abda310f3e2414e0552a644a12b4bf/src/test_green_bridge_contract.py "https://raw.githubusercontent.com/ScottBlizzard/idle_1/b8b5f7b509abda310f3e2414e0552a644a12b4bf/src/test_green_bridge_contract.py"
