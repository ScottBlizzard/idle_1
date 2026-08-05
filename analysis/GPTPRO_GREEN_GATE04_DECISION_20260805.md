<!-- filename: analysis/GPTPRO_GREEN_GATE04_DECISION_20260805.md -->

# GPTPRO Green Gate-04 Binding Decision — 2026-08-05

**Repository:** `https://github.com/ScottBlizzard/idle_1`  
**Reviewed branch:** `main`  
**Reviewed commit:** `0c81e05`  
**Theory base commit:** `126556f`  
**Previous stopped execution:** RTX 4090, frozen Python/CUDA/package environment  
**Scope of this decision:** numerical preflight Gate 04 and associated frozen numerical-error implementation  
**Binding verdict:** **B. AMEND_PREFLIGHT_AND_RERUN**

---

## 1. Binding Answers to the Nine Requested Questions

| Question | Binding answer |
|---|---|
| 1. Is there a documented frozen-version backend that should force Hugging Face and TransformerLens below `2e-5`? | **No.** Hugging Face eager attention is the correct canonical comparison backend, but it does not make the two implementations share an identical floating-point operation graph. |
| 2. Is the existing `2e-5` maximum absolute-logit tolerance realistically attainable? | **Not as a robust cross-implementation invariant.** It can pass accidentally on individual prompts, but it is too strict as a maximum over 3,200 float32 output coordinates produced by different faithful implementations. |
| 3. May Gate 04 be amended before scientific responses are observed? | **Yes.** Replace it with the exact multi-part fidelity audit in Section 4. |
| 4. Should manual-tail-versus-full-TransformerLens thresholds change? | **No.** They remain exactly unchanged. |
| 5. Is a different backend required? | **Hugging Face must be forced to `attn_implementation="eager"` to standardize the comparator.** This is not claimed to recover `2e-5`; it is part of the amended fidelity audit. |
| 6. Does the resolution preserve the theorem and experiment? | **Yes.** It changes neither the computational DAG nor any scientific intervention, target, gate coordinate, split, radius, baseline, or confirmatory decision. |
| 7. Binding verdict | **B. AMEND_PREFLIGHT_AND_RERUN.** |
| 8. Are exact implementation changes, tests, manifest changes, and commands provided? | **Yes.** See Sections 8–11. |
| 9. Is the amendment scientifically valid given the data-access state? | **Yes.** No development or confirmation responses were computed; the prior observations were donor-only technical diagnostics. The amended audit also uses a donor holdout panel disjoint from the original Gate-04 panel. |

The original run stopped at `04_HF_TL` with maximum year-logit error `1.526e-04`, and the repository records that no development or confirmation responses were computed and that confirmation remained locked. The TransformerLens source files used by the execution also matched the frozen source hashes. This is a numerical-contract failure before scientific observation, not evidence against the matched-bypass theorem or the transformer bridge. 

---

## 2. Numerical Finding: No Exact Cross-Implementation Backend Exists

### 2.1 Why Hugging Face eager and TransformerLens need not agree to `2e-5`

Under the frozen Hugging Face implementation, GPT-2 produces query, key, and value tensors through one fused `c_attn` projection and then splits its output. Under the frozen TransformerLens conversion, that fused tensor is separated into `W_Q`, `W_K`, and `W_V`; TransformerLens subsequently evaluates three separate linear projections. TransformerLens also performs its own attention-score, value aggregation, and per-head output-reduction sequence. The two libraries therefore implement the same mathematical weights and function with different float32 operation partitioning and summation orders. 

Hugging Face documents `attn_implementation="eager"` as a supported explicit backend choice, and its GPT-2 source shows the eager query–key and attention–value matrix multiplications. Nothing in that interface promises agreement with a separately implemented transformer to a fixed absolute tolerance. 

PyTorch explicitly states that float32 addition and multiplication are not associative, that mathematically identical computations are not guaranteed to be bitwise identical, and that batched or differently partitioned operations can produce different results even with identical inputs and deterministic execution. Deterministic algorithms reproduce a given operation graph; they do not make different operation graphs use the same reduction order. 

Consequently, none of the following is an exact documented fix for the original `2e-5` contract:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- disabling TF32;
- enabling deterministic algorithms;
- forcing Hugging Face eager attention;
- moving both implementations to CPU;
- setting float32 matmul precision to `highest`.

Those settings are still required for reproducibility, but they do not imply cross-library arithmetic identity.

### 2.2 Assessment of the observed discrepancy

The observed values are consistent with this operation-order explanation:

- original 32-prompt maximum on the 100 year logits: `1.526e-04`;
- forced Hugging Face eager, representative GPU prompt: `7.62939453125e-05`;
- eager CPU float32, representative prompt: `1.983642578125e-04`;
- deterministic layerwise residual maxima: approximately `9.5e-06` to `6.1e-05`.

These discrepancies are small, non-monotone across layers, sensitive to the backend, and considerably larger than `2e-5` without showing evidence of a weight, model-revision, or topology mismatch. 

A maximum tolerance of `2e-5` across

\[
32\times100=3200
\]

output coordinates is therefore not a realistic invariant for these two faithful float32 implementations. It is suitable for a same-implementation replay audit, but not as the sole test of cross-implementation model fidelity.

---

## 3. Binding Verdict

# B. AMEND_PREFLIGHT_AND_RERUN

`A. FIX_BACKEND_AND_RERUN` is rejected because no documented backend makes the two frozen implementations share an exact numerical operation graph, and forced eager execution has already exceeded `2e-5`.

`C. GREEN_STOP_AND_REDESIGN` is rejected because the failure does not obstruct the theorem, causal topology, selected mediator coordinates, path-specific target, or statistical protocol. It exposes an over-strict cross-library preflight criterion.

The authorization is conditional on all of the following:

1. Gate 04 is replaced exactly by the audit in Section 4.
2. Hugging Face is explicitly run with eager attention.
3. The original failed artifacts are preserved.
4. The frozen Richardson numerical-error propagation is implemented exactly as specified in Section 5.
5. All same-TransformerLens audits retain their original thresholds.
6. The amended code, decision document, manifest, and tests are committed before rerunning.
7. No second threshold adjustment is permitted after observing the amended holdout audit.

---

## 4. Exact Replacement for Gate 04

### 4.1 Purpose of the amended gate

The amended Gate 04 answers four separate questions:

1. **Were the exact GPT-2 parameters transferred into TransformerLens?**
2. **Do the two implementations reach the same block-10 mechanistic neighborhood to a small numerical tolerance?**
3. **Do their task-relevant year logits and Greater-Than contrast agree to a scientifically negligible tolerance?**
4. **Is any discrepancy consistent with float32 evaluation order rather than a wrong model, wrong hook, wrong tokenization, or wrong coordinate system?**

It must not attempt to prove bitwise equality between independent implementations.

### 4.2 Frozen comparison backend

Hugging Face must be loaded with:

```python
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    torch_dtype=torch.float32,
    attn_implementation="eager",
).eval().to(device)

hf_model.config.use_cache = False

if getattr(hf_model.config, "_attn_implementation", None) != "eager":
    raise GreenStop(
        "03_MODEL_CONFIG",
        "Hugging Face attention implementation is not eager",
    )
```

Every Gate-04 Hugging Face forward must use:

```python
hf_model(
    input_ids=tokens,
    use_cache=False,
    return_dict=True,
)
```

TransformerLens remains loaded by:

```python
HookedTransformer.from_pretrained_no_processing(...)
```

No TransformerLens weight processing, LayerNorm folding, centering, refactoring, or dtype change is authorized.

### 4.3 Exact prompt panel

The failed implementation selected donor pair records at sorted `pair_digest` ranks `0` through `15` and evaluated each clean and corrupt prompt. The amended binding audit must use a disjoint donor holdout:

```python
ranked = sorted(donor_records, key=lambda row: row.pair_digest)

legacy_gate04_records = ranked[0:16]
amended_gate04_records = ranked[16:32]
```

The binding panel is therefore:

- 16 donor pair records;
- one clean and one corrupt prompt per pair;
- 32 prompts total;
- exactly 16 clean and 16 corrupt;
- batch size one;
- clean prompt evaluated before corrupt prompt within each pair;
- records traversed in ascending `pair_digest` order.

The manifest must store:

- all 16 excluded legacy pair digests;
- all 16 amended holdout pair digests;
- the 32 ordered `(pair_digest, system)` keys;
- the SHA-256 hash of the ordered key list.

No prompt replacement is allowed.

### 4.4 Exact quantities compared

For each of the 32 prompts, compare the following.

#### A. Converted model parameters

Call the frozen TransformerLens conversion function:

```python
from transformer_lens.pretrained.weight_conversions.gpt2 import (
    convert_gpt2_weights,
)

expected_tl_state = convert_gpt2_weights(hf_model, model.cfg)
actual_tl_state = model.state_dict()
```

For every key returned by `convert_gpt2_weights`:

- the key must exist in `actual_tl_state`;
- the shape must match exactly;
- the dtype must match exactly;
- `torch.equal(actual.cpu(), expected.cpu())` must be true.

`unembed.b_U`, when present as an additional TransformerLens parameter, must be exactly zero. Any other nonzero forward-affecting parameter absent from the conversion dictionary is a failure.

This is a zero-tolerance parameter audit.

#### B. Raw 100-year-logit vector

At the final prompt position, gather the same frozen 100 suffix-token logits:

\[
L^{\mathrm{HF}}_p,\ L^{\mathrm{TL}}_p\in\mathbb R^{100},
\]

and define

\[
e^{\mathrm{raw}}_p
=
L^{\mathrm{HF}}_p-L^{\mathrm{TL}}_p.
\]

#### C. Centered 100-year-logit vector

Center each 100-vector separately:

\[
\bar L_p
=
L_p-\frac{1}{100}\mathbf 1\mathbf 1^\top L_p,
\]

and compare

\[
e^{\mathrm{ctr}}_p
=
\bar L^{\mathrm{HF}}_p-\bar L^{\mathrm{TL}}_p.
\]

Centering is task-relevant because the Greater-Than contrast has zero total weight and is invariant to a common logit offset.

#### D. Greater-Than task contrast

Using the clean suffix \(y\) associated with the donor pair for both the clean and corrupt prompt, construct the existing frozen contrast

\[
\ell_y[q]
=
\begin{cases}
-\dfrac{1}{y+1}, & q\le y,\\[4pt]
\dfrac{1}{99-y}, & q>y,
\end{cases}
\]

and compare the scalar margins

\[
e^{\mathrm{margin}}_p
=
\ell_y^\top
\left(
L^{\mathrm{HF}}_p-L^{\mathrm{TL}}_p
\right).
\]

#### E. Block-10 `resid_mid` anchor

Capture the input to Hugging Face block-10 `ln_2`, which is the post-attention residual corresponding to TransformerLens:

```text
blocks.10.hook_resid_mid
```

Compare the complete 768-dimensional final-position vector:

\[
e^{\mathrm{resid}}_p
=
R^{\mathrm{HF}}_{10,\mathrm{mid},p}
-
R^{\mathrm{TL}}_{10,\mathrm{mid},p}.
\]

The Hugging Face source computes `hidden_states = attn_output + residual` immediately before `ln_2`, so the `ln_2` forward-pre-hook is the correct corresponding anchor. 

#### F. Ten selected MLP-10 preactivations

Capture the output of:

```python
hf_model.transformer.h[10].mlp.c_fc
```

and compare its ten coordinates in the frozen order

```text
2326, 1138, 2287, 606, 2848, 2305, 46, 2659, 946, 1616
```

with:

```text
blocks.10.mlp.hook_pre
```

at the final prompt position.

#### G. Ten selected MLP-10 post-GELU values

Capture the input to:

```python
hf_model.transformer.h[10].mlp.c_proj
```

which is the post-activation MLP vector, and compare the same ten coordinates with:

```text
blocks.10.mlp.hook_post
```

at the final prompt position.

### 4.5 Exact aggregation

For each vector-valued quantity \(e_p\), concatenate all prompt-coordinate errors and compute:

\[
\operatorname{MAX}(e)
=
\max_{p,c}|e_{p,c}|,
\]

\[
\operatorname{RMS}(e)
=
\sqrt{
\frac{
\sum_{p,c}e_{p,c}^2
}{
\sum_p d_p
}
}.
\]

For the scalar task-margin errors:

\[
\operatorname{MAX}_{\mathrm{margin}}
=
\max_p |e^{\mathrm{margin}}_p|,
\]

\[
\operatorname{RMS}_{\mathrm{margin}}
=
\sqrt{
\frac1{32}
\sum_p
\left(e^{\mathrm{margin}}_p\right)^2
}.
\]

No percentile, median, trimmed mean, per-prompt averaging before the maximum, or pass-by-aggregate substitution is allowed.

Every per-prompt maximum and scalar margin error must also be serialized.

### 4.6 Exact thresholds

All rows must pass simultaneously.

| Quantity | Global maximum absolute error | Pooled RMS error |
|---|---:|---:|
| Converted parameter tensors | exactly `0` mismatches | not applicable |
| Raw 100-year logits | `3.0e-4` | `7.5e-5` |
| Centered 100-year logits | `2.5e-4` | `6.0e-5` |
| Greater-Than task contrast | `2.0e-4` | `5.0e-5` |
| Block-10 final-position `resid_mid`, 768 coordinates | `1.0e-4` | `2.0e-5` |
| Ten selected block-10 `hook_pre` coordinates | `5.0e-4` | `1.0e-4` |
| Ten selected block-10 `hook_post` coordinates | `5.0e-4` | `1.0e-4` |

The task-contrast maximum is the most directly relevant output threshold. An error of `2e-4` is:

\[
4\%
\]

of the smallest frozen per-bin absolute-RMSE improvement floor `0.005`, and

\[
0.2\%
\]

of the `0.10` absolute target-conditioning threshold. It is small enough to exclude a scientifically material task discrepancy while accommodating faithful float32 reduction-order variation.

### 4.7 Exact pass and stop rule

Gate 04 passes only when:

```python
weight_mapping_mismatch_count == 0
and raw_year_max_abs <= 3.0e-4
and raw_year_pooled_rms <= 7.5e-5
and centered_year_max_abs <= 2.5e-4
and centered_year_pooled_rms <= 6.0e-5
and margin_max_abs <= 2.0e-4
and margin_rms <= 5.0e-5
and resid_mid_max_abs <= 1.0e-4
and resid_mid_pooled_rms <= 2.0e-5
and selected_pre_max_abs <= 5.0e-4
and selected_pre_pooled_rms <= 1.0e-4
and selected_post_max_abs <= 5.0e-4
and selected_post_pooled_rms <= 1.0e-4
```

Use the terminal subgate identifiers:

```text
04_HF_TL_WEIGHT_MAP
04_HF_TL_FIDELITY
```

A failure of either subgate terminates the run. There is no subsequent backend search, threshold sweep, CPU fallback, float64 fallback, prompt substitution, or second amendment.

---

## 5. Downstream Numerical-Error Bounds

### 5.1 HF–TL disagreement must not enter \(\epsilon_y\)

The amended Gate-04 errors are **cross-implementation portability errors**. They do not measure the numerical repeatability of the implementation that generates the scientific endpoints.

All scientific predictor endpoints, target endpoints, basis anchors, radii, and confirmation quantities are computed using TransformerLens and the audited TransformerLens manual tail. Therefore:

\[
\boxed{
\epsilon_y
=
\max\left\{
10^{-7},
\text{same-implementation duplicate TL error}
\right\}.
}
\]

The HF–TL errors must be reported in `hook_audit.json` but must not be:

- added to \(\epsilon_y\);
- used as the finite-difference endpoint error;
- added to \(\epsilon_G,\epsilon_C,\epsilon_{\Delta H}\), or \(\epsilon_P\);
- used to change radii;
- used to invalidate a scientific item after Gate 04 has passed.

Adding the cross-library discrepancy to the finite-difference noise model would conflate model-portability validation with repeated evaluation of the actual scientific estimator.

### 5.2 Mandatory correction discovered during this review

The frozen bridge document specifies Richardson-aware numerical propagation:

\[
\eta_G^R=\frac{3\epsilon_y}{h_{2j}},
\qquad
\eta_C^R=\frac{64\epsilon_y}{3h_{2j}^2},
\qquad
\eta_J^R=\frac{3\epsilon_y}{h_1},
\qquad
\eta_H^R=\frac{17\epsilon_y}{3h_1h_{2j}},
\]

followed by full-versus-half discrepancy terms, inverse-conditioning propagation, and a gate-level Frobenius bound. 

The implementation at reviewed commit `0c81e05` instead currently uses:

```python
eps_g = epsilon_y / hz
eps_c = 4 * epsilon_y / (hz * hz)
eps_p = epsilon_y / (hx * hz)
```

and sets the item error to:

```python
abs(theta - theta_half)
```

without applying the frozen inverse and tensor error propagation. 

This is a pre-response implementation defect. It must be corrected before the rerun. It is not an authorized change to the theory or thresholds; it is required to make the executable implementation match the already-frozen numerical protocol.

### 5.3 Exact required numerical propagation

For each gate, define:

\[
\widehat{\Delta H}^R
=
\widehat H^{P,R}
-
\widehat H^{C,R},
\]

\[
\widehat{\Delta H}^{1/2}
=
\widehat H^{P,1/2}
-
\widehat H^{C,1/2}.
\]

With \(k=100\), compute:

\[
\eta_G
=
\frac{3\epsilon_y}{h_{2j}},
\]

\[
\eta_C
=
\frac{64\epsilon_y}{3h_{2j}^2},
\]

\[
\eta_J
=
\frac{3\epsilon_y}{h_1},
\]

\[
\eta_H
=
\frac{17\epsilon_y}{3h_1h_{2j}}.
\]

Then:

\[
\epsilon_G
=
\left\|
\widehat G^R-\widehat G^{1/2}
\right\|_2
+
\sqrt{100}\,\eta_G,
\]

\[
\epsilon_C
=
\left\|
\widehat C^R-\widehat C^{1/2}
\right\|_2
+
\sqrt{100}\,\eta_C,
\]

and, for each residual coordinate \(i\),

\[
\epsilon_{\Delta H,i}
=
\left\|
\widehat{\Delta H}^R_i
-
\widehat{\Delta H}^{1/2}_i
\right\|_2
+
2\sqrt{100}\,\eta_H.
\]

An active inverse is numerically admissible only when:

\[
\left\|\widehat C^R\right\|_2>\epsilon_C.
\]

Then compute:

\[
A_{\max,i}
=
\frac{
\left\|
\widehat{\Delta H}^R_i
\right\|_2
+
\epsilon_{\Delta H,i}
}{
\left\|
\widehat C^R
\right\|_2
-
\epsilon_C
},
\]

\[
\epsilon_{A,i}
=
\frac{
\epsilon_{\Delta H,i}
+
A_{\max,i}\epsilon_C
}{
\left\|
\widehat C^R
\right\|_2
},
\]

\[
\epsilon_{P,i}
=
\epsilon_G A_{\max,i}
+
\left\|
\widehat G^R
\right\|_2
\epsilon_{A,i},
\]

\[
\epsilon_{P,F}
=
\sqrt{
\sum_{i=1}^{4}
\epsilon_{P,i}^2
}.
\]

The active-gate signal-to-noise conditions must use:

\[
\left\|\widehat C^R\right\|_2
\ge20\epsilon_C,
\]

\[
\left\|\widehat G^R\right\|_2
\ge20\epsilon_G,
\]

\[
\left\|\widehat P^R\right\|_F
\ge20\epsilon_{P,F}.
\]

The item-direction uncertainty contributed by an active gate is:

\[
\epsilon_{\theta,j}
=
\|\ell_y\|_2
\|\delta_n\|_2
\epsilon_{P,F}.
\]

The system-level item bound is the sum over all ten selected gates:

\[
\epsilon_{\theta,s,n}
=
\sum_{j\in J}
\epsilon_{\theta,s,n,j}.
\]

### 5.4 Certified-target-null gates

For a candidate certified-null gate, define:

\[
B^R_j
=
\|\ell_y\|_2
\|\delta_n\|_2
\left(
\|\widehat G^R_j\|_2+\epsilon_{G,j}
\right)
\|A^{\mathrm{WB}}_j\|_2.
\]

The gate may be certified target-null only if:

\[
\|\widehat G^R_j\|_2
\le5\epsilon_{G,j},
\]

\[
B^R_j\le0.005,
\]

and the full-versus-half upper-bound change satisfies:

\[
\left|
B^1_j-B^{1/2}_j
\right|
\le0.005,
\]

where

\[
B^\rho_j
=
\|\ell_y\|_2
\|\delta_n\|_2
\|\widehat G^\rho_j\|_2
\|A^{\mathrm{WB}}_j\|_2.
\]

A certified-null gate contributes `0` to the point tensor prediction but contributes \(B^R_j\) to the item uncertainty bound. It must not disappear from `theta_error`.

### 5.5 Cell-level SNR

Retain the frozen conservative cell aggregation:

\[
E_c
=
\frac1{|N_c|}
\sum_{n\in N_c}
\left(
\epsilon_{\theta,\mathrm{pat},n}
+
\epsilon_{\theta,\mathrm{tar},n}
\right),
\]

\[
\operatorname{SNR}_c
=
\frac{
\widehat T^{\mathrm{MB}}_c
}{
\max\{E_c,10^{-8}\}
}.
\]

The development requirement remains:

- at least 10 of 16 development cells;
- each with \(\operatorname{SNR}_c\ge3\).

No threshold changes are authorized.

---

## 6. Manual-Tail and Same-TransformerLens Audits

The original strict thresholds remain binding.

| Audit | Binding threshold | Decision |
|---|---:|---|
| Manual tail versus full-hook year logits | maximum absolute error `≤2e-5` | unchanged |
| Manual tail versus full-hook derivative vector | relative error `≤1e-4` | unchanged |
| Center replay | RMS `≤2e-6` | unchanged |
| Center replay | maximum absolute error `≤2e-5` | unchanged |
| Block-8 clean-to-clean and corrupt-to-corrupt no-op patch | maximum absolute error `≤2e-5` | unchanged |
| Untouched hook entries | maximum mutation `≤1e-7` | unchanged |

The original protocol expressly treats manual-tail-versus-full-hook execution as a separate audit with 32 conditions: eight center, eight \(x\)-only, eight \(z\)-only, four path-mixed, and four control-mixed. 

These comparisons are different from HF versus TransformerLens:

- both sides use the same TransformerLens weights;
- both sides use the same TransformerLens block-11, final-LayerNorm, and unembedding code;
- the audit is intended to detect a wrong manual tail, wrong anchor, wrong operation order, wrong clamping rule, or wrong continuation point;
- it is therefore appropriate to retain a much stricter tolerance.

No HF–TL tolerance may be borrowed by `tail_audit`, `center_replay`, or `no_op_audit`.

---

## 7. Preservation of the Scientific Design

| Scientific component | Preserved? | Binding statement |
|---|---|---|
| Matched-bypass identification theorem | **Yes** | No equation or assumption changes. |
| Structural inverse \(A_i=\langle C,\Delta H_i\rangle/\|C\|^2\) | **Yes** | No regularizer, pseudo-inverse, or new structural assumption is introduced. |
| Actual MLP-10 gate coordinates | **Yes** | The exact ten frozen indices remain unchanged and unrotated. |
| Upstream intervention | **Yes** | Remains `blocks.10.hook_resid_mid`. |
| Gate preactivation intervention | **Yes** | Remains `blocks.10.mlp.hook_pre`. |
| Gate clamping and matched control | **Yes** | Path and control definitions remain unchanged. |
| Direct residual bypass preservation in path/control responses | **Yes** | Unchanged. |
| Independent target bypass subtraction | **Yes** | Remains at `blocks.10.hook_resid_post`. |
| Independent target module and import firewall | **Yes** | Unchanged. |
| MLP-8 clean-to-corrupt comparator patch | **Yes** | Unchanged. |
| Four-dimensional residual basis | **Yes** | Same donor construction and SVD. |
| Donor population | **Yes** | Same finite donor records; only the Gate-04 audit panel moves to disjoint ranks `16:32`. |
| Development/confirmation finite population | **Yes** | Byte-for-byte unchanged. |
| Confirmation lock | **Yes** | Remains closed until the frozen development decision. |
| Radii and radius floors | **Yes** | Unchanged. |
| Baselines and equal-budget design | **Yes** | Unchanged. |
| Statistical calibration | **Yes** | Unchanged. |
| Development thresholds | **Yes** | Unchanged. |
| Confirmation thresholds and bootstrap | **Yes** | Unchanged. |
| ICLR-oral-level main claim | **Yes** | The amendment improves numerical validity without weakening the causal or empirical claim. |

The code-isolated target already implements the joint ten-gate path intervention and residual-bypass subtraction, while the tail implements the distinct path/control/joint modes. Those files need no scientific modification for this decision. 

---

## 8. Exact Implementation Amendments

### 8.1 `src/green_bridge_spec.py`

Change:

```python
SCHEMA_VERSION = "green-bridge-v1"
```

to:

```python
SCHEMA_VERSION = "green-bridge-v1.1"
GATE04_AMENDMENT_ID = "GPTPRO-GREEN-GATE04-v2-20260805"
HF_ATTN_IMPLEMENTATION = "eager"
GATE04_LEGACY_PAIR_SLICE = (0, 16)
GATE04_HOLDOUT_PAIR_SLICE = (16, 32)
```

Remove:

```python
hf_tl_max_abs: float = 2e-5
```

and add exactly:

```python
hf_tl_raw_year_max_abs: float = 3.0e-4
hf_tl_raw_year_pooled_rms: float = 7.5e-5

hf_tl_centered_year_max_abs: float = 2.5e-4
hf_tl_centered_year_pooled_rms: float = 6.0e-5

hf_tl_margin_max_abs: float = 2.0e-4
hf_tl_margin_rms: float = 5.0e-5

hf_tl_resid_mid_max_abs: float = 1.0e-4
hf_tl_resid_mid_pooled_rms: float = 2.0e-5

hf_tl_selected_pre_max_abs: float = 5.0e-4
hf_tl_selected_pre_pooled_rms: float = 1.0e-4

hf_tl_selected_post_max_abs: float = 5.0e-4
hf_tl_selected_post_pooled_rms: float = 1.0e-4
```

The following existing values must remain bit-for-bit unchanged:

```python
hook_untouched_max = 1e-7
no_op_max_abs = 2e-5
tail_max_abs = 2e-5
tail_derivative_relative = 1e-4
center_rms = 2e-6
center_max_abs = 2e-5
```

Add to `FROZEN_SPEC`:

```python
"gate04_amendment": {
    "id": GATE04_AMENDMENT_ID,
    "hf_attention_implementation": HF_ATTN_IMPLEMENTATION,
    "legacy_pair_slice": GATE04_LEGACY_PAIR_SLICE,
    "holdout_pair_slice": GATE04_HOLDOUT_PAIR_SLICE,
    "prompts_per_pair": ("clean", "corrupt"),
    "prompt_count": 32,
    "batch_size": 1,
    "parameter_mapping_exact": True,
    "hf_tl_error_enters_epsilon_y": False,
},
"numerical_error_contract": "frozen-richardson-propagation-v1",
```

### 8.2 `src/exp_green_bridge_gpt2.py`: runtime

Before CUDA work begins, require:

```python
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise GreenStop(
        "01_ENVIRONMENT",
        "CUBLAS_WORKSPACE_CONFIG must equal :4096:8",
    )
```

After importing PyTorch:

```python
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)

if not torch.are_deterministic_algorithms_enabled():
    raise GreenStop("01_ENVIRONMENT", "deterministic algorithms are disabled")
```

Record all four values in the manifest.

### 8.3 `src/exp_green_bridge_gpt2.py`: model loading

Use the eager Hugging Face load shown in Section 4.2 and include in the observed configuration:

```python
"hf_attention_implementation": getattr(
    hf_model.config,
    "_attn_implementation",
    None,
),
"hf_use_cache": bool(hf_model.config.use_cache),
"float32_matmul_precision": torch.get_float32_matmul_precision(),
"cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
```

### 8.4 `src/exp_green_bridge_gpt2.py`: prompt selector

Add:

```python
def gate04_record_panels(records):
    ranked = sorted(records, key=lambda row: row.pair_digest)
    legacy = ranked[0:16]
    holdout = ranked[16:32]

    legacy_ids = {row.pair_digest for row in legacy}
    holdout_ids = {row.pair_digest for row in holdout}

    if len(legacy) != 16 or len(holdout) != 16:
        raise GreenStop("04_HF_TL_FIDELITY", "Gate-04 panel has wrong size")
    if legacy_ids & holdout_ids:
        raise GreenStop("04_HF_TL_FIDELITY", "Gate-04 panels overlap")

    return legacy, holdout
```

Both `hf_tl_audit` and `no_op_audit` must receive the same explicit `holdout` list. Neither function may independently select records.

### 8.5 `src/exp_green_bridge_gpt2.py`: Hugging Face anchor capture

Implement the capture using removable PyTorch hooks:

```python
def capture_hf_gate04(hf_model, tokens):
    captured = {}

    def resid_mid_pre_hook(_module, args):
        captured["resid_mid"] = args[0].detach()

    def pre_hook(_module, _args, output):
        captured["pre"] = output.detach()

    def post_pre_hook(_module, args):
        captured["post"] = args[0].detach()

    handles = [
        hf_model.transformer.h[10].ln_2.register_forward_pre_hook(
            resid_mid_pre_hook
        ),
        hf_model.transformer.h[10].mlp.c_fc.register_forward_hook(
            pre_hook
        ),
        hf_model.transformer.h[10].mlp.c_proj.register_forward_pre_hook(
            post_pre_hook
        ),
    ]

    try:
        with torch.inference_mode():
            logits = hf_model(
                input_ids=tokens,
                use_cache=False,
                return_dict=True,
            ).logits
    finally:
        for handle in handles:
            handle.remove()

    required = {"resid_mid", "pre", "post"}
    if set(captured) != required:
        raise GreenStop(
            "04_HF_TL_FIDELITY",
            f"incomplete Hugging Face capture: {sorted(captured)}",
        )

    return logits, captured
```

### 8.6 `src/exp_green_bridge_gpt2.py`: weight mapping

Add a zero-tolerance parameter audit before the first prompt comparison.

The result must serialize:

```json
{
  "mapped_tensor_count": "<integer>",
  "missing_keys": [],
  "shape_mismatches": [],
  "dtype_mismatches": [],
  "value_mismatches": [],
  "unembed_b_U_present": true,
  "unembed_b_U_nonzero_count": 0,
  "passed": true
}
```

Any nonempty mismatch list is `04_HF_TL_WEIGHT_MAP`.

### 8.7 `src/exp_green_bridge_gpt2.py`: Gate-04 metrics

For each holdout prompt:

1. tokenize once;
2. run Hugging Face eager, batch size one;
3. capture the TransformerLens anchor once;
4. cast compared values to float64 after forward execution;
5. calculate all six error families;
6. append raw error arrays only to the in-memory pooled accumulator;
7. serialize per-prompt maxima and the scalar margin error;
8. retain the TransformerLens year logits and MLP-8 output needed by the no-op audit.

The returned object must contain:

```json
{
  "audit_version": "hf-tl-fidelity-v2",
  "hf_attention_implementation": "eager",
  "batch_size": 1,
  "legacy_pair_digests": [],
  "holdout_pair_digests": [],
  "ordered_prompt_keys": [],
  "ordered_prompt_keys_sha256": "...",
  "n_pairs": 16,
  "n_prompts": 32,
  "weight_mapping": {},
  "metrics": {
    "raw_year_logits": {
      "max_abs": 0.0,
      "pooled_rms": 0.0
    },
    "centered_year_logits": {
      "max_abs": 0.0,
      "pooled_rms": 0.0
    },
    "task_margin": {
      "max_abs": 0.0,
      "rms": 0.0
    },
    "resid_mid": {
      "max_abs": 0.0,
      "pooled_rms": 0.0
    },
    "selected_pre": {
      "max_abs": 0.0,
      "pooled_rms": 0.0
    },
    "selected_post": {
      "max_abs": 0.0,
      "pooled_rms": 0.0
    }
  },
  "thresholds": {},
  "per_prompt": [],
  "hf_tl_error_enters_epsilon_y": false,
  "passed": true,
  "tl_references": {}
}
```

### 8.8 Frozen numerical propagation module

Add:

```text
src/green_bridge_numerics.py
```

It must:

- import only NumPy, `math`, dataclasses, and the numerical `GateJet` structures;
- implement the formulas in Section 5 exactly;
- contain no model hooks;
- contain no target access;
- contain no baseline code;
- contain no development or confirmation data access.

Add it to `SOURCE_FILES`.

`classify_gate` must consume the returned `epsilon_G`, `epsilon_C`, and `epsilon_P_F` instead of the existing simplified `eps_g`, `eps_c`, and `eps_p`.

`mixed_system` must sum active-gate contraction bounds and certified-null bounds into `theta_error`.

### 8.9 Files that must not receive scientific changes

No theory or intervention changes are authorized in:

```text
src/matched_bypass_gate.py
src/green_bridge_tail.py
src/green_bridge_path_target.py
src/green_bridge_dataset.py
src/analyze_green_bridge.py
requirements-green-bridge.lock
```

Only import/hash plumbing needed for the new numerical helper may touch surrounding files.

`requirements-green-bridge.lock` remains byte-for-byte unchanged. The reviewed lockfile already pins the required Python packages. 

---

## 9. Exact Manifest Amendments

The new manifest must use:

```yaml
schema_version: green-bridge-manifest-v1.1

amendment:
  id: GPTPRO-GREEN-GATE04-v2-20260805
  decision_document: analysis/GPTPRO_GREEN_GATE04_DECISION_20260805.md
  decision_base_commit: 0c81e05
  theory_base_commit: 126556f
  previous_terminal_gate: 04_HF_TL
  previous_observed_max_abs_year_logit_error: 0.0001526
  previous_development_responses_observed: false
  previous_confirmation_responses_observed: false
  confirmation_was_locked: true
  amendment_scope:
    - HF-versus-TransformerLens preflight fidelity audit
    - conformance repair for frozen Richardson numerical propagation
  scientific_design_changed: false
  second_threshold_amendment_allowed: false

gate04:
  audit_version: hf-tl-fidelity-v2
  hf_attention_implementation: eager
  transformer_lens_processing: none
  dtype: float32
  batch_size: 1
  use_cache: false

  prompt_selection:
    population: donor
    ordering: ascending pair_digest
    excluded_legacy_pair_ranks:
      start_inclusive: 0
      stop_exclusive: 16
    audited_holdout_pair_ranks:
      start_inclusive: 16
      stop_exclusive: 32
    prompts_per_pair:
      - clean
      - corrupt
    n_pairs: 16
    n_prompts: 32

  parameter_mapping:
    converter: transformer_lens.pretrained.weight_conversions.gpt2.convert_gpt2_weights
    mapped_tensors_must_be_bitwise_equal: true
    allowed_extra_parameter: unembed.b_U
    allowed_extra_parameter_must_be_zero: true

  thresholds:
    raw_year_logits:
      max_abs: 0.0003
      pooled_rms: 0.000075
    centered_year_logits:
      max_abs: 0.00025
      pooled_rms: 0.00006
    task_margin:
      max_abs: 0.0002
      rms: 0.00005
    resid_mid:
      max_abs: 0.0001
      pooled_rms: 0.00002
    selected_pre:
      max_abs: 0.0005
      pooled_rms: 0.0001
    selected_post:
      max_abs: 0.0005
      pooled_rms: 0.0001

  downstream_error:
    enters_epsilon_y: false
    reporting_only_after_gate_pass: true

same_transformerlens_audits:
  no_op_max_abs: 0.00002
  tail_max_abs: 0.00002
  tail_derivative_relative: 0.0001
  center_rms: 0.000002
  center_max_abs: 0.00002
  hook_untouched_max: 0.0000001

numerical_error_contract:
  version: frozen-richardson-propagation-v1
  epsilon_y_source: same-TransformerLens duplicate audit only
  eta_G: 3*epsilon_y/h2
  eta_C: 64*epsilon_y/(3*h2^2)
  eta_J: 3*epsilon_y/h1
  eta_H: 17*epsilon_y/(3*h1*h2)
  active_tensor_snr_uses_epsilon_P_F: true
  certified_null_bound_enters_theta_error: true

preserved:
  matched_bypass_theorem: true
  selected_gates: true
  resid_mid_site: true
  matched_control: true
  independent_target: true
  residual_bypass_subtraction: true
  basis_design: true
  radii: true
  finite_population: true
  baselines: true
  development_rules: true
  confirmation_lock: true
  confirmation_rules: true
```

Add a protocol hash section covering:

```text
analysis/GPTPRO_GREEN_BRIDGE_20260805.md
analysis/GREEN_SERVER_GATE04_20260805.md
analysis/GPTPRO_GREEN_GATE04_DECISION_20260805.md
requirements-green-bridge.lock
```

The execution commit must be a clean descendant of `0c81e05`.

---

## 10. Exact Required Tests

Add all of the following to `src/test_green_bridge_contract.py` or a new CPU-only contract test imported by it.

### 10.1 Gate-04 specification tests

```text
test_gate04_schema_is_v1_1
test_gate04_legacy_and_holdout_slices_are_exact
test_gate04_holdout_has_16_unique_pair_records
test_gate04_holdout_is_disjoint_from_legacy_panel
test_gate04_ordered_prompt_count_is_32
test_gate04_prompt_order_is_clean_then_corrupt
test_gate04_thresholds_equal_binding_values
test_original_hf_tl_max_abs_field_is_absent
test_same_tl_thresholds_are_unchanged
```

### 10.2 Aggregation tests

Using fixed synthetic arrays:

```text
test_pooled_rms_flattens_all_prompts_and_coordinates
test_global_max_is_not_mean_of_prompt_maxima
test_centered_logit_metric_centers_each_prompt_separately
test_margin_metric_uses_clean_suffix_for_clean_and_corrupt
test_margin_metric_is_invariant_to_constant_logit_shift
test_all_gate04_submetrics_must_pass
```

### 10.3 Weight-map tests

```text
test_weight_map_rejects_missing_key
test_weight_map_rejects_shape_mismatch
test_weight_map_rejects_dtype_mismatch
test_weight_map_rejects_one_bit_value_change
test_weight_map_accepts_exact_rearranged_gpt2_state
test_unembed_bias_must_be_exactly_zero
```

### 10.4 Backend-contract tests

Source or AST tests must establish:

```text
test_hf_load_forces_eager_attention
test_hf_gate04_forward_disables_cache
test_gate04_batch_size_is_one
test_gate04_captures_ln2_input_as_resid_mid
test_gate04_captures_c_fc_output_as_pre
test_gate04_captures_c_proj_input_as_post
test_hf_hooks_are_removed_in_finally
```

### 10.5 Numerical-propagation tests

For fixed \(k=100\), \(\epsilon_y\), \(h_1\), and \(h_2\), test every closed-form value:

```text
test_eta_G_matches_3_epsilon_over_h2
test_eta_C_matches_64_epsilon_over_3_h2_squared
test_eta_J_matches_3_epsilon_over_h1
test_eta_H_matches_17_epsilon_over_3_h1_h2
test_epsilon_G_includes_richardson_half_discrepancy
test_epsilon_C_includes_richardson_half_discrepancy
test_epsilon_delta_H_includes_both_path_and_control_noise
test_A_max_uses_C_norm_minus_epsilon_C
test_epsilon_A_matches_frozen_formula
test_epsilon_P_matches_frozen_formula
test_epsilon_P_F_is_axiswise_l2_norm
test_active_curvature_snr_uses_epsilon_C
test_active_gate_response_snr_uses_epsilon_G
test_active_tensor_snr_uses_epsilon_P_F
test_certified_null_bound_uses_contrast_delta_G_and_whitebox_A
test_certified_null_bound_is_added_to_theta_error
test_cell_error_bound_sums_target_and_patched_item_bounds
```

One test must explicitly fail against the reviewed `0c81e05` simplified formulas.

### 10.6 Separation tests

```text
test_hf_tl_metrics_do_not_enter_epsilon_y
test_epsilon_y_is_derived_only_from_duplicate_noise_audit
test_tail_thresholds_do_not_reference_gate04_thresholds
test_target_module_import_firewall_remains_intact
test_confirmation_lock_remains_intact
test_requirements_lock_is_unchanged
```

### 10.7 GPU prepare assertions

The actual `prepare` phase is the binding GPU test. After it returns successfully:

- `hook_audit.json` must have audit version `hf-tl-fidelity-v2`;
- backend must be `eager`;
- prompt count must be 32;
- parameter mismatch count must be zero;
- every named metric must pass;
- `hf_tl_error_enters_epsilon_y` must be false;
- the no-op audit must pass at `2e-5`;
- `tail_audit.json` must pass the unchanged thresholds;
- no development or confirmation artifact may exist.

---

## 11. Exact Rerun Procedure

### 11.1 Commit and archive

Run from the repository root:

```bash
set -euo pipefail

git checkout main
test "$(git rev-parse --short=7 HEAD)" = "0c81e05"
test -z "$(git status --porcelain)"

# Implement this binding decision, including the numerical propagation repair.

git add \
  analysis/GPTPRO_GREEN_GATE04_DECISION_20260805.md \
  src/green_bridge_spec.py \
  src/green_bridge_numerics.py \
  src/exp_green_bridge_gpt2.py \
  src/test_green_bridge_contract.py

git commit -m "Amend Gate04 fidelity audit before scientific responses"

git merge-base --is-ancestor 0c81e05 HEAD
test -z "$(git status --porcelain)"

AMENDMENT_COMMIT="$(git rev-parse HEAD)"
printf '%s\n' "${AMENDMENT_COMMIT}"
```

Preserve the failed run:

```bash
set -euo pipefail

OLD_ROOT="outputs/green_bridge"
ARCHIVE_ROOT="outputs/green_bridge_gate04_stop_0c81e05_20260805"

test -f "${OLD_ROOT}/result.json"
test ! -e "${ARCHIVE_ROOT}"

python - <<'PY'
import json
from pathlib import Path

result = json.loads(
    Path("outputs/green_bridge/result.json").read_text(encoding="utf-8")
)
assert result["verdict"] == "STOP"
assert result["first_failed_gate"] == "04_HF_TL"
PY

mv "${OLD_ROOT}" "${ARCHIVE_ROOT}"

sha256sum \
  "${ARCHIVE_ROOT}/result.json" \
  "${ARCHIVE_ROOT}/manifest.json" \
  > "${ARCHIVE_ROOT}/ARCHIVED_GATE04_SHA256.txt"

mkdir -p outputs/green_bridge
```

The archived directory must never be reused as a cache for the amended run.

### 11.2 Frozen environment

```bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${PWD}/src"

python --version
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

The visible `cuda:0` must correspond to the selected physical RTX 4090.

### 11.3 CPU contract tests

```bash
set -euo pipefail

export PYTHONPATH="${PWD}/src"

python -m unittest discover \
  -s src \
  -p 'test_green_bridge_contract.py' \
  -v
```

All existing tests and all newly required tests must pass.

### 11.4 Prepare phase only

Do not use `--phase all`.

```bash
set -euo pipefail

mkdir -p logs

python src/exp_green_bridge_gpt2.py \
  --phase prepare \
  --device cuda:0 \
  --output-root outputs/green_bridge \
  2>&1 | tee logs/green_bridge_prepare_gate04_v2.log
```

### 11.5 Mechanical prepare verification

```bash
set -euo pipefail

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/green_bridge")

manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
audit = json.loads((root / "hook_audit.json").read_text(encoding="utf-8"))
tail = json.loads((root / "tail_audit.json").read_text(encoding="utf-8"))

assert manifest["schema_version"] == "green-bridge-manifest-v1.1"
assert manifest["prepare_complete"] is True
assert manifest["amendment"]["id"] == "GPTPRO-GREEN-GATE04-v2-20260805"
assert manifest.get("confirmation_open", False) is False

hf = audit["hf_vs_tl"]
assert hf["audit_version"] == "hf-tl-fidelity-v2"
assert hf["hf_attention_implementation"] == "eager"
assert hf["batch_size"] == 1
assert hf["n_pairs"] == 16
assert hf["n_prompts"] == 32
assert len(hf["legacy_pair_digests"]) == 16
assert len(hf["holdout_pair_digests"]) == 16
assert not set(hf["legacy_pair_digests"]) & set(hf["holdout_pair_digests"])
assert hf["weight_mapping"]["passed"] is True
assert hf["weight_mapping"]["value_mismatches"] == []
assert hf["hf_tl_error_enters_epsilon_y"] is False
assert hf["passed"] is True

assert audit["no_op_patch"]["max_abs"] <= 2.0e-5
assert tail["max_abs"] <= 2.0e-5
assert tail["max_derivative_relative"] <= 1.0e-4

for forbidden in (
    "noise_audit_dev.json",
    "dev_tensor_scores.parquet",
    "dev_energy_targets.parquet",
    "dev_result.json",
    "frozen_analysis.json",
    "confirm_tensor_scores.parquet",
    "confirm_energy_targets.parquet",
):
    assert not (root / forbidden).exists(), forbidden
PY
```

If this verification fails, stop. Do not run development.

### 11.6 Development phase

Only after successful prepare verification:

```bash
set -euo pipefail

python src/exp_green_bridge_gpt2.py \
  --phase development \
  --device cuda:0 \
  --output-root outputs/green_bridge \
  2>&1 | tee logs/green_bridge_development.log
```

The existing code must mechanically stop unless the frozen development decision opens confirmation.

### 11.7 Confirmation phase

Run confirmation only when both conditions hold:

```python
manifest["confirmation_open"] is True
```

and:

```text
outputs/green_bridge/frozen_analysis.json
```

exists with matching source and manifest hashes.

```bash
set -euo pipefail

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/green_bridge")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

assert manifest["confirmation_open"] is True
assert (root / "frozen_analysis.json").is_file()
PY

python src/exp_green_bridge_gpt2.py \
  --phase confirmation \
  --device cuda:0 \
  --output-root outputs/green_bridge \
  2>&1 | tee logs/green_bridge_confirmation.log
```

No retry is authorized.

---

## 12. Scientific Validity of the Amendment

The amendment is scientifically valid for five independent reasons.

First, the previous run stopped before the donor basis, radii, development tensor responses, development target responses, development decision, or confirmation responses were computed. The repository explicitly records that no development or confirmation model responses were accessed. 

Second, the observations used to diagnose Gate 04 were donor-only numerical implementation diagnostics. They contained no value of the independent path-specific target, no mixed tensor score, no baseline score, no development RMSE, and no confirmation statistic.

Third, the amended binding prompt panel excludes the original Gate-04 pair records. Thresholds are fixed in this document before evaluating the disjoint ranks `16:32` holdout panel.

Fourth, the amendment does not change any variable whose value could improve the scientific result:

- no mediator coordinate;
- no intervention site;
- no corruption;
- no target;
- no basis dimension;
- no radius;
- no cell;
- no baseline;
- no development threshold;
- no confirmation threshold;
- no bootstrap rule.

Fifth, correcting the Richardson numerical propagation cannot be characterized as outcome adaptation. The formulas were already frozen in the authorized bridge document; the reviewed code simply did not yet implement them. The correction restores conformance and makes the eventual SNR and admissibility decisions more conservative, not less.

The old `2e-5` result remains permanently recorded as a failed preflight under protocol version 1. The amended execution is a separately manifested version-1.1 run. This provenance must be visible in the repository and eventual paper supplement.

---

# Final Binding Decision

The matched-bypass transformer bridge remains theoretically and scientifically GREEN.

The original HF-versus-TransformerLens `2e-5` maximum-logit gate is numerically over-strict for two faithful float32 implementations with different accumulation graphs. There is no documented frozen-version backend that makes that tolerance a justified invariant.

Authorize exactly one amended run under the Gate-04 fidelity contract, same-TransformerLens thresholds, numerical-propagation correction, manifest, tests, and commands specified above.

# B. AMEND_PREFLIGHT_AND_RERUN