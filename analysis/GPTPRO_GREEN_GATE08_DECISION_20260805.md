<!-- intended filename: analysis/GPTPRO_GREEN_GATE08_DECISION_20260805.md -->

# GPTPRO Green Gate-08 Binding Decision — 2026-08-05

**Repository:** `https://github.com/ScottBlizzard/idle_1`  
**Reviewed branch:** `main`  
**Reviewed commit:** `b87300a6f56cb4706db090486d8bec77a2fc2b23`  
**Stopped execution commit:** `5083774e03b99c9958312c6686cf3ead40c3c115`  
**Previous amendment:** `GPTPRO-GREEN-GATE04-v2-20260805`  
**New amendment authorized by this document:** `GPTPRO-GREEN-GATE08-v2-20260805`  
**Binding verdict:** **B. PREREGISTERED_BASIS_REDESIGN_AND_NEW_RUN**

---

## 1. Executive Ruling

The Gate-08 stop is scientifically real under the frozen four-dimensional contract:

\[
\frac{\sigma_4}{\sigma_5}=1.04<1.10,
\qquad
\frac{\sigma_4}{\sigma_1}=0.5501\gg10^{-4}.
\]

The failure is not rank collapse and not numerical invisibility of the fourth direction. It means that the boundary between the fourth and fifth donor singular directions is not isolated strongly enough to define a unique four-dimensional donor subspace under the preregistered criterion. The stopped run accessed all donor anchors but no development tensor responses, no development path targets, no development decision, and no confirmation data. Confirmation remains locked. 

The implemented basis construction conforms to every scientifically material part of the frozen rank-four protocol:

- exactly 512 basis donors and 512 pair-disjoint radius donors;
- final-position block-10 `resid_mid` clean-minus-corrupt chords;
- an uncentered \(512\times768\) matrix;
- float64 SciPy SVD;
- `full_matrices=False`;
- `lapack_driver="gesvd"`;
- \(U=V_{:,1:4}\);
- deterministic sign canonicalization;
- the exact `1.10` and `1e-4` gates.

There is one implementation-level deficiency: the code attempts to set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` inside the SVD function, after scientific libraries may already have initialized their native thread pools, and it does not control or inspect OpenBLAS or other loaded BLAS runtimes. This does not provide auditable enforcement of the frozen “one BLAS thread” requirement. 

That deficiency does **not** justify verdict A. No concrete mechanism connects it to a roughly six-percentage-point failure of the spectral-ratio gate. A float64 SVD computed with a different reduction order may vary at floating-point roundoff scale, but the observed result is a structural near-tie between two large singular directions. Replaying the same donor matrix merely with stricter thread control would be an unauthorized attempt to obtain a different result from an already observed donor population.

The correct resolution is a one-shot, donor-independent redesign that:

1. fixes the residual rank at **five** before obtaining any new model response;
2. treats the residual mechanism as an operator on a five-dimensional subspace, invariant to arbitrary orthogonal rotations within that subspace;
3. uses a completely new donor noun population and prompt-disjoint fit, holdout, and radius roles;
4. retains the original `1.10` spectral-gap criterion at the new fifth-versus-sixth boundary;
5. adds out-of-sample projector stability and noun-cluster stability audits;
6. preserves every causal intervention, actual gate coordinate, independent target, evaluation cell, statistical threshold, and confirmation rule;
7. terminates the oral line if the one preregistered rank-five design fails.

No exploratory rank search, alternative basis selection, threshold reduction, task-conditioned basis, or fallback to rank six is authorized.

---

## 2. Audit of the Existing Rank-Four Implementation

### 2.1 Conformance table

| Frozen requirement | Implementation at `b87300a` | Audit finding |
|---|---|---|
| 512 basis donors | Sixteen nouns × four centuries × two bins × four pairs | **Conforms** |
| 512 radius donors | Same factorial count, selected after basis pairs | **Conforms** |
| Basis/radius pair disjointness | One `excluded` pair set persists from `basis` to `radius` within each donor cell | **Conforms** |
| Chord site | Cached `blocks.10.hook_resid_mid` | **Conforms** |
| Position | `selected_position(...)` extracts the final prompt position | **Conforms** |
| Chord orientation | `clean_resid - corrupt_resid` | **Conforms** |
| Basis matrix | Stacked basis-role chords | **Conforms** |
| Matrix centering | No subtraction of a row or column mean | **Conforms** |
| Required shape | \(512\times768\) by construction | **Conforms** |
| CPU representation | GPU anchor converted by `.double().cpu().numpy()` | **Conforms** |
| SVD dtype | `np.asarray(..., dtype=np.float64)` | **Conforms** |
| SVD routine | `scipy.linalg.svd` | **Conforms** |
| `full_matrices` | `False` | **Conforms** |
| LAPACK driver | `"gesvd"` | **Conforms** |
| Overwrite/check | `overwrite_a=False`, `check_finite=True` | **Conforms** |
| Basis extraction | `vt[:4].T.copy()` | **Conforms** |
| Sign convention | Largest absolute coordinate made positive; `argmax` gives the first index on an exact tie | **Conforms** |
| Boundary gate | `singular[3] / singular[4] >= 1.10` | **Conforms** |
| Magnitude gate | `singular[3] / singular[0] >= 1e-4` | **Conforms** |
| One BLAS thread | Sets only `OMP_NUM_THREADS` and `MKL_NUM_THREADS` after importing the SVD function; no runtime inspection | **Not auditable; requires correction in the redesign** |

The deterministic donor builder selects four basis and four radius pairs in every noun–century–distance cell and uses the same exclusion set to prevent pair reuse.  The runner extracts clean and corrupt final-position `resid_mid` anchors, constructs `clean_resid - corrupt_resid`, and applies the exact frozen SVD and thresholds. 

### 2.2 The one-thread enforcement deficiency is not a Gate-08 explanation

The current code does this:

```python
from scipy.linalg import svd
old_threads = {
    name: os.environ.get(name)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
}
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
...
```

This is insufficient as an auditable single-thread contract because:

- a BLAS runtime may already be loaded before the environment variables change;
- an installation may use OpenBLAS, BLIS, or another backend rather than MKL;
- no runtime thread-pool state is inspected;
- no repeated-SVD equality audit is performed.

`threadpoolctl` provides runtime introspection and a scoped `threadpool_limits` mechanism for loaded BLAS implementations, so it is the correct enforcement tool for the redesigned run. 

Nevertheless, verdict A would be scientifically invalid. The observed fifth direction is nearly as large as the fourth, while the fourth itself is \(55.01\%\) of the first. The result therefore reflects where the donor covariance spectrum places a cluster boundary, not an underflow, rank-collapse, sign, centering, or indexing bug. The untracked Git bundle also has no causal route to the activation matrix: it was neither imported nor read by the dataset, model, hook, or SVD code. It is a provenance defect that must be prevented in the next run, but it is not a numerical explanation for the spectrum. The server report reaches the same factual conclusion. 

### 2.3 Why verdict A is rejected

`A. CONFORMANCE_FIX_AND_RERUN` is rejected because:

1. the scientific matrix and SVD estimator were implemented correctly;
2. no transposition, centering, role-selection, sign, dtype, position, hook, or threshold error was found;
3. stricter thread enforcement would not convert the observed rank-four object into a scientifically isolated subspace;
4. rerunning the same donor population after observing its spectral failure would amount to retrying a failed gate, not repairing a causal implementation error.

The thread-control defect must be corrected as part of the new protocol, but it cannot be used to erase or reinterpret the stopped Gate-08 result.

---

## 3. Why a Rank-Five Redesign Preserves the Theoretical Contribution

### 3.1 The matched-bypass theorem is rank-generic

The theorem does not fundamentally depend on \(r_1=4\). The current numerical inverse already accepts gate jets with arbitrary upstream rank:

```text
G, C: [k]
J_path, H_path, H_control: [r, k]
```

and applies the inverse independently along every upstream coordinate. 

Let

\[
U\in\mathbb R^{768\times r},
\qquad
U^\top U=I_r,
\]

for any fixed finite \(r\). Define

\[
a_{sj}(x)
=
\left[
\operatorname{LN}_{10,2}
\left(R_s+E_\pi Ux\right)_{\pi,:}
W_{\mathrm{in}}+b_{\mathrm{in}}
\right]_j
\]

and

\[
A_{sji}
=
\left.
\frac{\partial a_{sj}(x)}{\partial x_i}
\right|_{x=0}.
\]

The path and matched-control systems remain

\[
Y^P_{sj}(x,z)
=
F_s\!\left(
Ux+
c_j\left[
\psi(a_{sj}(x)+z)-\psi(a_{sj})
\right]
\right)
\]

and

\[
Y^C_{sj}(x,z)
=
F_s\!\left(
Ux+
c_j\left[
\psi(a_{sj}+z)-\psi(a_{sj})
\right]
\right).
\]

For every \(i\in\{1,\ldots,r\}\), exactly the same chain-rule proof gives

\[
H^P_{sij}-H^C_{sij}=C_{sj}A_{sji}.
\]

When \(C_{sj}\neq0\),

\[
A_{sji}
=
\frac{
\left\langle
C_{sj},
H^P_{sij}-H^C_{sij}
\right\rangle
}{
\|C_{sj}\|_2^2
},
\]

\[
P_{s,:,i,j}=G_{sj}A_{sji},
\]

and

\[
D_{s,:,i}
=
J^P_{sji}-P_{s,:,i,j}.
\]

No step of the proof sums over exactly four coordinates or invokes a four-dimensional property. The hardcoded four appears only in the chosen empirical basis, tensor shapes, probe loops, and compute accounting. The causal topology and matched-bypass factorization are unchanged. The original bridge document defines the same identities and uses rank four only as the frozen instantiation. 

### 3.2 Projector-covariant structural object

The redesigned claim must not assign mechanistic meaning to arbitrary PCA axes inside the five-dimensional subspace. It must identify an operator on the subspace.

Let

\[
\mathcal U=\operatorname{col}(U),
\qquad
\Pi=UU^\top.
\]

For gate \(j\), define the physical path operator

\[
\mathcal P_{sj}:\mathcal U\rightarrow\mathbb R^{100}
\]

by

\[
\mathcal P_{sj}(v)
=
DF_s(0)[c_j]\,
\psi'(a_{sj})\,
D a_{sj}(0)[v].
\]

In basis \(U\), its matrix is

\[
P_{sj}
=
G_{sj}A_{sj:}
\in\mathbb R^{100\times r}.
\]

For any orthogonal matrix \(Q\in O(r)\), set

\[
\widetilde U=UQ.
\]

The corresponding coordinate quantities transform as

\[
\widetilde A_{sj:}=A_{sj:}Q,
\]

\[
\widetilde P_{sj}=P_{sj}Q,
\]

\[
\widetilde D_s=D_sQ.
\]

For a physical perturbation with old coordinate \(\delta\), the new coordinate is

\[
\widetilde\delta=Q^\top\delta,
\]

so

\[
\widetilde U\widetilde\delta
=
UQQ^\top\delta
=
U\delta
\]

and

\[
\widetilde P_{sj}\widetilde\delta
=
P_{sj}QQ^\top\delta
=
P_{sj}\delta.
\]

Therefore the identified structural object is the equivalence class

\[
[U,A,P,D]
=
\left\{
(UQ,AQ,PQ,DQ):Q\in O(r)
\right\}.
\]

The independent finite-radius target is also projector-invariant. For a clean–corrupt chord \(d_n\), define the physical direction directly as

\[
v_n
=
h_1
\frac{\Pi d_n}{\|\Pi d_n\|_2}.
\]

Its coordinate representation is

\[
\delta_n=U^\top v_n.
\]

Replacing \(U\) by \(UQ\) leaves \(v_n\), the actual `resid_mid` intervention, the bypass subtraction, the gate response, and the tensor contraction unchanged.

This formulation strengthens the theoretical object. It separates:

- the scientifically meaningful five-dimensional projector \(\Pi\);
- the arbitrary orthonormal coordinates used to probe it;
- the actual, unrotated MLP-10 gate coordinates.

No rotation of GELU gates is introduced.

### 3.3 Why rank five is the only authorized new rank

Rank five is authorized as the **minimal cluster-completion rank**.

The observed donor spectrum establishes two facts:

1. rank four cuts through a non-isolated fourth/fifth pair;
2. the fourth direction is not small.

The least adaptive response is therefore to include the entire observed fourth/fifth boundary cluster. This does not assert that rank five will pass. It creates a new, falsifiable condition:

\[
\frac{\sigma_5}{\sigma_6}\ge1.10
\]

on completely fresh donors, with independent holdout and stability audits.

The following are prohibited:

- testing ranks \(4,5,6,\ldots\) and selecting the best;
- falling back to rank six if rank five fails;
- selecting rank from a scree plot in the new run;
- lowering `1.10`;
- replacing the rank-five projector with a supervised or target-conditioned direction;
- using development or confirmation responses to choose the rank.

If the new rank-five boundary is not isolated and stable, the bridge oral line terminates.

---

## 4. Frozen Rank-Five Structural and Probe Design

### 4.1 Dimensions

The redesigned fixed dimensions are

\[
r_1=5,
\qquad
r_2=10,
\qquad
k=100.
\]

The tensors become:

\[
A_s\in\mathbb R^{10\times5},
\]

\[
P_s\in\mathbb R^{100\times5\times10},
\]

\[
D_s\in\mathbb R^{100\times5}.
\]

The ten actual MLP-10 gate indices remain exactly:

```text
2326
1138
2287
606
2848
2305
46
2659
946
1616
```

### 4.2 Probe completeness

The upstream design is

\[
\mathcal X_5
=
\{e_1,e_2,e_3,e_4,e_5\}.
\]

The gate design remains

\[
\mathcal Z
=
\{e_j:j\in J\}.
\]

The paired design is

\[
\mathcal K_5
=
\{e_i\otimes e_j:
i=1,\ldots,5,\ j\in J\}.
\]

Its design matrix is a permutation of

\[
I_{50}.
\]

Therefore

\[
\operatorname{rank}(\mathcal X_5)=5,
\qquad
\operatorname{rank}(\mathcal Z)=10,
\qquad
\operatorname{rank}(\mathcal K_5)=50.
\]

Every selected path-tensor coordinate is directly probed. No regression, ridge penalty, pseudo-inverse, or learned probe direction is introduced.

### 4.3 Finite-difference calls

For one system, one gate, and one radius:

- two gate-axis endpoints;
- \(2r=10\) residual-axis endpoints;
- \(4r=20\) path mixed corners;
- \(4r=20\) matched-control mixed corners.

Thus

\[
2+10+20+20=52
\]

evaluations.

For ten gates and two radii:

\[
52\times10\times2=1040.
\]

With one shared center:

\[
1041
\]

evaluations per system and

\[
2082
\]

for target and patched systems together.

The existing implementation hardcodes 42-condition rank-four jets, four-dimensional zero arrays, 200 first-order directions, and the old `1682` budget; every such hardcoding must be removed. 

### 4.4 Equal-budget first-order baseline

The first-order baseline must remain exactly budget matched.

Construct 250 deterministic residual directions:

- directions 1–5 are \(e_1,\ldots,e_5\);
- directions 6–250 are 245 normalized standard-normal vectors in \(\mathbb R^5\);
- RNG is NumPy `PCG64`;
- seed is the first eight big-endian bytes of

```text
SHA256("idle1-gt-bridge-basis-v2-20260805:first-order-r5")
```

- the first nonzero coordinate is made positive;
- reject a proposal if its absolute inner product with any accepted vector exceeds `0.999999`.

The exact call count is

\[
2\ \text{systems}
\times
2\ \text{radii}
\times
2\ \text{signs}
\times
(250+10)
+
2\ \text{centers}
=
2082.
\]

The factorial and single-direction shared endpoint cache remains 16 evaluations per tensor item.

---

## 5. Completely Fresh Donor Population

### 5.1 Old donor responses

Every donor prompt evaluated in the stopped rank-four run is permanently restricted to:

- diagnosis of the failed rank-four basis;
- provenance reporting;
- preservation of the Gate-08 stop.

Those donor anchors must not be used in:

- the new basis fit;
- the new basis holdout;
- the new radius estimator;
- leave-one-noun analyses;
- bootstrap stability;
- manual-tail selection;
- throughput selection;
- development or confirmation computation.

The old donor response files must not be loaded by the redesigned runner.

### 5.2 New donor nouns

Use exactly this ordered noun tuple:

```python
BASIS_V2_DONOR_NOUNS = (
    "rebellion",
    "revolution",
    "occupation",
    "blockade",
    "crusade",
    "migration",
    "settlement",
    "construction",
    "administration",
    "regime",
    "competition",
    "partnership",
    "transition",
    "expansion",
    "uprising",
    "conflict",
)
```

These nouns are disjoint from:

- the eight evaluation nouns;
- the sixteen protocol-v1 donor nouns.

The donor centuries remain:

```python
BASIS_V2_DONOR_CENTURIES = (11, 13, 15, 17)
```

The distance bins remain:

```python
near = [8, 16]
far  = [40, 56]
```

The suffix range remains:

```text
05 through 94 inclusive
```

### 5.3 Roles and counts

Use three completely disjoint donor roles:

| Role | Pairs per noun–century–bin | Orientation quota | Total pairs |
|---|---:|---|---:|
| `basis_fit` | 4 | 2 up, 2 down | 512 |
| `basis_holdout` | 2 | 1 up, 1 down | 256 |
| `radius_v2` | 4 | 2 up, 2 down | 512 |
| **Total** | 10 | — | **1,280** |

For each noun and century, pair selection must follow this exact role/bin order:

```python
DONOR_V2_SELECTION_ORDER = (
    ("near", "basis_fit",     2, 2),
    ("far",  "basis_fit",     2, 2),
    ("near", "basis_holdout", 1, 1),
    ("far",  "basis_holdout", 1, 1),
    ("near", "radius_v2",     2, 2),
    ("far",  "radius_v2",     2, 2),
)
```

### 5.4 Prompt-level disjointness

Maintain one set

```python
used_suffixes: set[int]
```

for each `(noun, century)` across both distance bins and all three roles.

A candidate unordered pair \((a,b)\) is eligible only if:

```python
a not in used_suffixes
and b not in used_suffixes
```

After acceptance, add both suffixes.

Consequently, no clean or corrupt donor prompt is reused:

- between fit and holdout;
- between fit and radius;
- between holdout and radius;
- between near and far roles.

Each noun–century combination uses exactly 40 distinct suffixes:

\[
20\ \text{pairs}\times2=40.
\]

A quota failure is a pre-model-response technical stop. No noun, pair, or quota may be replaced.

### 5.5 Hashing and orientation

Use the salt:

```text
idle1-gt-bridge-basis-v2-20260805
```

For unordered pair \((a,b)\), \(a<b\), define:

```text
{salt}|pair|{noun}|{cc:02d}|{bin}|{role}|{a:02d}|{b:02d}
```

For orientation:

```text
{salt}|orient|{noun}|{cc:02d}|{bin}|{role}|{a:02d}|{b:02d}
```

Rank candidate pairs by ascending hexadecimal pair digest.

The preferred orientation is:

```python
"up" if int(orientation_digest[:2], 16) & 1 else "down"
```

Use the preferred orientation when its quota remains; otherwise use the opposite orientation when its quota remains; otherwise continue.

Store and hash:

```text
legacy_donor_plan_sha256
basis_v2_full_plan_sha256
basis_fit_ordered_keys_sha256
basis_holdout_ordered_keys_sha256
radius_v2_ordered_keys_sha256
basis_v2_all_prompt_keys_sha256
```

Also store exact ordered pair and prompt keys in `splits.json`.

### 5.6 Evaluation population remains immutable

The existing 48 evaluation cells, tensor/energy roles, development/confirmation assignment, pair salt, pair digests, prompts, and ordered split hash must remain byte-for-byte identical to protocol v1.1. The current evaluation builder and confirmation lock already keep development and confirmation records separate until the frozen analysis opens confirmation. 

A contract test must compare the complete serialized evaluation plan generated at `b87300a` with the redesigned plan. Any difference is a stop.

---

## 6. Exact Rank-Five Basis Estimator

### 6.1 Fit and holdout matrices

For every donor pair, cache the final-position block-10 `resid_mid` chord

\[
d_n
=
R_{\mathrm{clean},n}[\pi,:]
-
R_{\mathrm{corrupt},n}[\pi,:].
\]

Construct:

\[
D_F\in\mathbb R^{512\times768}
\]

from `basis_fit` and

\[
D_H\in\mathbb R^{256\times768}
\]

from `basis_holdout`.

Both matrices are:

- uncentered;
- ordered by the frozen donor-plan order;
- stored in float64 on CPU;
- finite;
- hashed before SVD.

The radius matrix

\[
D_R\in\mathbb R^{512\times768}
\]

is separate and must not participate in basis construction or basis admissibility.

### 6.2 Exact thread control

Add:

```text
threadpoolctl==3.6.0
```

to `requirements-green-bridge.lock`. The current lock contains only the eight existing package lines and does not pin a thread-pool controller. 

At process launch, require:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Inside the basis function, use:

```python
from threadpoolctl import threadpool_info, threadpool_limits
```

and:

```python
with threadpool_limits(limits=1, user_api="blas"):
    pools = [
        row for row in threadpool_info()
        if row.get("user_api") == "blas"
    ]
    if not pools:
        raise GreenStop(
            "08A_BASIS_THREAD_CONTRACT",
            "no loaded BLAS runtime was introspectable",
        )
    if any(int(row.get("num_threads", -1)) != 1 for row in pools):
        raise GreenStop(
            "08A_BASIS_THREAD_CONTRACT",
            canonical_json(pools),
        )
    ...
```

Serialize the full `threadpool_info()` output.

### 6.3 Canonical SVD

Implement exactly:

```python
def canonical_rank_basis(
    chords: np.ndarray,
    *,
    rank: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from scipy.linalg import svd
    from threadpoolctl import threadpool_info, threadpool_limits

    matrix = np.asarray(chords, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 768:
        raise ValueError(f"invalid chord matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("chord matrix contains NaN or infinity")

    def one_svd():
        _, singular, vt = svd(
            matrix,
            full_matrices=False,
            lapack_driver="gesvd",
            overwrite_a=False,
            check_finite=True,
        )
        basis = vt[:rank].T.copy()
        for column in range(rank):
            pivot = int(np.argmax(np.abs(basis[:, column])))
            if basis[pivot, column] < 0:
                basis[:, column] *= -1.0
        return basis, singular

    with threadpool_limits(limits=1, user_api="blas"):
        pools = [
            row for row in threadpool_info()
            if row.get("user_api") == "blas"
        ]
        if not pools or any(
            int(row.get("num_threads", -1)) != 1
            for row in pools
        ):
            raise RuntimeError("single-thread BLAS contract failed")

        basis_1, singular_1 = one_svd()
        basis_2, singular_2 = one_svd()

    if not np.array_equal(singular_1, singular_2):
        raise RuntimeError("repeated singular values were not bitwise equal")
    if not np.array_equal(basis_1, basis_2):
        raise RuntimeError("repeated canonical bases were not bitwise equal")

    orthogonal_error = np.max(
        np.abs(basis_1.T @ basis_1 - np.eye(rank))
    )
    if orthogonal_error > 5e-13:
        raise RuntimeError(
            f"basis orthogonality error {orthogonal_error}"
        )

    return basis_1, singular_1, {
        "threadpools": pools,
        "orthogonal_max_abs": float(orthogonal_error),
    }
```

No randomized SVD, covariance eigendecomposition, centering, whitening, shrinkage, or task weighting is allowed.

### 6.4 Projector

Define

\[
U_F=V_F[:,1:5]
\]

and

\[
\Pi_F=U_FU_F^\top.
\]

Store both `U` and `projector`. Scientific comparisons between bases must use principal angles or projectors, not columnwise vector matching.

---

## 7. Binding Basis Admissibility Gates

All gates below must pass. They are conjunctive.

### 7.1 Fit spectrum

Require:

\[
\boxed{
\frac{\sigma_{F,5}}{\sigma_{F,6}}\ge1.10
}
\]

and

\[
\boxed{
\frac{\sigma_{F,5}}{\sigma_{F,1}}\ge10^{-4}.
}
\]

Terminal identifier:

```text
08B_BASIS_FIT_SPECTRUM
```

### 7.2 Holdout spectrum

Compute an entirely separate SVD of \(D_H\) and require:

\[
\boxed{
\frac{\sigma_{H,5}}{\sigma_{H,6}}\ge1.10
}
\]

and

\[
\boxed{
\frac{\sigma_{H,5}}{\sigma_{H,1}}\ge10^{-4}.
}
\]

Terminal identifier:

```text
08C_BASIS_HOLDOUT_SPECTRUM
```

### 7.3 Fit–holdout principal angle

Let \(U_H\) be the top-five holdout basis. Define

\[
s_{\min}
=
\sigma_{\min}(U_F^\top U_H)
\]

and

\[
\theta_{\max}
=
\arccos\left(
\operatorname{clip}(s_{\min},-1,1)
\right).
\]

Require:

\[
\boxed{
\theta_{\max}\le15^\circ.
}
\]

Terminal identifier:

```text
08D_BASIS_FIT_HOLDOUT_ANGLE
```

### 7.4 Holdout energy efficiency

The optimal rank-five captured energy on the holdout matrix is

\[
E_H^\star
=
\sum_{i=1}^{5}\sigma_{H,i}^2.
\]

The energy captured by the fit projector is

\[
E_H(U_F)
=
\|D_HU_F\|_F^2.
\]

Require:

\[
\boxed{
\frac{E_H(U_F)}{E_H^\star}\ge0.90.
}
\]

This threshold is scale-free. It requires the fit projector to retain at least \(90\%\) of the energy captured by the holdout-optimal rank-five subspace. It was not selected from the stopped value `1.04`.

Terminal identifier:

```text
08E_BASIS_HOLDOUT_ENERGY
```

### 7.5 Leave-one-noun stability

For every one of the sixteen new donor nouns:

1. remove all 32 `basis_fit` rows belonging to that noun;
2. recompute the rank-five basis with the exact estimator;
3. require
   \[
   \sigma_{-n,5}/\sigma_{-n,1}\ge10^{-4};
   \]
4. compute the largest principal angle to \(U_F\);
5. require
   \[
   \boxed{
   \theta_{\max}(U_F,U_{-n})\le15^\circ.
   }
   \]

All sixteen omissions must pass.

Terminal identifier:

```text
08F_BASIS_LEAVE_ONE_NOUN
```

### 7.6 Noun-cluster bootstrap stability

Use exactly 256 bootstrap replicates.

The seed is:

```python
int.from_bytes(
    hashlib.sha256(
        b"idle1-gt-bridge-basis-v2-20260805:noun-bootstrap"
    ).digest()[:8],
    "big",
)
```

Use NumPy `PCG64`.

For each replicate:

1. sample 16 noun indices with replacement from `0..15`;
2. include all 32 fit rows for every sampled noun occurrence;
3. compute the rank-five basis using the exact SVD estimator;
4. if
   \[
   \sigma_{b,5}/\sigma_{b,1}<10^{-4},
   \]
   set the angle to \(90^\circ\);
5. otherwise compute the largest principal angle to \(U_F\).

Compute:

```python
np.quantile(
    bootstrap_angles,
    0.95,
    method="higher",
)
```

and require:

\[
\boxed{
Q_{0.95}^{\mathrm{higher}}(\theta_{\max})\le15^\circ.
}
\]

Store:

- the \(256\times16\) sampled noun-index matrix;
- its SHA-256;
- all 256 angles;
- the exact quantile;
- the number of floor failures.

Terminal identifier:

```text
08G_BASIS_NOUN_BOOTSTRAP
```

### 7.7 Why these thresholds are principled

The redesign does **not** change `1.10` to accommodate `1.04`. It moves the same boundary criterion to the only newly authorized boundary, five versus six.

The `15°` limit is inherited from the original leave-one-noun contract. In projector terms,

\[
\|\Pi_1-\Pi_2\|_2
=
\sin(\theta_{\max}),
\]

so \(15^\circ\) bounds the worst physical subspace discrepancy by approximately \(0.259\).

The `0.90` holdout-efficiency gate is dimensionless and compares the fit projector with the best possible rank-five holdout projector. It does not depend on the absolute scale or the stopped spectrum.

No result from the old donor matrix is used to tune any of these thresholds.

---

## 8. Radius Construction Under Rank Five

Use only the 512 `radius_v2` pairs.

### 8.1 Residual radius

For radius chord \(d_n\), define

\[
\sigma_x
=
\operatorname{median}_n
\frac{\|U_F^\top d_n\|_2}{\sqrt5}.
\]

Set

\[
h_1=0.20\,\sigma_x.
\]

The full and half radii remain:

\[
h_1,\qquad h_1/2.
\]

The residual floor remains:

\[
h_1
\ge
2^{-10}
\operatorname{median}_{n\in\mathrm{radius\_v2}}
\operatorname{RMS}
\left(
R_n[\pi,:]
\right).
\]

Only radius-role anchors enter this floor.

### 8.2 Gate radii

For gate \(j\), using only `radius_v2` clean and corrupt preactivations:

\[
\sigma_j
=
\max\left\{
1.4826\,\operatorname{MAD}(a_j),
\operatorname{median}_n
|a^{\mathrm{clean}}_{nj}-a^{\mathrm{corrupt}}_{nj}|
\right\},
\]

\[
h_{2j}=0.20\,\sigma_j.
\]

All existing gate-radius floors remain unchanged.

### 8.3 Radius stability

For each of the sixteen new donor nouns:

- omit all `radius_v2` pairs for that noun;
- recompute \(h_1\) and all ten \(h_{2j}\);
- require every relative change to be at most \(20\%\).

No radius search or inflation is authorized.

### 8.4 Evaluation direction

For every tensor or energy item:

\[
d_n
=
R_{\mathrm{tar},n}[\pi,:]
-
R_{\mathrm{cor},n}[\pi,:],
\]

\[
q_n=U_F^\top d_n.
\]

Retain the existing validity gate:

\[
\|q_n\|_2\ge0.10\,\sigma_x.
\]

Define:

\[
\delta_n
=
h_1\frac{q_n}{\|q_n\|_2}.
\]

Equivalently, the physical intervention is

\[
U_F\delta_n
=
h_1
\frac{\Pi_Fd_n}{\|\Pi_Fd_n\|_2}.
\]

This is invariant to rotations inside the fitted subspace.

---

## 9. Causal Experiment and Independent Target Remain Unchanged

The rank redesign changes only the residual subspace and its coordinate count.

The following remain exact:

| Component | Binding status |
|---|---|
| Model and revision | unchanged |
| TransformerLens version and source hashes | unchanged |
| MLP-8 clean-to-corrupt comparator patch | unchanged |
| Upstream intervention site | `blocks.10.hook_resid_mid` |
| Gate preactivation site | `blocks.10.mlp.hook_pre` |
| Gate postactivation site | `blocks.10.mlp.hook_post` |
| Actual selected gate indices | unchanged |
| One-gate path mode | unchanged |
| Matched-bypass control | unchanged |
| Omitted-gate anchoring | unchanged |
| Direct residual bypass in path/control | preserved identically |
| Independent target module | unchanged in causal logic |
| Target bypass subtraction | `blocks.10.hook_resid_post` |
| Block-11 continuation | unchanged |
| Year-logit output | unchanged |
| Greater-Than logit contrast | unchanged |
| Tensor/energy prompt disjointness | unchanged |
| Development/confirmation split | unchanged |
| Confirmation lock | unchanged |

The target implementation currently lets the ten declared gates respond jointly, anchors the other gates, forms the block-10 MLP output, and subtracts the injected residual before block 11. Only its hardcoded input shape `[batch,4]` must become `[batch,5]`; the causal operation itself must not change. 

The manual tail similarly preserves the exact path/control/joint modes. Only its rank-four shape assertions must become rank-parameterized and then frozen to rank five. 

---

## 10. Uncertainty Propagation and Statistical Protocol

### 10.1 Numerical propagation

The Richardson formulas remain exactly:

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

For rank five, the axiswise quantities simply have length five:

\[
\epsilon_{\Delta H}\in\mathbb R^5,
\qquad
A_{\max}\in\mathbb R^5,
\qquad
\epsilon_A\in\mathbb R^5,
\qquad
\epsilon_P\in\mathbb R^5.
\]

The Frobenius bound becomes

\[
\epsilon_{P,F}
=
\left(
\sum_{i=1}^{5}\epsilon_{P,i}^2
\right)^{1/2}.
\]

The existing numerical module already derives the residual rank from the first axis of the mixed-response array rather than assuming four, so its equations require shape tests but no mathematical change. 

### 10.2 Same-TransformerLens numerical noise

As in the Gate-04 decision:

\[
\epsilon_y
=
\max\left\{
10^{-7},
\text{same-TransformerLens duplicate error}
\right\}.
\]

HF–TL portability differences remain reporting-only and do not enter finite-difference uncertainty.

### 10.3 Gate admissibility

All existing thresholds remain unchanged:

- curvature RMS and SNR;
- gate-response RMS and SNR;
- factorization residual;
- white-box \(A\) agreement;
- full/half tensor cosine;
- symmetric relative change;
- Richardson correction;
- tensor SNR;
- bypass consistency;
- active-gate count;
- certified-null bounds.

Only the residual-axis dimension changes from four to five.

### 10.4 Cell and confirmation rules

All existing statistical rules remain unchanged:

- at least six admissible tensor items per cell;
- at least six admissible energy items per cell;
- development survival;
- target conditioning;
- development SNR;
- development gain required to open confirmation;
- nonnegative affine calibration for baselines only;
- no calibration for the mixed predictor;
- 100,000 paired confirmation bootstrap replicates;
- overall and per-bin RMSE thresholds;
- cancellation-subset AUROC;
- half-radius rank and magnitude stability;
- all survival floors.

The current source freezes these thresholds in `green_bridge_spec.py`; none may change in the rank-five amendment. 

### 10.5 Target-basis stability

Retain the existing development-only target-basis audit, now using the sixteen rank-five leave-one-noun bases.

For the first hash-ranked energy item in every development cell and both target and patched systems, compare each leave-one-noun basis with the full basis.

Every omission must satisfy:

\[
\operatorname{Spearman}\ge0.90
\]

and median symmetric relative change at most \(20\%\), with denominator floor `0.05`.

This occurs only after all donor-only basis gates pass and before development tensor scoring.

---

## 11. Exact Forward-Pass Budget

### 11.1 Tensor estimators

For each of 384 tensor items:

| Component | Evaluations |
|---|---:|
| Rank-five mixed estimator | 2,082 |
| Equal-budget first-order baseline | 2,082 |
| Shared factorial/PIE/single-direction cache | 16 |
| **Total** | **4,180** |

Therefore:

\[
384\times4180
=
1,605,120
\]

tail evaluations.

### 11.2 Other tail computation

| Component | Tail evaluations |
|---|---:|
| Tensor items | 1,605,120 |
| Energy targets: 384 items × 3 systems × 4 endpoints | 4,608 |
| Sixteen leave-one-basis target audits | 1,024 |
| Manual-tail and duplicate audit overhead | 96 |
| **Total** | **1,610,848** |

### 11.3 JVP and full-model computation

| Component | Count |
|---|---:|
| Target JVP invocations | 1,152 |
| New donor full-model prompts | 2,560 |
| Tensor-item anchors | 1,152 |
| Energy-item anchors | 1,152 |
| Full-model audit overhead | 192 |
| **Full-model total** | **5,056** |

Raw invocations:

\[
1,610,848+1,152+5,056
=
1,617,056.
\]

JVP-at-two-tail-equivalents total:

\[
1,610,848+2(1,152)+5,056
=
1,618,208.
\]

Conservative total with full-model evaluations costed at six tail units:

\[
1,610,848+2(1,152)+6(5,056)
=
1,643,488.
\]

### 11.4 Phase accounting

| Phase | Effective units |
|---|---:|
| Clean preflight, donors, prepare, and development | 541,920 |
| Locked confirmation | 1,076,288 |
| **Total** | **1,618,208** |

### 11.5 Hardware cap

For an RTX 4090:

- target manual-tail batch: 512;
- target full-model batch: 64;
- target JVP batch: 64;
- peak allocated memory: at most 20 GB;
- planning range: 10–20 GPU hours;
- hard cap: 24 GPU hours.

For an A40-class 48 GB GPU:

- target manual-tail batch: 1024;
- target full-model batch: 128;
- target JVP batch: 128;
- peak allocated memory: at most 32 GB;
- planning range: 17–35 GPU hours;
- hard cap: 40 GPU hours.

The deterministic 2% operation-mixture throughput preflight remains required, and all outputs from it must be reused.

A throughput failure stops the run. It does not authorize changing rank, donors, radii, gates, target, or hardware precision.

---

## 12. Gate-04 Treatment

### 12.1 Prior Gate-04 evidence

The passed Gate-04 v1.1 artifact must be preserved permanently. It used:

- eager Hugging Face attention;
- batch size one;
- 32 prompts;
- exact converted-weight equality;
- the disjoint legacy donor panel at pair ranks `16:32`;
- ordered prompt-key hash

```text
619d21c10d4f30e6ce2597c3ba4df1de72cf0cb4f6cce322d82c2d3ec62803ce
```

and passed every amended portability threshold. Its hook-audit SHA-256 is:

```text
49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99
``` 


### 12.2 Binding action

The new clean run must **replay**, not replace, the exact Gate-04 v1.1 panel.

Implement a separate:

```python
build_legacy_donor_records()
```

using:

- the original sixteen donor nouns;
- original donor centuries;
- original salt;
- original `basis` then `radius` role ordering;
- original pair-selection semantics.

The Gate-04 selector must again take legacy sorted pair-digest ranks:

```python
legacy[0:16]
holdout[16:32]
```

and must assert the exact ordered prompt-key hash above.

The same Gate-04 thresholds remain unchanged.

A new disjoint Gate-04 panel is not needed because Gate 04 validates implementation portability, not donor-basis identification. Replaying it in the clean worktree verifies that the new execution environment still matches the already authorized portability contract.

### 12.3 Separation of numerical concepts

The redesigned manifest and result reports must keep four concepts separate:

1. **Portability validation:** Hugging Face versus TransformerLens Gate 04.
2. **Donor-basis identification:** fresh rank-five fit, holdout, and stability gates.
3. **Numerical repeatability:** same-TransformerLens duplicate and manual-tail audits.
4. **Scientific prediction:** development and confirmation tensor/target comparisons.

No metric from one category may be substituted into another.

---

## 13. Clean-Worktree Contract and Stopped-Run Provenance

### 13.1 Current stopped run

The current Gate-08 stop must be archived without alteration. Preserve these recorded hashes:

```text
7d52411b487f7e85f0dc539c760541d16bf5c9b756da75490edd8b9ad5ad7f90  result.json
baff192581726f4cae8f23418df5600ccb0fff549b0c81edff8c2c1f95d914df  manifest.json
49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99  hook_audit.json
845cb7746be048dacbcb6c841e45d29e3d51d7e7632074e08b63c92dea5d8fb8  green_bridge_prepare_gate04_v2.log
```

The archive metadata must explicitly retain:

```yaml
repository_dirty_at_launch: true
dirty_reason: untracked offline Git transport bundle
tracked_changes_at_launch: false
development_responses_observed: false
confirmation_responses_observed: false
first_failed_gate: 08_BASIS_SPECTRUM
```

### 13.2 New hard gate

Before creating the new output root or initial manifest, run:

```python
def assert_clean_repository() -> dict:
    branch = git_text("branch", "--show-current")
    commit = git_text("rev-parse", "HEAD")
    status = git_text(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    if branch != "main":
        raise GreenStop(
            "00_REPOSITORY_CLEAN",
            f"branch={branch!r}, expected 'main'",
        )
    if status != "":
        raise GreenStop(
            "00_REPOSITORY_CLEAN",
            status,
        )

    subprocess.check_call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            "b87300a6f56cb4706db090486d8bec77a2fc2b23",
            commit,
        ],
        cwd=PROJECT_ROOT,
    )

    return {
        "branch": branch,
        "commit": commit,
        "status_porcelain": status,
        "clean": True,
        "review_commit_is_ancestor": True,
    }
```

This function must execute before:

```python
output_root.mkdir(...)
```

and before any model or tokenizer is loaded.

The initial manifest must contain:

```json
"repository_dirty_at_launch": false
```

as a hard assertion, not merely an observation.

### 13.3 Empty output-root gate

For phase `prepare`, the output root must not exist or must be an empty directory.

Any existing file triggers:

```text
00_OUTPUT_ROOT_NOT_EMPTY
```

This prevents accidental reuse of rank-four donor artifacts or a second rank-five attempt.

---

## 14. Exact Source Changes

### 14.1 `analysis/GPTPRO_GREEN_GATE08_DECISION_20260805.md`

Add this document verbatim.

Do not edit the historical bridge, Gate-04 decision, Gate-04 stop, or Gate-08 stop reports.

### 14.2 `src/green_bridge_spec.py`

Make these exact conceptual changes:

```python
SCHEMA_VERSION = "green-bridge-v1.2"
GATE08_AMENDMENT_ID = "GPTPRO-GREEN-GATE08-v2-20260805"

LEGACY_SALT = "idle1-gt-bridge-20260805"
BASIS_V2_SALT = "idle1-gt-bridge-basis-v2-20260805"

LEGACY_DONOR_NOUNS = (...)
BASIS_V2_DONOR_NOUNS = (...)

BASIS_V2_DONOR_CENTURIES = (11, 13, 15, 17)

BASIS_V2_FIT_PAIRS = 512
BASIS_V2_HOLDOUT_PAIRS = 256
BASIS_V2_RADIUS_PAIRS = 512

BASIS_V2_BOOTSTRAP_REPLICATES = 256
BASIS_V2_BOOTSTRAP_QUANTILE = 0.95

FIRST_ORDER_RESIDUAL_DIRECTIONS = 250
```

Change:

```python
residual_rank: int = 4
```

to:

```python
residual_rank: int = 5
```

Add exact basis thresholds:

```python
basis_fit_gap_min: float = 1.10
basis_holdout_gap_min: float = 1.10
basis_rank_floor: float = 1e-4
basis_angle_max_degrees: float = 15.0
basis_holdout_efficiency_min: float = 0.90
basis_bootstrap_q95_max_degrees: float = 15.0
```

Do not alter any causal, numerical, development, or confirmation threshold.

### 14.3 `src/green_bridge_dataset.py`

Leave `build_evaluation_records()` semantically unchanged.

Retain or add:

```python
build_legacy_donor_records()
```

for Gate-04 replay.

Add:

```python
build_basis_v2_donor_records()
```

implementing:

- the new nouns and salt;
- the exact selection order;
- prompt-level suffix disjointness across roles and bins;
- exact quotas;
- deterministic hashes;
- quota failure before model loading.

Add validators for:

- 512/256/512 role counts;
- 1,280 total pairs;
- 2,560 unique donor prompt keys;
- no donor-v2/evaluation noun overlap;
- no donor-v2/legacy noun overlap;
- no suffix reuse within a noun–century;
- exact orientation quotas;
- exact role order.

### 14.4 New file: `src/green_bridge_basis.py`

Move all basis and radius CPU analytics into this file.

It must contain only:

- NumPy;
- SciPy;
- `threadpoolctl`;
- dataclasses;
- hashing and serialization utilities.

Required functions:

```python
canonical_rank_basis(...)
principal_angle_degrees(...)
holdout_efficiency(...)
leave_one_noun_audit(...)
noun_cluster_bootstrap(...)
fit_rank5_basis(...)
construct_rank5_radii(...)
```

It must not import:

- Torch;
- TransformerLens;
- target code;
- matched-bypass inverse code;
- development or confirmation analysis.

### 14.5 `src/matched_bypass_gate.py`

The inverse remains unchanged.

Replace:

```python
def expected_tensor_calls(
    n_gates=10,
    n_radii=2,
    n_systems=2,
):
    per_gate_radius_system = 2 + 8 + 16 + 16
```

with:

```python
def expected_tensor_calls(
    residual_rank: int,
    n_gates: int = 10,
    n_radii: int = 2,
    n_systems: int = 2,
) -> int:
    if residual_rank <= 0:
        raise ValueError("residual_rank must be positive")
    per_gate_radius_system = 2 + 10 * residual_rank
    return n_systems * (
        n_gates * n_radii * per_gate_radius_system + 1
    )
```

Require:

```python
expected_tensor_calls(5) == 2082
```

### 14.6 `src/green_bridge_tail.py`

Replace all hardcoded rank-four shape checks with:

```python
rank = residual_basis.shape[1]
if tuple(residual_basis.shape) != (
    model.cfg.d_model,
    DIMENSIONS.residual_rank,
):
    ...
if tuple(x.shape) != (batch, DIMENSIONS.residual_rank):
    ...
```

Require:

```python
DIMENSIONS.residual_rank == 5
```

No path, control, clamping, bypass, block-10, or continuation code may change.

### 14.7 `src/green_bridge_path_target.py`

Replace the hardcoded:

```python
if tuple(x.shape) != (batch, 4):
```

with the frozen rank-five assertion.

Do not change:

- selected-gate joint response;
- omitted-gate anchoring;
- block-10 computation;
- residual-bypass subtraction;
- block-11 continuation;
- final logit computation;
- import firewall.

### 14.8 `src/green_bridge_numerics.py`

No equation changes.

Add shape assertions establishing:

```python
rich.H_path.shape == (5, 100)
rich.H_control.shape == (5, 100)
rich.J_path.shape == (5, 100)
```

Retain dynamic axiswise propagation and Frobenius aggregation.

### 14.9 `src/exp_green_bridge_gpt2.py`

Required changes:

1. call `assert_clean_repository()` before output creation;
2. reject a nonempty prepare output root;
3. reconstruct legacy donors only for Gate-04 replay;
4. build the new 1,280-pair donor-v2 plan separately;
5. validate and hash all donor roles before model loading;
6. collect donor-v2 anchors only;
7. delegate basis/radius analytics to `green_bridge_basis.py`;
8. make `_jet_at_radius` rank-generic;
9. for rank five, batch 32 path conditions and 20 control conditions;
10. replace all `np.zeros(4)` and `(batch,4)` constructions;
11. replace all `range(4)` loops;
12. change duplicate-noise axis cycling to modulo five;
13. construct 250 first-order residual directions;
14. update equal-budget baseline arrays and counts;
15. update `sigma_x` normalization to `sqrt(5)`;
16. update `FORWARD_COUNTS`;
17. serialize all new basis audits;
18. prevent loading old donor anchors;
19. preserve the confirmation lock and existing phase ordering;
20. remove or disable `--phase all`; only separate phases are authorized.

### 14.10 `src/analyze_green_bridge.py`

No statistical estimator or threshold changes are authorized.

Only schema/count plumbing required to accept rank-five artifacts may change.

### 14.11 `src/test_green_bridge_contract.py`

Add the tests listed in Section 15.

### 14.12 `requirements-green-bridge.lock`

Append exactly:

```text
threadpoolctl==3.6.0
```

No other package version may change.

---

## 15. Exact CPU Contract Tests

All existing tests must continue to pass.

Add at least the following named tests.

### 15.1 Historical-preservation tests

```text
test_legacy_evaluation_plan_is_byte_identical
test_legacy_gate04_plan_is_byte_identical
test_gate04_ordered_prompt_hash_is_frozen
test_gate04_thresholds_are_unchanged
test_gate04_error_still_does_not_enter_epsilon_y
test_old_gate04_and_gate08_reports_are_protocol_hashed
```

### 15.2 New donor-plan tests

```text
test_basis_v2_nouns_are_exact
test_basis_v2_nouns_disjoint_from_evaluation
test_basis_v2_nouns_disjoint_from_legacy_donors
test_basis_v2_fit_count_is_512
test_basis_v2_holdout_count_is_256
test_basis_v2_radius_count_is_512
test_basis_v2_total_count_is_1280
test_basis_v2_role_order_is_exact
test_basis_v2_orientation_quotas_are_exact
test_basis_v2_prompt_keys_are_unique
test_basis_v2_suffixes_are_disjoint_across_roles_and_bins
test_basis_v2_pair_hash_is_deterministic
test_basis_v2_orientation_hash_is_deterministic
test_basis_v2_quota_failure_stops_before_model_loading
```

### 15.3 Basis-estimator tests

```text
test_basis_estimator_requires_float64
test_basis_estimator_rejects_centered_substitution
test_basis_estimator_uses_scipy_gesvd
test_basis_estimator_uses_full_matrices_false
test_basis_estimator_uses_overwrite_false
test_basis_estimator_requires_single_thread_blas
test_basis_estimator_serializes_threadpool_info
test_basis_estimator_repeated_svd_is_bitwise_equal
test_basis_estimator_sign_rule_is_exact
test_basis_estimator_returns_rank5_shape
test_basis_projector_is_symmetric
test_basis_projector_is_idempotent
test_basis_projector_has_trace_five
test_basis_orthogonality_threshold_is_enforced
```

### 15.4 Spectrum and stability tests

Use synthetic diagonal-spectrum matrices.

```text
test_fit_gap_equal_1_10_passes
test_fit_gap_below_1_10_fails
test_holdout_gap_equal_1_10_passes
test_holdout_gap_below_1_10_fails
test_rank5_floor_equal_1e_minus_4_passes
test_rank5_floor_below_1e_minus_4_fails
test_old_rank4_boundary_is_not_used
test_rank5_boundary_is_five_versus_six
test_principal_angle_equal_15_degrees_passes
test_principal_angle_above_15_degrees_fails
test_holdout_efficiency_equal_0_90_passes
test_holdout_efficiency_below_0_90_fails
test_leave_one_noun_requires_all_sixteen_pass
test_bootstrap_seed_is_exact
test_bootstrap_replicate_count_is_256
test_bootstrap_uses_noun_clusters_not_rows
test_bootstrap_quantile_method_is_higher
test_bootstrap_q95_equal_15_passes
test_bootstrap_q95_above_15_fails
test_no_rank6_fallback_exists
```

### 15.5 Rank-generic theorem tests

```text
test_matched_bypass_identity_rank5
test_structural_inverse_rank5
test_direct_bypass_recovery_rank5
test_probe_design_rank_is_50
test_orthogonal_basis_rotation_transforms_A
test_orthogonal_basis_rotation_transforms_P
test_orthogonal_basis_rotation_transforms_D
test_path_contraction_is_rotation_invariant
test_projected_physical_direction_is_rotation_invariant
test_independent_target_is_rotation_invariant
test_actual_gate_coordinates_do_not_rotate
```

### 15.6 Compute-contract tests

```text
test_rank5_calls_per_gate_radius_system_is_52
test_rank5_calls_per_system_is_1041
test_rank5_mixed_calls_per_item_is_2082
test_first_order_direction_count_is_250
test_first_order_calls_per_item_is_2082
test_factorial_calls_per_item_remain_16
test_tensor_tail_total_is_1605120
test_total_tail_count_is_1610848
test_jvp_count_remains_1152
test_full_model_count_is_5056
test_raw_invocation_count_is_1617056
test_effective_unit_count_is_1618208
test_conservative_unit_count_is_1643488
```

### 15.7 Hardcoding and causal-preservation tests

```text
test_runner_contains_no_np_zeros_4_for_residual_coordinates
test_runner_contains_no_range_4_for_residual_axes
test_tail_contains_no_batch_4_shape_contract
test_target_contains_no_batch_4_shape_contract
test_residual_rank_is_exactly_five
test_selected_gates_are_unchanged
test_resid_mid_site_is_unchanged
test_gate_pre_site_is_unchanged
test_gate_post_site_is_unchanged
test_target_bypass_subtraction_site_is_unchanged
test_matched_control_code_is_unchanged
test_target_import_firewall_is_unchanged
test_confirmation_lock_is_unchanged
test_statistical_thresholds_are_unchanged
```

### 15.8 Provenance and one-run tests

```text
test_clean_repository_rejects_tracked_change
test_clean_repository_rejects_staged_change
test_clean_repository_rejects_untracked_file
test_clean_repository_requires_main_branch
test_clean_repository_requires_review_commit_ancestor
test_prepare_rejects_nonempty_output_root
test_manifest_requires_repository_clean_false_is_impossible
test_manifest_attempt_index_is_one
test_manifest_retry_allowed_is_false
test_prepare_cannot_be_run_twice
test_stopped_rank5_run_cannot_open_development
```

### 15.9 Lockfile test

```text
test_requirements_lock_adds_only_threadpoolctl_3_6_0
```

The complete CPU suite must pass locally and on the server before any model load.

---

## 16. Required GPU Assertions

The prepare phase must mechanically assert all of the following.

### 16.1 Provenance

```text
branch == main
repository status porcelain == ""
repository_dirty_at_launch == false
review commit b87300a... is an ancestor
output root was empty
attempt_index == 1
retry_allowed == false
```

### 16.2 Gate 04

```text
backend == eager
batch_size == 1
prompt_count == 32
ordered prompt hash == 619d21c...
converted-weight mismatch count == 0
all Gate-04 v1.1 thresholds pass
same-TL block-8 no-op <= 2e-5
HF-TL error enters epsilon_y == false
```

### 16.3 New donors

```text
basis_fit pairs == 512
basis_holdout pairs == 256
radius_v2 pairs == 512
total donor-v2 pairs == 1280
unique donor-v2 prompts == 2560
old/new donor noun overlap == 0
evaluation/new donor noun overlap == 0
prompt overlap among new roles == 0
```

### 16.4 CPU basis matrices

```text
fit matrix shape == [512, 768]
holdout matrix shape == [256, 768]
radius matrix shape == [512, 768]
all matrices dtype == float64
all matrices finite
fit and holdout matrix hashes stored
```

### 16.5 Thread and repeatability contract

```text
at least one BLAS runtime introspected
every BLAS runtime num_threads == 1
repeat singular values bitwise equal
repeat canonical U bitwise equal
orthogonality max abs <= 5e-13
```

### 16.6 Rank-five gates

```text
fit sigma5/sigma6 >= 1.10
fit sigma5/sigma1 >= 1e-4
holdout sigma5/sigma6 >= 1.10
holdout sigma5/sigma1 >= 1e-4
fit-holdout largest angle <= 15 degrees
holdout efficiency >= 0.90
all 16 leave-one-noun angles <= 15 degrees
bootstrap q95 higher <= 15 degrees
```

### 16.7 Artifact shapes

```text
U shape == [768, 5]
projector shape == [768, 768]
fit singular array length >= 6
holdout singular array length >= 6
leave-one bases shape == [16, 768, 5]
first-order directions shape == [250, 5]
```

### 16.8 Prepare boundary

After prepare succeeds, none of the following may exist:

```text
noise_audit_dev.json
dev_tensor_scores.parquet
dev_energy_targets.parquet
dev_result.json
frozen_analysis.json
confirm_tensor_scores.parquet
confirm_energy_targets.parquet
```

---

## 17. Required Artifacts

The redesigned prepare phase must produce:

```text
outputs/green_bridge/manifest.json
outputs/green_bridge/model_fingerprint.json
outputs/green_bridge/splits.json
outputs/green_bridge/development_splits.json
outputs/green_bridge/gate04_legacy_panel.json
outputs/green_bridge/hook_audit.json
outputs/green_bridge/donor_v2_plan.json
outputs/green_bridge/donor_v2_matrix_hashes.json
outputs/green_bridge/donor_basis.npz
outputs/green_bridge/basis_audit.json
outputs/green_bridge/basis_bootstrap.npz
outputs/green_bridge/radii.json
outputs/green_bridge/first_order_directions.npy
outputs/green_bridge/tail_audit.json
outputs/green_bridge/run_ledger.json
```

`donor_basis.npz` must contain:

```text
U
projector
singular_fit
singular_holdout
U_holdout
leave_one_bases
leave_one_angles
fit_pair_digests
holdout_pair_digests
```

`basis_bootstrap.npz` must contain:

```text
sampled_noun_indices
angles_degrees
rank_floor_failures
```

---

## 18. Exact Manifest Amendment

The new manifest must have:

```yaml
schema_version: green-bridge-manifest-v1.2

repository:
  url: https://github.com/ScottBlizzard/idle_1
  branch: main
  review_commit: b87300a6f56cb4706db090486d8bec77a2fc2b23
  execution_commit: "<full committed implementation hash>"
  review_commit_is_ancestor: true
  status_porcelain: ""
  repository_dirty_at_launch: false

run:
  protocol_run_id: green-bridge-v1.2-one-shot
  attempt_index: 1
  retry_allowed: false
  prepare_restart_allowed: false
  development_restart_allowed: false
  confirmation_restart_allowed: false

amendment:
  id: GPTPRO-GREEN-GATE08-v2-20260805
  decision_document: analysis/GPTPRO_GREEN_GATE08_DECISION_20260805.md
  prior_gate04_amendment: GPTPRO-GREEN-GATE04-v2-20260805
  prior_execution_commit: 5083774e03b99c9958312c6686cf3ead40c3c115
  prior_first_failed_gate: 08_BASIS_SPECTRUM
  prior_sigma4_over_sigma5: 1.04
  prior_sigma4_over_sigma1: 0.5501
  prior_development_responses_observed: false
  prior_confirmation_responses_observed: false
  confirmation_remained_locked: true
  scientific_design_change:
    residual_rank: "4 -> 5"
    basis_object: projector-covariant
    new_donor_population: true
  unchanged:
    theorem_identity: true
    actual_gate_coordinates: true
    intervention_sites: true
    matched_control: true
    independent_target: true
    residual_bypass_subtraction: true
    evaluation_population: true
    radii_multiplier: true
    baselines: true
    development_rules: true
    confirmation_rules: true

prior_artifacts:
  protocol_v1_gate04_stop_preserved: true
  protocol_v1_1_gate08_stop_preserved: true
  gate08_stop:
    result_sha256: 7d52411b487f7e85f0dc539c760541d16bf5c9b756da75490edd8b9ad5ad7f90
    manifest_sha256: baff192581726f4cae8f23418df5600ccb0fff549b0c81edff8c2c1f95d914df
    hook_audit_sha256: 49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99
    log_sha256: 845cb7746be048dacbcb6c841e45d29e3d51d7e7632074e08b63c92dea5d8fb8
    repository_dirty_at_launch: true
    dirty_reason: untracked_offline_transport_bundle

dimensions:
  residual_rank: 5
  selected_gates: 10
  output_dimension: 100
  tensor_shape: [100, 5, 10]
  kronecker_design_rank: 50

structural_object:
  equivalence: "(U,A,P,D) ~ (UQ,AQ,PQ,DQ), Q in O(5)"
  physical_projector: "Pi = U U^T"
  gate_coordinates_rotated: false
  matched_bypass_identity: "H_path - H_control = C A"
  inverse_changed: false

donor_v2:
  salt: idle1-gt-bridge-basis-v2-20260805
  nouns:
    - rebellion
    - revolution
    - occupation
    - blockade
    - crusade
    - migration
    - settlement
    - construction
    - administration
    - regime
    - competition
    - partnership
    - transition
    - expansion
    - uprising
    - conflict
  centuries: [11, 13, 15, 17]
  roles:
    basis_fit:
      pairs: 512
      pairs_per_cell: 4
      orientation: {up: 2, down: 2}
    basis_holdout:
      pairs: 256
      pairs_per_cell: 2
      orientation: {up: 1, down: 1}
    radius_v2:
      pairs: 512
      pairs_per_cell: 4
      orientation: {up: 2, down: 2}
  prompt_level_disjointness: true
  unique_prompts: 2560
  failed_quota_replacement_allowed: false
  old_donor_responses_reused: false

basis:
  fit_matrix_shape: [512, 768]
  holdout_matrix_shape: [256, 768]
  centered: false
  dtype: float64
  device: CPU
  scipy_function: scipy.linalg.svd
  full_matrices: false
  lapack_driver: gesvd
  overwrite_a: false
  check_finite: true
  rank: 5
  sign_rule: largest_absolute_coordinate_positive_first_index_tie
  repeated_svd_bitwise_equal: true
  orthogonality_max_abs: 5.0e-13
  threadpoolctl_version: 3.6.0
  blas_threads: 1
  thresholds:
    fit_sigma5_over_sigma6: 1.10
    fit_sigma5_over_sigma1: 1.0e-4
    holdout_sigma5_over_sigma6: 1.10
    holdout_sigma5_over_sigma1: 1.0e-4
    fit_holdout_angle_degrees: 15.0
    holdout_efficiency: 0.90
    leave_one_noun_angle_degrees: 15.0
    bootstrap_q95_angle_degrees: 15.0
  bootstrap:
    replicates: 256
    unit: noun
    rng: numpy.PCG64
    seed_material: idle1-gt-bridge-basis-v2-20260805:noun-bootstrap
    quantile: 0.95
    quantile_method: higher

radii:
  residual_scale: "median(norm(U^T d)/sqrt(5))"
  multiplier: 0.20
  gate_scale_unchanged: true
  donor_role: radius_v2
  leave_one_noun_relative_change_max: 0.20
  search_allowed: false
  inflation_allowed: false

gate04_replay:
  audit_version: hf-tl-fidelity-v2
  legacy_panel_replayed: true
  ordered_prompt_keys_sha256: 619d21c10d4f30e6ce2597c3ba4df1de72cf0cb4f6cce322d82c2d3ec62803ce
  backend: eager
  batch_size: 1
  thresholds_changed: false
  error_enters_epsilon_y: false

compute:
  mixed_per_tensor_item: 2082
  first_order_per_tensor_item: 2082
  factorial_per_tensor_item: 16
  first_order_residual_directions: 250
  tensor_items_total: 384
  energy_items_total: 384
  tail_evaluations_total: 1610848
  jvp_invocations_total: 1152
  full_model_evaluations_total: 5056
  raw_invocations_total: 1617056
  effective_units_total: 1618208
  conservative_units_total: 1643488
  development_effective_units: 541920
  confirmation_effective_units: 1076288

confirmation:
  locked_at_prepare: true
  all_v1_1_rules_unchanged: true
  retries: 0

protocol_files:
  - analysis/GPTPRO_GREEN_BRIDGE_20260805.md
  - analysis/GPTPRO_GREEN_GATE04_DECISION_20260805.md
  - analysis/GREEN_SERVER_GATE04_20260805.md
  - analysis/GREEN_SERVER_GATE08_20260805.md
  - analysis/GPTPRO_GREEN_GATE08_DECISION_20260805.md
  - requirements-green-bridge.lock

protocol_sha256:
  analysis/GPTPRO_GREEN_BRIDGE_20260805.md: "<computed at launch>"
  analysis/GPTPRO_GREEN_GATE04_DECISION_20260805.md: "<computed at launch>"
  analysis/GREEN_SERVER_GATE04_20260805.md: "<computed at launch>"
  analysis/GREEN_SERVER_GATE08_20260805.md: "<computed at launch>"
  analysis/GPTPRO_GREEN_GATE08_DECISION_20260805.md: "<computed at launch>"
  requirements-green-bridge.lock: "<computed at launch>"

terminal_rule:
  rank5_gate_failure: terminate_oral_line
  rank6_fallback: false
  donor_replacement: false
  threshold_amendment: false
  second_basis_run: false
```

---

## 19. Exact Archive and Commit Procedure

Run from the repository root.

### 19.1 Verify the reviewed state

```bash
set -euo pipefail

git checkout main
test "$(git rev-parse HEAD)" = \
  "b87300a6f56cb4706db090486d8bec77a2fc2b23"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

### 19.2 Archive the current Gate-08 stop

```bash
set -euo pipefail

CURRENT_ROOT="outputs/green_bridge"
ARCHIVE_ROOT="outputs/green_bridge_gate08_stop_5083774_20260805"

test -d "${CURRENT_ROOT}"
test ! -e "${ARCHIVE_ROOT}"

python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("outputs/green_bridge")
result = json.loads((root / "result.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

assert result["verdict"] == "STOP"
assert result["first_failed_gate"] == "08_BASIS_SPECTRUM"
assert manifest["repository_dirty_at_launch"] is True
assert not (root / "noise_audit_dev.json").exists()
assert not (root / "dev_tensor_scores.parquet").exists()
assert not (root / "dev_energy_targets.parquet").exists()
assert not (root / "frozen_analysis.json").exists()

expected = {
    "result.json":
        "7d52411b487f7e85f0dc539c760541d16bf5c9b756da75490edd8b9ad5ad7f90",
    "manifest.json":
        "baff192581726f4cae8f23418df5600ccb0fff549b0c81edff8c2c1f95d914df",
    "hook_audit.json":
        "49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99",
}
for name, digest in expected.items():
    observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert observed == digest, (name, observed, digest)
PY

mv "${CURRENT_ROOT}" "${ARCHIVE_ROOT}"

if test -f "logs/green_bridge_prepare_gate04_v2.log"; then
  cp \
    "logs/green_bridge_prepare_gate04_v2.log" \
    "${ARCHIVE_ROOT}/green_bridge_prepare_gate04_v2.log"
fi

cat > "${ARCHIVE_ROOT}/ARCHIVE_METADATA.json" <<'JSON'
{
  "execution_commit": "5083774e03b99c9958312c6686cf3ead40c3c115",
  "review_commit": "b87300a6f56cb4706db090486d8bec77a2fc2b23",
  "first_failed_gate": "08_BASIS_SPECTRUM",
  "sigma4_over_sigma5": 1.04,
  "sigma4_over_sigma1": 0.5501,
  "development_responses_observed": false,
  "confirmation_responses_observed": false,
  "repository_dirty_at_launch": true,
  "dirty_reason": "untracked offline Git transport bundle",
  "tracked_changes_at_launch": false
}
JSON

(
  cd "${ARCHIVE_ROOT}"
  sha256sum \
    result.json \
    manifest.json \
    hook_audit.json \
    ARCHIVE_METADATA.json \
    > ARCHIVED_GATE08_SHA256.txt
)
```

Do not modify:

```text
outputs/green_bridge_gate04_stop_0c81e05_20260805
```

### 19.3 Implement and commit

After implementing this document:

```bash
set -euo pipefail

git add \
  analysis/GPTPRO_GREEN_GATE08_DECISION_20260805.md \
  requirements-green-bridge.lock \
  src/green_bridge_spec.py \
  src/green_bridge_dataset.py \
  src/green_bridge_basis.py \
  src/matched_bypass_gate.py \
  src/green_bridge_tail.py \
  src/green_bridge_path_target.py \
  src/green_bridge_numerics.py \
  src/exp_green_bridge_gpt2.py \
  src/analyze_green_bridge.py \
  src/test_green_bridge_contract.py

git commit -m \
  "Preregister rank-five projector bridge after donor-only Gate08 stop"

git merge-base --is-ancestor \
  b87300a6f56cb4706db090486d8bec77a2fc2b23 \
  HEAD

test -z "$(git status --porcelain=v1 --untracked-files=all)"

EXECUTION_COMMIT="$(git rev-parse HEAD)"
printf '%s\n' "${EXECUTION_COMMIT}"
```

No offline bundle may remain anywhere inside the worktree.

---

## 20. Exact Server Environment

```bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONPATH="${PWD}/src"

test -z "$(git status --porcelain=v1 --untracked-files=all)"
git branch --show-current | grep -Fx main
git merge-base --is-ancestor \
  b87300a6f56cb4706db090486d8bec77a2fc2b23 \
  HEAD

python --version
python -m pip check
nvidia-smi \
  --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader
```

Verify package versions:

```bash
python - <<'PY'
import importlib.metadata
import platform
import torch

expected = {
    "torch": "2.7.1",
    "transformer-lens": "3.6.0",
    "transformers": "5.13.0",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "pandas": "2.2.3",
    "pyarrow": "19.0.1",
    "threadpoolctl": "3.6.0",
}

assert platform.python_version() == "3.11.13"
for package, version in expected.items():
    observed = importlib.metadata.version(package)
    if package == "torch":
        assert observed in {version, version + "+cu126"}
    else:
        assert observed == version, (package, observed, version)

assert torch.version.cuda == "12.6"
assert torch.cuda.is_available()
PY
```

---

## 21. Exact Test and Run Commands

### 21.1 CPU tests

```bash
set -euo pipefail

export PYTHONPATH="${PWD}/src"

python -m unittest discover \
  -s src \
  -p 'test_*.py' \
  -v \
  2>&1 | tee logs/green_bridge_v12_cpu_tests.log
```

Every existing and new test must pass.

### 21.2 Empty output root

```bash
set -euo pipefail

test ! -e outputs/green_bridge
mkdir -p logs
```

Do not manually create `outputs/green_bridge`; the runner must create it only after the clean-worktree gate passes.

### 21.3 Prepare phase

```bash
set -euo pipefail

python src/exp_green_bridge_gpt2.py \
  --phase prepare \
  --device cuda:0 \
  --output-root outputs/green_bridge \
  2>&1 | tee logs/green_bridge_v12_prepare.log
```

Do not use `--phase all`.

### 21.4 Mechanical prepare verification

```bash
set -euo pipefail

python - <<'PY'
import json
from pathlib import Path

import numpy as np

root = Path("outputs/green_bridge")

manifest = json.loads(
    (root / "manifest.json").read_text(encoding="utf-8")
)
basis_audit = json.loads(
    (root / "basis_audit.json").read_text(encoding="utf-8")
)
hook_audit = json.loads(
    (root / "hook_audit.json").read_text(encoding="utf-8")
)
tail_audit = json.loads(
    (root / "tail_audit.json").read_text(encoding="utf-8")
)
plan = json.loads(
    (root / "donor_v2_plan.json").read_text(encoding="utf-8")
)

assert manifest["schema_version"] == "green-bridge-manifest-v1.2"
assert manifest["repository"]["repository_dirty_at_launch"] is False
assert manifest["run"]["attempt_index"] == 1
assert manifest["run"]["retry_allowed"] is False
assert manifest["prepare_complete"] is True
assert manifest.get("confirmation_open", False) is False

assert plan["counts"] == {
    "basis_fit": 512,
    "basis_holdout": 256,
    "radius_v2": 512,
}
assert plan["unique_prompt_count"] == 2560
assert plan["prompt_overlap_count"] == 0
assert plan["legacy_noun_overlap_count"] == 0
assert plan["evaluation_noun_overlap_count"] == 0

hf = hook_audit["hf_vs_tl"]
assert hf["audit_version"] == "hf-tl-fidelity-v2"
assert hf["hf_attention_implementation"] == "eager"
assert hf["ordered_prompt_keys_sha256"] == (
    "619d21c10d4f30e6ce2597c3ba4df1de"
    "72cf0cb4f6cce322d82c2d3ec62803ce"
)
assert hf["passed"] is True
assert hf["hf_tl_error_enters_epsilon_y"] is False
assert hook_audit["no_op_patch"]["max_abs"] <= 2e-5

assert basis_audit["rank"] == 5
assert basis_audit["fit"]["sigma5_over_sigma6"] >= 1.10
assert basis_audit["fit"]["sigma5_over_sigma1"] >= 1e-4
assert basis_audit["holdout"]["sigma5_over_sigma6"] >= 1.10
assert basis_audit["holdout"]["sigma5_over_sigma1"] >= 1e-4
assert basis_audit["fit_holdout_angle_degrees"] <= 15.0
assert basis_audit["holdout_efficiency"] >= 0.90
assert max(
    basis_audit["leave_one_noun_angles_degrees"].values()
) <= 15.0
assert basis_audit["bootstrap"]["q95_higher_degrees"] <= 15.0
assert basis_audit["repeated_svd_bitwise_equal"] is True
assert all(
    row["num_threads"] == 1
    for row in basis_audit["threadpools"]
    if row["user_api"] == "blas"
)

npz = np.load(root / "donor_basis.npz", allow_pickle=False)
assert npz["U"].shape == (768, 5)
assert npz["projector"].shape == (768, 768)
assert npz["U_holdout"].shape == (768, 5)
assert npz["leave_one_bases"].shape == (16, 768, 5)

directions = np.load(
    root / "first_order_directions.npy",
    allow_pickle=False,
)
assert directions.shape == (250, 5)

assert tail_audit["max_abs"] <= 2e-5
assert tail_audit["max_derivative_relative"] <= 1e-4

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

If any assertion fails, the oral line terminates. Do not prepare again.

### 21.5 Development

Only after the one prepare run passes:

```bash
set -euo pipefail

python src/exp_green_bridge_gpt2.py \
  --phase development \
  --device cuda:0 \
  --output-root outputs/green_bridge \
  2>&1 | tee logs/green_bridge_v12_development.log
```

The frozen development rules mechanically decide whether confirmation opens.

### 21.6 Confirmation

Run confirmation only if:

```text
manifest.confirmation_open == true
```

and the frozen-analysis hash is valid.

```bash
set -euo pipefail

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/green_bridge")
manifest = json.loads(
    (root / "manifest.json").read_text(encoding="utf-8")
)

assert manifest["confirmation_open"] is True
assert (root / "frozen_analysis.json").is_file()
assert manifest["run"]["attempt_index"] == 1
assert manifest["run"]["retry_allowed"] is False
PY

python src/exp_green_bridge_gpt2.py \
  --phase confirmation \
  --device cuda:0 \
  --output-root outputs/green_bridge \
  2>&1 | tee logs/green_bridge_v12_confirmation.log
```

---

## 22. One-Run and No-Retry Rule

This document authorizes exactly one rank-five protocol execution.

The prepare phase may be invoked once. Development and confirmation are continuations of the same run, not separate attempts.

After the first model response under protocol v1.2:

- no donor noun may change;
- no pair may change;
- no basis role may change;
- no rank may change;
- no threshold may change;
- no basis estimator may change;
- no bootstrap rule may change;
- no hardware precision may change;
- no failed item may be replaced;
- no second output root may be created for the same protocol;
- no rank-six fallback may be executed.

If any of the new Gate-08 subgates fails, the binding result is:

```text
TERMINATE_GREEN_BRIDGE_ORAL_LINE
```

and no further server experiment is authorized.

A process crash, infrastructure failure, or package mismatch after the one-shot run ledger is created also requires a new senior-auditor decision; the executor may not silently restart.

---

## 23. Why This Is Not Outcome Adaptation

This redesign is scientifically valid for the following reasons.

### 23.1 No scientific prediction outcome was observed

The stopped execution ended before:

- leave-one-noun testing;
- radii;
- manual-tail auditing;
- duplicate-noise evaluation;
- development tensor scores;
- development targets;
- baseline scores;
- development RMSE;
- confirmation access.

The only new scientific information is that the frozen rank-four donor projector was not isolated at its fourth/fifth boundary. 

### 23.2 The old donor responses are excluded

The redesigned basis, holdout, radii, and stability analyses use a new noun population. No old donor prompt response can enter the redesigned estimator.

### 23.3 Rank five is fixed before new data

Rank five is not selected after inspecting a new spectrum. It is fixed here as the minimal completion of the only observed boundary cluster.

### 23.4 The new boundary uses the old threshold

The design retains:

\[
1.10
\]

rather than moving it below `1.04`.

### 23.5 The new stability gates are external to the scientific target

The fit/holdout angle, holdout efficiency, leave-one-noun angle, and noun bootstrap use only donor chords. They do not access:

- clean–patched path mismatch;
- mixed-response predictions;
- behavior baselines;
- development labels;
- confirmation labels.

### 23.6 There is no model-selection menu

Only one estimator and one rank are authorized. A failure terminates the line.

### 23.7 Gate 04 remains separate

The Gate-04 replay verifies implementation portability. It does not select the residual projector or determine scientific scores.

---

## 24. ICLR-Oral-Level Claim Boundary

### 24.1 Claim preserved and strengthened

If every development and confirmation condition passes, the project may claim:

> In the pinned GPT-2-small Greater-Than setting, matched-bypass output-response interventions non-tautologically identify a projector-covariant selected path operator from a preregistered five-dimensional block-10 residual subspace through ten actual MLP-10 GELU gates. The identified operator predicts an independently implemented, finite-radius, residual-bypass-subtracted path-specific mismatch between a clean target system and an MLP-8-patched corrupted system.

This retains the original theoretical height:

- actual nonlinear gate coordinates;
- explicit structural inverse;
- matched bypass cancellation;
- arbitrary smooth downstream transformer computation;
- no complete-cut assumption;
- no hidden bi-Lipschitz constant;
- constructive non-identifiability at zero curvature;
- independent absolute path target;
- frozen finite-radius and finite-sample protocol.

The projector-covariant formulation improves the claim by making it invariant to arbitrary orthogonal coordinates within the donor subspace.

### 24.2 Required wording change

The only necessary main-claim wording change is:

```text
four-dimensional donor-derived residual subspace
```

to:

```text
preregistered five-dimensional cluster-complete donor projector
```

The paper must report:

- the failed rank-four donor-only preflight;
- the reason rank five was preregistered;
- the entirely fresh donor population;
- the one-shot rank-five gate;
- all stopped-run and amendment provenance.

### 24.3 Claims still prohibited

The redesign does not authorize claims of:

- complete transformer-circuit identification;
- identification outside the five-dimensional projector;
- identification of all MLP-10 gates;
- identification of MLPs 8, 9, or 11 as simultaneous mediators;
- global intervention equivalence;
- transfer to other templates, tasks, or models;
- adaptive rank recovery;
- task-supervised subspace discovery;
- a universal IRS causal certificate.

---

## 25. Final Binding Determination

There is no material implementation defect that licenses replaying the failed rank-four donor gate.

The failed fourth-versus-fifth separation does not invalidate the matched-bypass theorem or the transformer causal bridge. It invalidates only the claim that the original donor distribution uniquely identifies a four-dimensional residual projector.

A single rank-five, projector-covariant, fresh-donor redesign is theoretically sound, computationally feasible, strictly more stable in its scientific object, and fully falsifiable. It preserves the independent target, actual MLP gate coordinates, matched control, residual-bypass subtraction, evaluation population, confirmation lock, and oral-level empirical decision rules.

No alternative rank or second attempt is authorized.

B. PREREGISTERED_BASIS_REDESIGN_AND_NEW_RUN