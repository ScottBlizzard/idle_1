# Bridge Verdict

**Decision: GREEN.**

A valid real-transformer bridge exists. The admissible construction is not the rejected two-block \(M_1\times M_2\) cut. It is a **matched-bypass gate-identification experiment** at the actual GPT-2-small block-10 MLP preactivation coordinates:

\[
\text{block-10 resid\_mid subspace}
\;\longrightarrow\;
\text{selected block-10 preactivations}
\;\longrightarrow\;
\text{their exact GELU gates}
\;\longrightarrow\;
\text{all downstream routes to year logits}.
\]

The bridge has four essential properties.

1. The upstream perturbation is inserted at `blocks.10.hook_resid_mid`, after block-10 attention and before block-10 LayerNorm/MLP.
2. The downstream mediators are actual coordinates of `blocks.10.mlp.hook_pre` and `hook_post`; no rotation of GELU coordinates is used.
3. A matched control preserves the direct residual \(x\)-bypass, the downstream nonlinear computation, and the selected gate write while severing only the edge from \(x\) to the selected gate.
4. The absolute target is a separately implemented, finite-radius, edge-isolated path-specific effect in Greater-Than logit-margin units. It uses disjoint prompts and does not reuse the mixed-response endpoints or inverse.

The resulting identification identity is

\[
H^{P}_{sij}-H^{C}_{sij}=C_{sj}A_{sji},
\]

where \(A_{sji}\) is the derivative from residual-subspace coordinate \(i\) to actual gate preactivation \(j\), \(C_{sj}\) is an observed vector-valued gate-curvature response, and the structural path tensor is

\[
P_{s,:,i,j}=G_{sj}A_{sji}.
\]

Whenever \(C_{sj}\neq 0\), the inverse is explicit:

\[
A_{sji}
=
\frac{
\left\langle
C_{sj},
H^{P}_{sij}-H^{C}_{sij}
\right\rangle
}{
\|C_{sj}\|_2^2
}.
\]

This does not assume a positive unidentified \(\kappa\), a complete mediator cut, absent residual paths, faithfulness, or an unspecified bi-Lipschitz structural map. The only nondegeneracy condition is the directly measured vector \(C_{sj}\neq0\); failure of that condition has a constructive non-identifiability converse and is an empirical early-stop condition rather than a hidden assumption.

The Round-3 entry point states that the restricted ASG-RDAG theorem and its CPU implementation are already complete, and that the only unresolved issue is a real-transformer bridge; those results are accepted as closed records here and are not repeated. 

Authorization is therefore limited to **one frozen run under the manifest below**. GREEN does not pre-assert empirical success. Any failed topology audit, numerical audit, development gate, or confirmatory threshold terminates the oral line without changing the layer, gates, corruption, basis dimension, radii, target, or statistical analysis.

# Exact Transformer DAG Audit

## Why the previous two-block cut was invalid

GPT-2-small is a serial pre-LayerNorm residual network. Within an ordinary block, TransformerLens computes

\[
R_{\ell}^{\mathrm{pre}}
\rightarrow
\operatorname{LN}_{\ell,1}
\rightarrow
\operatorname{Attn}_{\ell}
\rightarrow
R_{\ell}^{\mathrm{mid}}
\rightarrow
\operatorname{LN}_{\ell,2}
\rightarrow
\operatorname{MLP}_{\ell}
\rightarrow
R_{\ell}^{\mathrm{post}}.
\]

Specifically, `hook_resid_mid` is applied to `resid_pre + attn_out`, the MLP consumes `ln2(resid_mid)`, and `hook_resid_post` is applied to `resid_mid + mlp_out`. The MLP itself applies `hook_pre` before the activation, `hook_post` after the exact configured activation, and then the output projection. 

Consequently:

- MLPs 8, 9, 10, and 11 are not parallel mediator blocks.
- attention head 9.1 is computed after MLP 8 but before MLP 9;
- block-10 attention is computed before the block-10 MLP;
- the residual stream carries direct writes around every later MLP;
- a raw MLP output is a residual write, not a coordinatewise nonlinear gate input;
- MLP 10 affects logits both directly through the residual stream and indirectly through block 11.

The published Greater-Than circuit analysis identifies MLPs 8–11 and attention head 9.1 as relevant direct contributors, reports that MLP 9 relies on head 9.1, and treats MLP 10’s direct-to-logit and via-MLP-11 paths jointly.  This makes the rejected grouping of raw head/MLP outputs into two simultaneous blocks topologically indefensible.

## Frozen computational DAG

Let \(\pi=L-1\) be the last prompt position, whose logits predict the two-digit ending year. The complete forward DAG relevant to this experiment is:

| Order | Node or edge | Exact tensor/site | Role |
|---:|---|---|---|
| 1 | Token and position embeddings | full sequence | Fixed prompt input |
| 2 | Blocks 0–7 | full sequence | Unmodified upstream computation |
| 3 | Block-8 attention | block 8 | Upstream computation |
| 4 | Block-8 MLP input and gates | block 8 | Upstream computation |
| 5 | Block-8 MLP residual write | `blocks.8.hook_mlp_out[:, π, :]` | Clean-to-corrupt patch defining the patched system |
| 6 | Block-8 residual addition | `blocks.8.hook_resid_post` | Carries patched write downstream |
| 7 | Block-9 attention, including head 9.1 | block 9 | Downstream of the MLP-8 patch; upstream of MLP 9 |
| 8 | Block-9 MLP and residual addition | block 9 | Serial upstream transformation |
| 9 | Block-10 attention | block 10 | Completes before the selected \(x\) intervention |
| 10 | Upstream mediator \(X\) | `blocks.10.hook_resid_mid[:, π, :]` | Add \(Ux\) only at \(\pi\) |
| 11 | Block-10 second LayerNorm | `blocks[10].ln2` | Exact nonlinear map from residual to MLP input |
| 12 | Gate preactivation \(Z_j\) | `blocks.10.mlp.hook_pre[:, π, j]` | Add scalar \(z\) to actual gate \(j\) |
| 13 | Exact GPT-2 activation | configured `model.blocks[10].mlp.act_fn` | Coordinatewise GELU |
| 14 | Post-GELU gate \(W_j\) | `blocks.10.mlp.hook_post[:, π, j]` | Selected gate remains live; omitted gates are anchored |
| 15 | MLP-10 output projection | `W_out[j, :]` and `b_out` | Gate write into residual stream |
| 16 | Block-10 residual addition | `blocks.10.hook_resid_post` | Contains gate path plus direct \(Ux\) bypass |
| 17 | Optional target-only subtraction | `blocks.10.hook_resid_post[:, π, :]` | Subtract \(Ux\), isolating selected gate paths |
| 18 | Block-11 attention | full sequence | Arbitrary smooth downstream route |
| 19 | Block-11 MLP | full sequence | Arbitrary smooth downstream route |
| 20 | Final LayerNorm | `model.ln_final` | Retained exactly |
| 21 | Unembedding | `model.unembed` | Retained exactly |
| 22 | Year-suffix output | logits `[:, π, ν_00:ν_99]` by indexed gather | \(k=100\) vector output |

TransformerLens exposes the residual hooks with shape `[batch, position, d_model]`, and the MLP hooks with shape `[batch, position, d_mlp]`. Its unembedding applies the learned \(W_U,b_U\) after the final normalization. 

## Topological-order proof

The \(x\) intervention is made at `blocks.10.hook_resid_mid`. In GPT-2’s serial block implementation this tensor is formed only after block-10 attention has returned and has been added to `resid_pre`. The block-10 MLP preactivation is then computed from `ln2(resid_mid)`. Therefore every directed edge

\[
X_i\rightarrow Z_j\rightarrow W_j
\]

is forward-topological.

Conversely, block-8 MLP, block-9 attention including head 9.1, block-9 MLP, and block-10 attention have already executed before \(X\) exists. They cannot be downstream gate mediators in this experiment. Their effects are incorporated into the system-specific anchor \(R_s\).

After the selected block-10 gate writes are added to the residual stream, block 11, final LayerNorm, and unembedding remain downstream. They are represented by a single arbitrary smooth map \(F_s\); they need not be linearized internally or partitioned into artificial “direct” and “via MLP 11” blocks.

## Complete bypass ledger

| Possible bypass or nuisance route | Treatment |
|---|---|
| Direct residual \(Ux\rightarrow R_{10}^{\mathrm{post}}\rightarrow\) logits | Preserved identically in path and matched-control responses; therefore cancels in \(H^P-H^C\). Subtracted explicitly in the independent target. |
| Dependence of omitted MLP-10 gates on \(x\) | Every omitted post-GELU coordinate at position \(\pi\) is clamped to its system-specific anchor. |
| Dependence of another selected gate on \(x\) during one-gate identification | All gates other than the currently identified \(j\) are anchored, including the other nine gates in the declared set. |
| Block-10 LayerNorm mixing residual coordinates | Included exactly in \(a_{sj}(x)\); not approximated as diagonal or linear. |
| MLP-10 output bias | Common to the anchor and all interventions; cancels in residual increments. |
| Direct MLP-10-to-logit route | Included in \(F_s\). |
| MLP-10-to-block-11-to-logit route | Included in the same \(F_s\). |
| Block-11 attention and MLP nonlinearities | Included without decomposition in \(F_s\). |
| Final LayerNorm nonlinearity | Included without approximation in \(F_s\). |
| Head 9.1 and MLPs 8–9 | Strictly upstream of \(X\); absorbed into \(R_s\). |
| Block-10 attention | Strictly upstream of `resid_mid`; absorbed into \(R_s\). |
| Earlier token positions | Never directly edited; retained in the full-sequence anchor passed to block 11. |
| Attention effects caused by the edited last-position residual in block 11 | Included in \(F_s\). |
| PCA rotation of gate coordinates | Prohibited. PCA/SVD is used only to choose the residual-stream \(X\) subspace. Gate coordinates remain actual MLP coordinates. |

# Structural Object and Identification Result

## Frozen notation

Systems are

\[
s\in\{\mathrm{tar},\mathrm{pat},\mathrm{cor}\},
\]

where `tar` is the clean prompt, `cor` is the suffix-corrupted prompt, and `pat` is the corrupted prompt with the clean block-8 MLP residual write patched at the final position.

Let

- \(d=768\) be GPT-2-small’s residual dimension;
- \(m=3072\) be its MLP width;
- \(k=100\) be the selected year-suffix logit dimension;
- \(r_1=4\) be the residual-subspace dimension;
- \(r_2=10\) be the number of selected actual MLP-10 gates;
- \(U\in\mathbb R^{768\times4}\) be the frozen orthonormal residual basis;
- \(E_\pi v\) denote insertion of \(v\in\mathbb R^{768}\) only at sequence position \(\pi\);
- \(J=(2326,1138,2287,606,2848,2305,46,2659,946,1616)\).

The first three listed neurons are the three most important MLP-10 neurons reported in the Greater-Than analysis, and the remaining seven are ranks 4–10 from its published ordering. The paper reports that no individual neuron implements Greater-Than, whereas the top ten form an imperfect Greater-Than pattern as a group. 

For each system \(s\), cache:

\[
R_s\in\mathbb R^{L\times768}
\]

at `blocks.10.hook_resid_mid`, and

\[
R_s^+\in\mathbb R^{L\times768}
\]

at `blocks.10.hook_resid_post`.

Let \(c_j=W_{\mathrm{out}}[j,:]\in\mathbb R^{768}\). Let \(\psi\) be the model’s exact configured activation function, called through `model.blocks[10].mlp.act_fn`.

The selected gate preactivation is

\[
a_{sj}(x)
=
\left[
\operatorname{LN}_{10,2}
\!\left(R_s+E_\pi Ux\right)_{\pi,:}
W_{\mathrm{in}}
+b_{\mathrm{in}}
\right]_j,
\]

with anchor

\[
a_{sj}=a_{sj}(0)
\]

and upstream edge derivative

\[
A_{sji}
=
\left.
\frac{\partial a_{sj}(x)}{\partial x_i}
\right|_{x=0}.
\]

Define the exact downstream map

\[
F_s(v)
=
\mathcal L_{100}
\!\left(
\operatorname{Continue}_{11:\mathrm{logits}}
\left(R_s^+ + E_\pi v\right)
\right),
\]

where \(\mathcal L_{100}\) gathers the 100 frozen two-digit suffix logits. Thus \(F_s:\mathbb R^{768}\to\mathbb R^{100}\) contains block-11 attention, block-11 MLP, their residual routes, final LayerNorm, and unembedding.

All maps are smooth in a neighborhood of the anchors: GPT-2 evaluation uses fixed weights and masks; its LayerNorm has positive \(\epsilon\); softmax and the configured GELU approximation are smooth.

## Observable path and matched-control systems

For one selected gate \(j\), define the path response

\[
Y^P_{sj}(x,z)
=
F_s\!\left(
Ux+
c_j\left[
\psi\!\left(a_{sj}(x)+z\right)-\psi(a_{sj})
\right]
\right).
\]

Operationally, all post-GELU coordinates except \(j\) are clamped at their system-specific anchors.

Define the matched-bypass control

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

The control is matched in every respect except one:

- it preserves the same \(Ux\) residual bypass;
- it preserves the same selected-gate \(z\) write;
- it uses the same \(F_s\) and the same downstream basepoint;
- it anchors the same 3071 omitted gates;
- but the selected gate’s preactivation no longer depends on \(x\).

Define observable derivatives at \((x,z)=(0,0)\):

\[
G_{sj}
=
\partial_zY^P_{sj}(0,0)
\in\mathbb R^{100},
\]

\[
C_{sj}
=
\partial_{zz}^2Y^P_{sj}(0,0)
\in\mathbb R^{100},
\]

\[
H^P_{sij}
=
\partial_{x_i z}^2Y^P_{sj}(0,0),
\qquad
H^C_{sij}
=
\partial_{x_i z}^2Y^C_{sj}(0,0),
\]

and

\[
J^P_{sji}
=
\partial_{x_i}Y^P_{sj}(0,0).
\]

## Independently defined structural object

The structural path tensor is defined from the actual computational edges, not from a mixed Hessian:

\[
\boxed{
P_{s,\alpha i j}
=
\left.
\frac{\partial Y_{s,\alpha}}{\partial W_j}
\right|_0
\left.
\frac{\partial W_j}{\partial Z_j}
\right|_0
\left.
\frac{\partial Z_j}{\partial X_i}
\right|_0
}
\]

for output coordinate \(\alpha\in\{0,\ldots,99\}\), residual coordinate \(i\in\{1,\ldots,4\}\), and actual gate \(j\in J\).

Since

\[
\frac{\partial W_j}{\partial Z_j}=\psi'(a_{sj})
\]

and a unit post-GELU change writes \(c_j\) into the residual stream,

\[
P_{s,:,i,j}
=
DF_s(0)[c_j]\psi'(a_{sj})A_{sji}.
\]

The direct \(X_i\)-bypass tensor is

\[
D_{s,:,i}=DF_s(0)[Ue_i].
\]

The identified equivalence class is therefore

\[
\mathcal S_{s,U,J}
=
\left(
D_s,\,
\{P_{s,:,i,j}\}_{i,j}
\right),
\]

together with \(A_{sj:}\) for gates satisfying the curvature condition. It is explicitly a **local, fixed-basis, selected-gate path object**. It is not a decomposition of the entire transformer circuit.

## Matched-Bypass Gate Identification Theorem

**Theorem.**  
Fix a system \(s\), a residual basis \(U\), and an actual MLP gate \(j\). Suppose \(F_s\) and \(a_{sj}\) are twice continuously differentiable near their anchors and that path and control interventions are constructed exactly as above. Then:

\[
\boxed{
H^P_{sij}-H^C_{sij}=C_{sj}A_{sji}
}
\]

for every \(i\). If \(C_{sj}\neq0\), then

\[
\boxed{
A_{sji}
=
\frac{
\langle C_{sj},H^P_{sij}-H^C_{sij}\rangle
}{
\|C_{sj}\|_2^2
}
}
\]

and

\[
\boxed{
P_{s,:,i,j}=G_{sj}A_{sji}.
}
\]

Moreover,

\[
\boxed{
D_{s,:,i}
=
J^P_{sji}-P_{s,:,i,j},
}
\]

which must be independent of \(j\). Thus the observable response jet

\[
\left\{
G_{sj},C_{sj},H^P_{sij},H^C_{sij},J^P_{sji}
\right\}_{i,j}
\]

maps injectively to \(\mathcal S_{s,U,J}\) on every gate for which \(C_{sj}\neq0\).

If \(C_{sj}=0\) and \(G_{sj}\neq0\), the path tensor is not identified from this response class. If \(G_{sj}=0\), then \(P_{s,:,i,j}=0\) for all \(i\), although \(A_{sj:}\) need not be identified.

# Complete Proof or Impossibility Argument

## Proof of the identification identity

Fix \(s,j,i\), and abbreviate

\[
a=a_{sj},\qquad A=A_{sji},\qquad c=c_j,\qquad u=Ue_i.
\]

Let

\[
b=c\,\psi'(a).
\]

Write \(L=DF_s(0)\) and \(Q=D^2F_s(0)\), where \(Q\) is the symmetric bilinear Hessian map of the vector-valued \(F_s\).

For the path system, its downstream residual increment is

\[
v^P(x,z)
=
Ux+
c\left[\psi(a_{sj}(x)+z)-\psi(a)\right].
\]

At the origin,

\[
v_x^P=u+bA,
\]

\[
v_z^P=b,
\]

\[
v_{xz}^P=c\,\psi''(a)A,
\]

and

\[
v_{zz}^P=c\,\psi''(a).
\]

The multivariate chain rule gives

\[
H^P_{sij}
=
Q[v_x^P,v_z^P]+L[v_{xz}^P].
\]

Substituting,

\[
\begin{aligned}
H^P_{sij}
&=
Q[u+bA,b]+L[c\psi''(a)A]\\
&=
Q[u,b]
+
A\,Q[b,b]
+
A\,L[c\psi''(a)].
\end{aligned}
\]

For the matched control,

\[
v^C(x,z)
=
Ux+c[\psi(a+z)-\psi(a)],
\]

so

\[
v_x^C=u,\qquad
v_z^C=b,\qquad
v_{xz}^C=0,\qquad
v_{zz}^C=c\psi''(a).
\]

Therefore

\[
H^C_{sij}=Q[u,b].
\]

The pure gate curvature is identical in path and control:

\[
\begin{aligned}
C_{sj}
&=
\partial_{zz}^2Y^P_{sj}(0,0)\\
&=
Q[b,b]+L[c\psi''(a)].
\end{aligned}
\]

Subtracting the matched control from the path response,

\[
\begin{aligned}
H^P_{sij}-H^C_{sij}
&=
A\left(
Q[b,b]+L[c\psi''(a)]
\right)\\
&=
C_{sj}A_{sji}.
\end{aligned}
\]

This proves the factorization.

If \(C_{sj}\neq0\), the scalar \(A_{sji}\) is the unique least-squares coefficient of the known vector \(C_{sj}\):

\[
A_{sji}
=
\frac{\langle C_{sj},H^P_{sij}-H^C_{sij}\rangle}
{\|C_{sj}\|_2^2}.
\]

There is no unspecified lower-Lipschitz constant. The empirical norm and noise separation of \(C_{sj}\) are measured directly.

## Proof that \(P=GA\)

The first derivative with respect to the gate intervention is

\[
G_{sj}
=
L[b]
=
DF_s(0)[c_j]\psi'(a_{sj}).
\]

The edge-product definition gives

\[
\begin{aligned}
P_{s,:,i,j}
&=
DF_s(0)[c_j]\,
\psi'(a_{sj})\,
A_{sji}\\
&=
G_{sj}A_{sji}.
\end{aligned}
\]

Thus the identified object coincides with the independently defined computational-path derivative.

## Proof of direct-bypass recovery

The path system’s first derivative with respect to \(x_i\) is

\[
\begin{aligned}
J^P_{sji}
&=
L[u+bA]\\
&=
L[u]+L[b]A\\
&=
D_{s,:,i}+P_{s,:,i,j}.
\end{aligned}
\]

Hence

\[
D_{s,:,i}
=
J^P_{sji}-P_{s,:,i,j}.
\]

Because the direct residual route does not depend on which selected gate is left live, this recovered quantity must agree across all active \(j\). Cross-gate disagreement is therefore a falsifiable audit of hook matching, anchoring, and finite-radius error.

## Injectivity

Suppose two structural parameter tuples produce the same observable jet and satisfy \(C_{sj}\neq0\). Equality of \(C_{sj}\) and \(H^P_{sij}-H^C_{sij}\) forces equality of every \(A_{sji}\) by the displayed inverse. Equality of \(G_{sj}\) then forces equality of every \(P_{s,:,i,j}=G_{sj}A_{sji}\). Finally, equality of \(J^P_{sji}\) forces equality of \(D_{s,:,i}\). Therefore the observable-to-structural map is injective.

## Why downstream LayerNorm and nonlinear bypasses do not invalidate the result

No linearity of \(F_s\) was used. Its Hessian terms appear in both the path and control:

\[
Q[u,b].
\]

That term is exactly the interaction between the direct residual bypass and the selected gate write through all downstream nonlinear computation. It cancels because the control preserves both arguments \(u\) and \(b\) at the same downstream basepoint.

The remaining downstream curvature

\[
Q[b,b]+L[c\psi''(a)]
\]

is not assumed known or nonzero. It is exactly the observable \(C_{sj}\). Final LayerNorm, block-11 attention, block-11 MLP, and both direct and indirect routes contribute to \(C_{sj}\) and \(G_{sj}\), but do not create an unobserved allocation ambiguity.

## Constructive converses and minimal failure cases

### First-order responses alone cannot identify the path

With only

\[
J^P=D+GA,
\]

choose any alternative scalar \(\widetilde A\) and define

\[
\widetilde D
=
D+G(A-\widetilde A).
\]

Then

\[
\widetilde D+G\widetilde A=D+GA,
\]

but the structural path changes from \(GA\) to \(G\widetilde A\). Therefore first-order restoration cannot allocate the response between the direct residual route and the gate path.

### A total mixed Hessian without a matched control is insufficient

Without \(H^C\),

\[
H^P=B+CA,
\qquad
B=Q[u,b].
\]

For any \(\widetilde A\), define

\[
\widetilde B=B+C(A-\widetilde A).
\]

Then

\[
\widetilde B+C\widetilde A=B+CA.
\]

A local downstream Hessian jet can realize the modified cross term \(\widetilde B\), while preserving the observed first-order gate response and pure gate curvature. Thus calling \(H^P\) a path tensor would be non-identifying.

### Zero curvature is a real obstruction

If

\[
C_{sj}=0
\]

while \(G_{sj}\neq0\), then

\[
H^P_{sij}-H^C_{sij}=0
\]

for every value of \(A_{sji}\). Altering \(A_{sji}\) changes the path tensor \(G_{sj}A_{sji}\), while a compensating change in \(D\) preserves \(J^P\). The path is not identified.

This is the minimal non-identifiability obstruction for the selected response design. The experiment therefore measures \(\|C_{sj}\|\) and its numerical error rather than asserting an unspecified positive inverse constant.

### Rank-deficient \(x\) probes leave structural directions unidentified

If the finite \(x\)-design matrix has null vector \(v\neq0\), then replacing

\[
A_{sj:}\mapsto A_{sj:}+tv
\]

does not change any measured directional response but changes \(P_{sj}\). The frozen coordinate design uses all four basis axes and has rank four.

### Rotating gate coordinates invalidates the structural equation

For a non-permutation rotation \(Q\),

\[
Q^\top\psi(a)
\neq
\psi(Q^\top a)
\]

in general. Therefore a PCA coordinate in post-GELU space is not an actual scalar gate with a known coordinatewise \(\psi',\psi''\). Gate PCA is prohibited.

### Moving \(x\) after the gate destroys the edge

An intervention at `hook_resid_post` cannot influence `mlp.hook_pre` in the same block. Then \(A=0\) by construction, regardless of the natural mechanism. The \(x\) intervention must occur at `hook_resid_mid`.

### Failing to subtract the target bypass changes the estimand

A target curve that injects \(Ux\) at `resid_mid` but does not subtract it at `resid_post` measures

\[
D\delta+\sum_jP_j\delta
\]

at first order. It is not a selected-gate path-specific target.

### Mismatched anchors invalidate cancellation

If path and control use different residual anchors, different omitted-gate anchors, different \(Ux\) writes, different selected-gate \(z\) writes, or different downstream basepoints, additional terms remain in \(H^P-H^C\). Exact center and no-op audits are therefore theorem preconditions, not implementation conveniences.

### A white-box gate Jacobian is not the claimed estimator

For this transparent model, \(A\) can also be computed from LayerNorm and \(W_{\mathrm{in}}\). That quantity is used only as an implementation audit. It is forbidden as an input to the primary predictor. The claim is that output-response interventions identify the selected path tensor, not that they are computationally cheaper than direct differentiation when internal weights are available.

# Independent Path-Specific Target

## Greater-Than logit-margin functional

Let \(\nu_q\) be the frozen token ID for two-digit suffix \(q\in\{00,\ldots,99\}\). For a clean start-year suffix \(y\in\{05,\ldots,94\}\), define

\[
\ell_y[q]
=
\begin{cases}
\dfrac{1}{99-y}, & q>y,\\[6pt]
-\dfrac{1}{y+1}, & q\le y.
\end{cases}
\]

For year-logit vector \(Y\in\mathbb R^{100}\),

\[
\ell_y^\top Y
=
\operatorname{mean}_{q>y}Y_q
-
\operatorname{mean}_{q\le y}Y_q.
\]

The clean \(y\) defines \(\ell_y\) for the target, patched, and corrupted systems. This is an absolute signed logit contrast, not a probability, cosine, attention score, restoration ratio, or quantity divided by a possibly small clean–corrupt gap. The original Greater-Than code likewise evaluates the final position and supports a logit-mean metric over year tokens. 

## Frozen semantic direction

For item \(n\), let

\[
d_n
=
R_{\mathrm{tar},n}[\pi,:]
-
R_{\mathrm{cor},n}[\pi,:]
\]

at block-10 `resid_mid`, and

\[
q_n=U^\top d_n.
\]

The item is direction-valid only if

\[
\|q_n\|_2\ge0.10\,\sigma_x.
\]

Define the frozen one-radius residual direction

\[
\delta_n
=
h_1\frac{q_n}{\|q_n\|_2}
\in\mathbb R^4.
\]

The clean–corrupt chord supplies only a direction. The magnitude \(h_1\) comes exclusively from the disjoint radius-donor set.

## Joint edge-isolated target curve

For all selected gates \(j\in J\), define

\[
\Gamma_{s,n}(t)
=
\sum_{j\in J}
c_j
\left[
\psi\!\left(a_{sj}(t\delta_n)\right)
-
\psi(a_{sj})
\right].
\]

The independent target curve is

\[
g_{s,n}(t)
=
\ell_y^\top F_s\!\left(\Gamma_{s,n}(t)\right).
\]

Operationally:

1. add \(E_\pi U(t\delta_n)\) at block-10 `hook_resid_mid`;
2. allow exactly the ten selected gate preactivations and GELUs to respond naturally;
3. clamp the other 3062 post-GELU coordinates to their system anchors;
4. after the MLP residual addition, subtract \(E_\pi U(t\delta_n)\) at `hook_resid_post`;
5. run block 11, final LayerNorm, and unembedding;
6. compute the logit-margin contrast.

The subtraction removes the residual bypass while retaining the selected-gate writes. Therefore every varying route from \(t\) to the output crosses one of the declared

\[
X\rightarrow Z_j\rightarrow W_j
\]

edges.

## Absolute finite-radius path-specific effect

For \(\rho\in\{1,\frac12\}\), define

\[
\operatorname{PSE}^{\rho}_{s,n}
=
\frac{
g_{s,n}(\rho)-g_{s,n}(-\rho)
}{
2\rho
}.
\]

The primary target uses \(\rho=1\):

\[
\operatorname{PSE}_{s,n}
=
\operatorname{PSE}^{1}_{s,n}.
\]

This is the signed average path-specific directional effect over a fixed, symmetric, one-radius intervention, expressed directly in logit-margin units.

For energy-evaluation items \(E_c\) in semantic cell \(c\),

\[
\operatorname{PSE}_{s,c}
=
\frac1{|E_c|}
\sum_{n\in E_c}
\operatorname{PSE}_{s,n},
\]

and the independently measured structural-mismatch target is

\[
\boxed{
T_c
=
\left|
\operatorname{PSE}_{\mathrm{pat},c}
-
\operatorname{PSE}_{\mathrm{tar},c}
\right|.
}
\]

The target-conditioning gap is

\[
C_c
=
\left|
\operatorname{PSE}_{\mathrm{tar},c}
-
\operatorname{PSE}_{\mathrm{cor},c}
\right|.
\]

This follows the repository’s required absolute path-specific target

\[
|\operatorname{PSE}_{\mathrm{patched}}-\operatorname{PSE}_{\mathrm{target}}|
\]

and avoids the prohibited behavioral and near-zero-denominator alternatives. 

## Relation to the identified tensor

Differentiating at zero gives

\[
\begin{aligned}
g'_{s,n}(0)
&=
\ell_y^\top
\sum_{j\in J}
DF_s(0)[c_j]\,
\psi'(a_{sj})\,
A_{sj:}\delta_n\\
&=
\ell_y^\top
\sum_{j\in J}
\sum_{i=1}^{4}
P_{s,:,i,j}\delta_{n,i}.
\end{aligned}
\]

Define the tensor prediction

\[
\theta_{s,n}
=
\ell_y^\top
\sum_{j\in J}
\sum_{i=1}^{4}
P_{s,:,i,j}\delta_{n,i}.
\]

Taylor’s theorem gives the finite-radius relation

\[
\left|
\operatorname{PSE}^{\rho}_{s,n}
-
\theta_{s,n}
\right|
\le
\frac{\rho^2}{6}
\sup_{|t|\le\rho}
|g'''_{s,n}(t)|.
\]

Thus the theorem identifies the infinitesimal selected-path coefficient, while the independent target tests whether it predicts a frozen nonzero-radius path intervention. The empirical question is not an algebraic same-endpoint equality.

The tensor-item cell predictor is

\[
\widehat{\operatorname{PSE}}_{s,c}
=
\frac1{|N_c|}
\sum_{n\in N_c}
\widehat\theta_{s,n},
\]

\[
\boxed{
\widehat T^{\mathrm{MB}}_c
=
\left|
\widehat{\operatorname{PSE}}_{\mathrm{pat},c}
-
\widehat{\operatorname{PSE}}_{\mathrm{tar},c}
\right|,
}
\]

where \(N_c\) and \(E_c\) are disjoint eight-item sets.

## Independence firewall

The target is implemented in `src/green_bridge_path_target.py`. That module may import only:

- the frozen specification;
- model-loading utilities;
- tensor dataclasses;
- generic hashing and serialization utilities.

It must not import:

- `mixed_path_identification.py`;
- `matched_bypass_gate.py`;
- `green_bridge_tail.py`;
- predictor finite-difference code;
- predictor score code;
- baseline code.

The target:

- uses disjoint energy items;
- uses joint ten-gate interventions rather than ten one-gate inverses;
- uses direct finite path endpoints rather than \(H^P-H^C\);
- subtracts the residual bypass explicitly;
- never reads \(\widehat A,\widehat G,\widehat C,\widehat H,\widehat P\);
- is computed before loading predictor result files in the analysis process.

As an implementation and locality audit, `torch.func.jvp` independently computes \(g'_{s,n}(0)\). It is compared with the target-only Richardson derivative

\[
D^{\mathrm{target}}_{s,n}
=
\frac{
4\operatorname{PSE}^{1/2}_{s,n}
-
\operatorname{PSE}^{1}_{s,n}
}{3}.
\]

The JVP is an audit, not the primary target and not an input to \(\widehat T^{\mathrm{MB}}\).

# Probe-Complete Finite Design

## Fixed dimensions and coordinates

\[
r_1=4,\qquad r_2=10,\qquad k=100.
\]

The residual mediator is represented by the four columns of \(U\). The gate mediator uses the ten literal MLP-10 indices in \(J\). No gate rotation, learned selector, ridge penalty, or confirmatory selection is permitted.

The marginal designs are

\[
\mathcal X=\{e_1,e_2,e_3,e_4\},
\qquad
\mathcal Z=\{e_j:j\in J\}.
\]

The paired mixed design is

\[
\mathcal K
=
\{e_i\otimes e_j:
i=1,\ldots,4,\ j\in J\}.
\]

Under the canonical vectorization ordering, the \(40\times40\) design matrix is a permutation of \(I_{40}\). Therefore

\[
\operatorname{rank}(\mathcal X)=4,\qquad
\operatorname{rank}(\mathcal Z)=10,\qquad
\operatorname{rank}(\mathcal K)=40,
\]

and every selected tensor coordinate is directly probed. No regression inverse is needed.

## Residual basis construction

Use 512 basis-donor pairs, entirely disjoint from the 512 radius-donor pairs and all evaluation items. For every basis pair, form the final-position chord

\[
d_n
=
R_{\mathrm{clean},n}[\pi,:]
-
R_{\mathrm{corrupt},n}[\pi,:].
\]

Stack the uncentered matrix

\[
D_U\in\mathbb R^{512\times768}.
\]

Compute, on CPU in float64 and with one BLAS thread,

\[
D_U=L\Sigma V^\top
\]

using `scipy.linalg.svd(full_matrices=False, lapack_driver="gesvd")`. Set

\[
U=V[:,0:4].
\]

For each column, choose its sign so that its largest-magnitude coordinate is positive, breaking magnitude ties by the lowest residual index.

The basis is admissible only if

\[
\sigma_4/\sigma_5\ge1.10
\]

and

\[
\sigma_4\ge10^{-4}\sigma_1.
\]

For each of the 16 basis-donor nouns, recompute \(U_{-n}\) after omitting that noun. The largest principal angle between \(U\) and \(U_{-n}\) must be at most \(15^\circ\).

This SVD gives a scientifically meaningful low-dimensional subspace of actual clean–corrupt residual variation immediately upstream of the selected MLP. It does not justify claims about the remaining 764 residual dimensions.

## Frozen radii

From the separate 512-pair radius set, define

\[
\sigma_x
=
\operatorname{median}_n
\frac{\|U^\top d_n\|_2}{\sqrt4},
\qquad
h_1=0.20\,\sigma_x.
\]

For gate \(j\), pool clean and corrupt preactivation anchors over the radius set and define

\[
\sigma_j
=
\max\left\{
1.4826\,\operatorname{MAD}(a_{j}),
\;
\operatorname{median}_n
|a^{\mathrm{clean}}_{nj}-a^{\mathrm{corrupt}}_{nj}|
\right\},
\]

\[
h_{2j}=0.20\,\sigma_j.
\]

The half radii are \(h_1/2\) and \(h_{2j}/2\). No radius search is allowed.

Numerical-support floors are

\[
h_1
\ge
2^{-10}
\operatorname{median}_n
\operatorname{RMS}
\big(R_n[\pi,:]\big),
\]

\[
h_{2j}
\ge
2^{-10}
\max\left\{
1,\operatorname{median}_n|a_{nj}|
\right\}.
\]

Failure of a floor is a technical stop. The radius must not be inflated.

For each of the 16 radius-donor nouns, leave it out and recompute \(h_1,h_{2j}\). Every relative change must be at most \(20\%\).

## Finite-difference estimators

For radius multiplier \(\rho\in\{1,\frac12\}\), write

\[
h_x^\rho=\rho h_1,
\qquad
h_{zj}^\rho=\rho h_{2j}.
\]

The gate response is

\[
\widehat G^\rho_{sj}
=
\frac{
Y^P_{sj}(0,h_{zj}^\rho)
-
Y^P_{sj}(0,-h_{zj}^\rho)
}{
2h_{zj}^\rho
}.
\]

The pure gate curvature is

\[
\widehat C^\rho_{sj}
=
\frac{
Y^P_{sj}(0,h_{zj}^\rho)
-
2Y^P_{sj}(0,0)
+
Y^P_{sj}(0,-h_{zj}^\rho)
}{
(h_{zj}^\rho)^2
}.
\]

The residual response is

\[
\widehat J^{P,\rho}_{sji}
=
\frac{
Y^P_{sj}(h_x^\rho e_i,0)
-
Y^P_{sj}(-h_x^\rho e_i,0)
}{
2h_x^\rho
}.
\]

For \(Q\in\{P,C\}\), the four-corner mixed response is

\[
\widehat H^{Q,\rho}_{sij}
=
\frac{
Y^Q_{sj}(+h_x^\rho e_i,+h_{zj}^\rho)
-
Y^Q_{sj}(+h_x^\rho e_i,-h_{zj}^\rho)
-
Y^Q_{sj}(-h_x^\rho e_i,+h_{zj}^\rho)
+
Y^Q_{sj}(-h_x^\rho e_i,-h_{zj}^\rho)
}{
4h_x^\rho h_{zj}^\rho
}.
\]

For every derivative \(Q\), the primary estimate is Richardson extrapolation:

\[
\widehat Q^R
=
\frac{4\widehat Q^{1/2}-\widehat Q^1}{3}.
\]

The inverse is applied after extrapolation:

\[
\widehat A^R_{sji}
=
\frac{
\left\langle
\widehat C^R_{sj},
\widehat H^{P,R}_{sij}
-
\widehat H^{C,R}_{sij}
\right\rangle
}{
\|\widehat C^R_{sj}\|_2^2
},
\]

\[
\widehat P^R_{s,:,i,j}
=
\widehat G^R_{sj}\widehat A^R_{sji}.
\]

Raw full-radius and half-radius inverses are also computed independently for stability analysis.

## Exact response count per tensor item

For one system, one gate, and one radius:

- two \(z\)-axis endpoints for \(G,C\);
- eight \(x\)-axis endpoints for four \(J_i\);
- sixteen path mixed corners;
- sixteen control mixed corners.

Total:

\[
2+8+16+16=42.
\]

Across ten gates and two radii:

\[
42\times10\times2=840.
\]

One center is shared across all gates and radii:

\[
841
\]

tail evaluations per system, and

\[
1682
\]

for target and patched systems together.

## Finite-radius derivative bounds

Let directional derivative suprema be taken over the relevant intervention rectangle.

For a centered first derivative,

\[
\|\widehat G^1-G\|
\le
\frac{h_2^2}{6}M_3.
\]

For a centered second derivative,

\[
\|\widehat C^1-C\|
\le
\frac{h_2^2}{12}M_4.
\]

For a mixed derivative,

\[
\|\widehat H^1-H\|
\le
\frac{h_1^2}{6}M_{31}
+
\frac{h_2^2}{6}M_{13}.
\]

After simultaneous half-radius Richardson extrapolation,

\[
\|\widehat G^R-G\|
\le
\frac{h_2^4}{480}M_5,
\]

\[
\|\widehat C^R-C\|
\le
\frac{h_2^4}{1440}M_6,
\]

and

\[
\|\widehat H^R-H\|
\le
\frac{h_1^4}{480}M_{51}
+
\frac{h_1^2h_2^2}{144}M_{33}
+
\frac{h_2^4}{480}M_{15}.
\]

These are genuine finite-radius bounds in terms of local derivative suprema. Full-versus-half discrepancies are operational diagnostics; they are not represented as theorem-level upper bounds without additional derivative-shape assumptions.

## Numerical-error propagation

Let \(\epsilon_y\) be the maximum per-logit discrepancy found by frozen duplicate evaluations. Richardson noise amplification per coordinate is bounded by

\[
\eta_G^R=\frac{3\epsilon_y}{h_{2j}},
\]

\[
\eta_C^R=\frac{64\epsilon_y}{3h_{2j}^2},
\]

\[
\eta_J^R=\frac{3\epsilon_y}{h_1},
\]

\[
\eta_H^R=\frac{17\epsilon_y}{3h_1h_{2j}}.
\]

The shared center is accounted for in the \(64/3\) coefficient for \(C\).

Define conservative operational errors

\[
\epsilon_G
=
\|\widehat G^R-\widehat G^{1/2}\|_2
+
\sqrt{k}\eta_G^R,
\]

\[
\epsilon_C
=
\|\widehat C^R-\widehat C^{1/2}\|_2
+
\sqrt{k}\eta_C^R,
\]

\[
\epsilon_{\Delta H,i}
=
\|
\widehat{\Delta H}^{R}_i
-
\widehat{\Delta H}^{1/2}_i
\|_2
+
2\sqrt{k}\eta_H^R.
\]

When

\[
\|\widehat C^R\|_2>\epsilon_C,
\]

let

\[
A_{\max,i}
=
\frac{
\|\widehat{\Delta H}^R_i\|_2+\epsilon_{\Delta H,i}
}{
\|\widehat C^R\|_2-\epsilon_C
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
\|\widehat C^R\|_2
},
\]

and

\[
\epsilon_{P,i}
=
\epsilon_GA_{\max,i}
+
\|\widehat G^R\|_2\epsilon_{A,i}.
\]

The gate-level Frobenius error is

\[
\epsilon_{P,j,F}
=
\left(
\sum_{i=1}^{4}\epsilon_{P,i}^2
\right)^{1/2}.
\]

For an item-direction contraction,

\[
\epsilon_{\theta,s,n}
\le
\|\ell_y\|_2
\|\delta_n\|_2
\sum_{j\in J}\epsilon_{P,sj,F}.
\]

## Independent white-box audit of \(A\)

For a residual vector \(r\), write

\[
\mu=\frac1d\mathbf 1^\top r,
\qquad
\bar r=r-\mu\mathbf1,
\qquad
s^2=\frac1d\|\bar r\|_2^2+\epsilon_{\mathrm{LN}}.
\]

For learned LayerNorm scale \(w\),

\[
J_{\mathrm{LN}}(r)
=
\frac{\operatorname{diag}(w)}{s}
\left[
I-\frac1d\mathbf1\mathbf1^\top
-
\frac{\bar r\bar r^\top}{ds^2}
\right].
\]

The exact architecture-derived audit value is

\[
A^{\mathrm{WB}}_{sj:}
=
W_{\mathrm{in}}[:,j]^\top
J_{\mathrm{LN}}(R_s[\pi,:])U.
\]

It is computed in float64 from cached tensors. It is used only to audit the response inverse and to bound an effectively null gate. It must never be substituted for \(\widehat A\) in the primary tensor predictor.

# Frozen Greater-Than Experiment

## Model and numerical environment

The model is `openai-community/gpt2` at revision

```text
607a30d783dfa663caf39e06633721c8d4cfcd7e
```

loaded in float32 with no TransformerLens weight processing. The Hugging Face revision and TransformerLens release/commit are pinned rather than resolved by an unversioned alias. 

Frozen environment:

```text
Python              3.11.13
PyTorch             2.7.1+cu126
CUDA runtime        12.6
TransformerLens     3.6.0
TransformerLens git 4a4dc26
transformers        5.13.0
NumPy               2.2.6
SciPy               1.15.3
pandas              2.2.3
pyarrow             19.0.1
```

The lockfile must resolve exactly. Failure is a technical stop; versions may not be silently substituted.

Runtime configuration:

- `model.eval()`;
- float32 model computation;
- no autocast;
- no TF32;
- `torch.backends.cuda.matmul.allow_tf32=False`;
- `torch.backends.cudnn.allow_tf32=False`;
- `torch.use_deterministic_algorithms(True)`;
- fixed CUDA and NumPy seeds;
- selected 100-logit vectors cast to float64 before finite-difference arithmetic;
- no dropout;
- no generation cache.

The runtime must assert:

```text
n_layers = 12
d_model = 768
n_heads = 12
d_mlp = 3072
normalization_type = LN
activation = gelu_new or the exact loaded equivalent
layer_norm_epsilon = 1e-5
```

Any mismatch is a stop.

## Prompt

The literal prompt is

```text
<|endoftext|> The {noun} lasted from the year {cc:02d}{y:02d} to the year {cc:02d}
```

with:

```text
add_special_tokens=False
default_prepend_bos=False
```

The original task code uses the same sentence family, optionally prepends the literal end-of-text token, requires a valid year to tokenize into a century token plus a two-digit suffix token, and evaluates the final prompt position. 

For each \(q=00,\ldots,99\), define

```python
ids = tokenizer.encode(f"{q:02d}", add_special_tokens=False)
```

and require:

- exactly one token;
- exact decode to the same two digits;
- all 100 IDs unique.

For a start year, require

```python
tokenizer.encode(f" {cc:02d}{y:02d}", add_special_tokens=False)
```

to have exactly two tokens, with its second token equal to \(\nu_y\). Also require

```python
tokenizer.encode(f" {cc:02d}", add_special_tokens=False)
```

to contain exactly one token and to equal the final prompt token.

Clean and corrupted prompts must have identical token length and the same final-position index. No padding is used within an item. Batched items are grouped by exact sequence length.

## Evaluation cells

Evaluation nouns, in frozen index order:

```text
0 campaign
1 dynasty
2 reign
3 siege
4 treaty
5 warfare
6 expedition
7 kingdom
```

Evaluation centuries:

```text
12
14
16
```

Threshold-distance bins:

```text
near: 8 <= |y-y'| <= 16
far:  40 <= |y-y'| <= 56
```

Candidate suffixes satisfy:

```text
05 <= y,y' <= 94
```

and both full years must satisfy the tokenization contract.

A semantic cell is

\[
(\text{noun},\text{century},\text{distance bin}).
\]

There are

\[
8\times3\times2=48
\]

cells.

Let \(t\) be the noun index and \(c\in\{0,1,2\}\) the century index. A cell is development exactly when

\[
t\bmod3=c.
\]

This gives 16 development cells, eight per distance bin. The remaining 32 cells are untouched confirmation cells, 16 per bin. The repository’s earlier review required 48 held-out semantic cells split 16/32 and cell-level resampling. 

Each cell contains:

- eight tensor-fit items;
- eight disjoint energy-target items.

Each role has exactly four orientations with \(y'>y\) and four with \(y'<y\). No unordered pair may appear in both roles.

## Donor cells

Donor nouns:

```text
invasion
insurgency
rivalry
hostility
raids
sanctions
domination
confrontation
pilgrimage
journey
voyage
operation
outbreak
reforms
relationship
modernization
```

Donor centuries:

```text
11
13
15
17
```

For every donor noun, century, and distance bin, select:

- four basis pairs;
- four disjoint radius pairs.

Within each role, exactly two pairs have \(y'>y\) and two have \(y'<y\).

Thus:

\[
16\times4\times2\times4=512
\]

basis pairs and 512 radius pairs.

Donor prompts are never used for development calibration, tensor scoring, energy targets, or confirmation.

## Deterministic pair selection

The global salt is

```text
idle1-gt-bridge-20260805
```

For each cell and role, enumerate unordered pairs

\[
5\le a<b\le94
\]

that satisfy the tokenization and bin rules. Rank them by the ascending hexadecimal SHA-256 digest of

```text
{salt}|pair|{noun}|{cc:02d}|{bin}|{role}|{a:02d}|{b:02d}
```

The preferred orientation is the low bit of the first byte of

```text
SHA256(
  "{salt}|orient|{noun}|{cc:02d}|{bin}|{role}|{a:02d}|{b:02d}"
)
```

Accept the preferred orientation if its quota remains; otherwise accept the reverse orientation if that quota remains; otherwise continue. Stop once the exact role quota is filled. Pairs used by an earlier role in the same cell are excluded.

The role order is:

```text
evaluation: tensor, energy
donor:      basis, radius
```

Failure to fill any quota is a preflight stop. No replacement is permitted after model responses are observed.

## Three systems

For every item:

- **Target:** clean prompt containing \(y\).
- **Corrupt:** replace only \(y\) by the paired \(y'\) in the same century.
- **Patched:** run the corrupted prompt but replace only  
  `blocks.8.hook_mlp_out[:, π, :]`  
  with the clean prompt’s vector.

All other positions and components remain corrupted in the patched system. MLP 8 is only a frozen comparator intervention; it is not a mediator in the identification theorem. This choice is mechanistically motivated by the published finding that MLP 8 is an upstream contributor to the MLP 9–11 Greater-Than circuit, while MLP 10 has both direct and via-MLP-11 effects. 

No-op audits:

- corrupt output patched with its own MLP-8 vector;
- clean output patched with its own MLP-8 vector.

The maximum absolute error over the 100 selected logits must be at most \(2\times10^{-5}\).

## Fair baselines

All baselines use the same tensor items, anchors, \(U\), radii, selected gates, year logits, and cell split. The required baseline families are those frozen in the prior review: single interaction direction, equally budgeted first-order responses, PIE/INT-style factorial decomposition, and behavior alone. 

Define the common joint-response map

\[
\widetilde Y_s(x,z)
=
F_s\!\left(
Ux+
\sum_{j\in J}
c_j
\left[
\psi(a_{sj}(x)+z_j)-\psi(a_{sj})
\right]
\right),
\]

with selected gates live and the other 3062 post-GELU coordinates anchored. Unlike the independent target, this common baseline map retains the \(Ux\) residual route.

Let

\[
m_{s,n}(x,z)=\ell_y^\top\widetilde Y_s(x,z).
\]

### Behavioral restoration

\[
S^{\mathrm{beh}}_n
=
|m_{\mathrm{pat},n}(0,0)-m_{\mathrm{tar},n}(0,0)|.
\]

Centers are shared with the tensor and first-order computations.

### Single clean–corrupt interaction direction

Let

\[
v_{nj}
=
\frac{
a^{\mathrm{tar}}_{nj}-a^{\mathrm{cor}}_{nj}
}{
h_{2j}
}.
\]

If \(\|v_n\|_2>10^{-12}\), set

\[
\zeta_{nj}
=
h_{2j}\frac{v_{nj}}{\|v_n\|_2};
\]

otherwise set \(\zeta_n=0\).

For \(\rho\in\{1,\frac12\}\),

\[
R^{\rho}_{s,n}
=
\frac{
m_s(\rho\delta_n,\rho\zeta_n)
-
m_s(-\rho\delta_n,-\rho\zeta_n)
}{
2\rho
},
\]

\[
R^R_{s,n}
=
\frac{4R^{1/2}_{s,n}-R^1_{s,n}}3,
\]

\[
S^{\mathrm{single}}_n
=
|R^R_{\mathrm{pat},n}-R^R_{\mathrm{tar},n}|.
\]

This requires eight conceptual calls per item. Its endpoints are a subset of the PIE endpoint cache.

### Equally budgeted first-order-only baseline

Construct 200 deterministic unit vectors \(w_a\in\mathbb R^4\):

- \(w_1,\ldots,w_4=e_1,\ldots,e_4\);
- the remaining 196 are normalized standard-normal vectors generated by NumPy `PCG64`;
- the seed integer is the first eight big-endian bytes of  
  `SHA256("idle1-gt-bridge-20260805:first-order")`;
- make the first nonzero coordinate positive;
- reject a proposal when its absolute inner product with an accepted vector exceeds `0.999999`.

Hash the final \(200\times4\) matrix.

Residual first-order responses:

\[
R^{x,\rho}_{s,a}
=
\frac{
m_s(\rho h_1w_a,0)-m_s(-\rho h_1w_a,0)
}{
2\rho
}.
\]

Gate first-order responses:

\[
R^{z,\rho}_{s,j}
=
\frac{
m_s(0,\rho h_{2j}e_j)-m_s(0,-\rho h_{2j}e_j)
}{
2\rho
}.
\]

Apply Richardson separately. The item score is

\[
S^{\mathrm{FO}}_n
=
\left[
\frac1{200}
\sum_a
\left(
R^{x,R}_{\mathrm{pat},a}
-
R^{x,R}_{\mathrm{tar},a}
\right)^2
+
\frac1{10}
\sum_j
\left(
R^{z,R}_{\mathrm{pat},j}
-
R^{z,R}_{\mathrm{tar},j}
\right)^2
\right]^{1/2}.
\]

Its exact budget is

\[
2\text{ systems}
\times
2\text{ radii}
\times
2\text{ signs}
\times
(200+10)
+
2\text{ centers}
=
1682,
\]

equal to the mixed estimator.

### PIE/INT-style scalar factorial baseline

For \(\rho\in\{1,\frac12\}\),

\[
I^\rho_s
=
\frac{
m_s(+\rho\delta,+\rho\zeta)
-
m_s(+\rho\delta,-\rho\zeta)
-
m_s(-\rho\delta,+\rho\zeta)
+
m_s(-\rho\delta,-\rho\zeta)
}{
4\rho^2
}.
\]

Apply Richardson and define

\[
S^{\mathrm{PIE}}_n
=
|I^R_{\mathrm{pat},n}-I^R_{\mathrm{tar},n}|.
\]

The same factorial endpoints define first-order block terms

\[
B^{x,\rho}_s
=
\frac{
m_{++}+m_{+-}-m_{-+}-m_{--}
}{
4\rho
},
\]

\[
B^{z,\rho}_s
=
\frac{
m_{++}-m_{+-}+m_{-+}-m_{--}
}{
4\rho
}.
\]

Their Richardson versions define the preregistered cancellation subset.

The PIE design uses 16 conceptual calls per item. Its `++` and `--` endpoints are reused by the single-direction baseline, so the unique shared factorial endpoint count is 16, not 24.

### Cell aggregation

For each baseline \(b\),

\[
S^b_c
=
\frac1{|N_c|}
\sum_{n\in N_c}S^b_n.
\]

Only the four baselines receive development calibration. The mixed tensor predictor remains in its theorem-derived logit units with identity calibration.

# TransformerLens Hook Contract

## Exact hooks

| Hook | Shape | Required mutation |
|---|---|---|
| `blocks.8.hook_mlp_out` | `[B,L,768]` | Patched system only: replace `[b,π_b,:]` with clean cached vector. |
| `blocks.10.hook_resid_mid` | `[B,L,768]` | Add `U @ x[b]` only at `[b,π_b,:]`. |
| `blocks.10.mlp.hook_pre` | `[B,L,3072]` | Add selected scalar \(z\) at `[b,π_b,j]`; no other mutation. |
| `blocks.10.mlp.hook_post` | `[B,L,3072]` | Apply the path/control/joint clamping rules below. |
| `blocks.10.hook_resid_post` | `[B,L,768]` | Independent target only: subtract `U @ x[b]` at `[b,π_b,:]`. |
| model output | `[B,L,d_vocab]` | Gather `[b,π_b,ν_q]` for all \(q\). |

Every hook must:

- clone before mutation;
- use per-example final-position metadata;
- act exactly once;
- leave other batch entries, positions, and coordinates unchanged;
- record an invocation counter;
- be removed through `model.reset_hooks()` in `finally`;
- pass an untouched-entry maximum-difference audit of \(10^{-7}\).

## `hook_post` modes

### One-gate path mode

At position \(\pi\):

- gate \(j\) retains the exact model-computed value  
  \(\psi(a_{sj}(x)+z)\);
- every \(k\neq j\) is overwritten with the cached system anchor  
  \(\psi(a_{sk})\).

### One-gate matched-control mode

At position \(\pi\):

- gate \(j\) is overwritten with  
  `model.blocks[10].mlp.act_fn(anchor_pre_j + z)`;
- every \(k\neq j\) is overwritten with its cached post-GELU anchor.

Thus \(x\) remains in the residual stream but cannot change any gate.

### Joint target and baseline mode

At position \(\pi\):

- the ten coordinates in \(J\) retain their exact live values, after any declared \(z_j\) addition;
- the other 3062 coordinates are overwritten with system anchors.

All other positions remain unchanged.

The activation must never be reimplemented with an approximate formula. The actual loaded `act_fn` is called.

## Model loading

```python
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformer_lens import HookedTransformer
import torch

REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

tokenizer = AutoTokenizer.from_pretrained(
    "openai-community/gpt2",
    revision=REVISION,
    use_fast=True,
)

hf_model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2",
    revision=REVISION,
    torch_dtype=torch.float32,
)
hf_model.eval()

model = HookedTransformer.from_pretrained_no_processing(
    "gpt2",
    hf_model=hf_model,
    tokenizer=tokenizer,
    device="cuda",
    dtype=torch.float32,
    default_prepend_bos=False,
    default_padding_side="right",
)
model.eval()
```

`from_pretrained_no_processing` disables LayerNorm folding, weight centering, unembedding centering, factored-attention refactoring, and value-bias folding. 

## Manual tail contract

The high-volume estimator must not call the full transformer for every endpoint. It continues from cached block-10 `resid_mid` anchors:

1. clone full-sequence \(R_s\);
2. add \(E_\pi Ux\);
3. call the exact block-10 `ln2`;
4. compute MLP preactivations using TransformerLens’s `batch_addmm` order;
5. add \(z\);
6. call the exact `act_fn`;
7. apply the declared `hook_post` clamping rule;
8. apply the exact MLP-10 output projection and bias;
9. add the result to the edited `resid_mid`;
10. continue through the complete block 11;
11. call the exact final LayerNorm;
12. call the exact unembedding and gather the 100 suffix logits.

The MLP source uses fused `batch_addmm` specifically to match the model’s parameter layout and Hugging Face computation. 

Do not call

```python
model(..., start_at_layer=10)
```

on `resid_mid` data. `start_at_layer=10` is inclusive and expects the residual before block 10; it would rerun block-10 attention and create the wrong DAG. 

## Separate target tail

`green_bridge_path_target.py` contains a separate pure function that implements the joint target curve and is compatible with `torch.func.jvp`. It must not share path/control constructors, inverse functions, or response endpoint tables with the predictor.

Code-structure tests must inspect its import graph and fail if it imports any prohibited predictor module.

## Numerical audits before scientific scores

### Hugging Face versus TransformerLens

On 32 hash-selected donor prompts, compare the 100 final-position year logits from:

- native Hugging Face GPT-2;
- unmodified TransformerLens GPT-2.

Maximum absolute discrepancy must be at most

\[
2\times10^{-5}.
\]

### Manual tail versus full-hook execution

Use 32 hash-selected donor conditions:

```text
8 center
8 x-only
8 z-only
4 path-mixed
4 control-mixed
```

For every condition:

- maximum absolute year-logit discrepancy \(\le2\times10^{-5}\);
- derivative-vector relative discrepancy \(\le10^{-4}\), with denominator  
  \(\max(\|v\|_2,10^{-5})\).

### Center replay

For every tensor item and system, the clamped tail at \(x=z=0\) must reproduce the cached natural 100-logit vector with:

\[
\operatorname{RMS}\le2\times10^{-6},
\qquad
\|\cdot\|_\infty\le2\times10^{-5}.
\]

### Patch no-op

Clean-to-clean and corrupt-to-corrupt block-8 patches must have maximum year-logit error at most \(2\times10^{-5}\).

### Fallback rule

Full-hook endpoint evaluation may replace the manual tail only when:

- all full-hook audits pass;
- the exact frozen interventions remain unchanged;
- the 2% throughput preflight demonstrates completion under the hardware cap.

Otherwise the run stops. Precision, sites, gate set, radius, and target may not be changed.

# Statistical Analysis and Decision Thresholds

## Frozen finite population

The scientific population is the declared finite set of 48 semantic cells and their deterministic tensor/energy items. There is no claim of automatic generalization to arbitrary templates, years, models, or tasks.

The 16 development cells may be used only for:

- baseline calibration;
- numerical threshold validation;
- deciding whether confirmation is opened.

The 32 confirmation cells remain unopened until `frozen_analysis.json`, all source hashes, split hashes, basis/radius hashes, and the complete manifest are written.

## Numerical noise protocol

Before confirmation, repeat exactly:

- 64 hash-lowest full-model anchor conditions from the donor/development plan;
- 32 hash-lowest development tail corners.

Define

\[
\epsilon_y^{\mathrm{dev}}
=
\max\left\{
10^{-7},
\text{maximum absolute duplicate-logit discrepancy}
\right\}.
\]

Freeze this value before confirmation.

During confirmation, repeat exactly:

- 32 hash-lowest confirmation full-model anchors;
- 32 hash-lowest confirmation tail corners.

The run stops if confirmation duplicate discrepancy exceeds

\[
\max\{2\epsilon_y^{\mathrm{dev}},2\times10^{-6}\}.
\]

## Gate-level structural admissibility

For each tensor item, system \(s\in\{\mathrm{tar},\mathrm{pat}\}\), and gate \(j\), classify the gate as **active-identified**, **certified target-null**, or **invalid**.

### Active-identified gate

All conditions must hold.

1. **Curvature magnitude**
   \[
   \frac{\|\widehat C^R_{sj}\|_2}{\sqrt{100}}
   \ge5\times10^{-4},
   \qquad
   \|\widehat C^R_{sj}\|_2\ge20\epsilon_C.
   \]

2. **Gate-response magnitude**
   \[
   \frac{\|\widehat G^R_{sj}\|_2}{\sqrt{100}}
   \ge5\times10^{-4},
   \qquad
   \|\widehat G^R_{sj}\|_2\ge20\epsilon_G.
   \]

3. **Factorization residual**
   \[
   r_{\mathrm{fac}}
   =
   \frac{
   \left(
   \sum_i
   \|
   \widehat{\Delta H}^R_i
   -
   \widehat C^R\widehat A^R_i
   \|_2^2
   \right)^{1/2}
   }{
   \max\left\{
   \left(
   \sum_i\|\widehat{\Delta H}^R_i\|_2^2
   \right)^{1/2},
   10^{-8}
   \right\}
   }
   \le0.15.
   \]

4. **White-box \(A\) audit**
   \[
   \frac{
   \|\widehat A^R-A^{\mathrm{WB}}\|_2
   }{
   \max\{\|A^{\mathrm{WB}}\|_2,10^{-6}\}
   }
   \le0.05
   \]
   or, when \(\|A^{\mathrm{WB}}\|_2<10^{-6}\),
   \[
   \|\widehat A^R-A^{\mathrm{WB}}\|_2\le10^{-4}.
   \]

5. **Finite-radius tensor stability**
   \[
   \cos(\widehat P^1,\widehat P^{1/2})\ge0.95,
   \]
   \[
   \frac{
   2\|\widehat P^1-\widehat P^{1/2}\|_F
   }{
   \|\widehat P^1\|_F+\|\widehat P^{1/2}\|_F+10^{-12}
   }
   \le0.25,
   \]
   \[
   \frac{
   \|\widehat P^R-\widehat P^{1/2}\|_F
   }{
   \max\{\|\widehat P^R\|_F,10^{-8}\}
   }
   \le0.25,
   \]
   and
   \[
   \|\widehat P^R\|_F\ge20\epsilon_{P,F}.
   \]

### Certified target-null gate

A gate that fails the curvature condition may be treated as target-null only when all of the following hold:

\[
\|\widehat G^R\|_2\le5\epsilon_G,
\]

\[
\|\ell_y\|_2\|\delta_n\|_2
\left(
\|\widehat G^R\|_2+\epsilon_G
\right)
\|A^{\mathrm{WB}}\|_2
\le0.005
\]

logit units, and the corresponding full-versus-half upper-bound change is at most \(0.005\) logit.

For such a gate, \(A\) is not claimed identified. Its selected-target contribution is reported only as bounded below the frozen relevance floor.

### Invalid gate

Every gate that is neither active-identified nor certified target-null is invalid. It may not be silently omitted.

## Tensor-item admissibility

A tensor item is admissible only if:

- all ten gates are active-identified or certified target-null in both target and patched systems;
- at least three gates are active-identified in each system;
- the projected chord satisfies \(\|q_n\|\ge0.10\sigma_x\);
- every center, hook, tail, and no-op audit passes;
- the recovered direct bypass is consistent across active gates.

For bypass consistency, define

\[
\widehat D_{sji}
=
\widehat J^P_{sji}
-
\widehat P_{s,:,i,j}
\]

and average over active \(j\). The relative root-mean-square disagreement across \(i,j\) must be at most \(0.15\).

A cell retains its tensor score only if at least six of its eight tensor items are admissible. The same surviving items are used by every predictor and baseline.

## Energy-item admissibility

An energy item is admissible only if:

- \(\|q_n\|\ge0.10\sigma_x\);
- all target-tail and no-op audits pass;
- for all three systems,
  \[
  |D^{\mathrm{target}}_{s,n}-g'^{\mathrm{JVP}}_{s,n}|
  \le0.01
  \]
  and
  \[
  \frac{
  |D^{\mathrm{target}}_{s,n}-g'^{\mathrm{JVP}}_{s,n}|
  }{
  \max\{|g'^{\mathrm{JVP}}_{s,n}|,0.05\}
  }
  \le0.05;
  \]
- the full-radius target remains local:
  \[
  |
  \operatorname{PSE}^1_{s,n}
  -
  D^{\mathrm{target}}_{s,n}
  |
  \le
  \max\left\{
  0.02,\,
  0.25|D^{\mathrm{target}}_{s,n}|
  \right\}
  \]
  for target, patched, and corrupt systems.

A cell retains its energy target only if at least six of eight energy items are admissible.

No failed item or cell is replaced.

## Global survival requirements

Technical stop before confirm if:

- fewer than 40 of 48 cells survive;
- fewer than 15 of 16 development cells survive;
- fewer than 28 of 32 confirmation cells survive;
- either confirmation bin has fewer than 14 surviving cells.

Oral-level confirmation additionally requires at least:

\[
29/32
\]

confirmation cells and at least 14 in each bin. The original review required at least 90% conditioned confirmation cells and set 40-total/28-confirmation as hard survival floors. 

## Donor stability gates

Before any development score is interpreted:

- \(\sigma_4/\sigma_5\ge1.10\);
- \(\sigma_4/\sigma_1\ge10^{-4}\);
- every leave-one-basis-noun-out principal angle \(\le15^\circ\);
- every leave-one-radius-noun-out radius change \(\le20\%\).

For target-basis stability, for each of the 16 omitted basis nouns:

1. recompute \(U_{-n}\);
2. use the first hash-ranked energy item in each development cell;
3. compute finite-radius target/patch path effects;
4. form the 16 cell mismatches.

Against the full-basis vector, every omission must have:

\[
\operatorname{Spearman}\ge0.90
\]

and median symmetric relative change at most \(20\%\), using denominator

\[
\max\left\{
\frac{|x|+|y|}{2},
0.05
\right\}.
\]

## Cell conditioning

Let \(s_{\mathrm{dev}}\) be the sample standard deviation across surviving development cells of the signed clean–corrupt cell path gaps

\[
\operatorname{PSE}_{\mathrm{tar},c}
-
\operatorname{PSE}_{\mathrm{cor},c}.
\]

A cell is target-conditioned exactly when

\[
C_c\ge0.10
\]

logit-margin units or

\[
C_c\ge0.25s_{\mathrm{dev}}.
\]

The development set must contain at least 15 conditioned cells to proceed. Confirmation success requires at least 29 conditioned cells.

## Predictor uncertainty and development SNR

For tensor cell \(c\), aggregate the item contraction-error bounds conservatively:

\[
E_c
=
\frac1{|N_c|}
\sum_{n\in N_c}
\left(
\epsilon_{\theta,\mathrm{pat},n}
+
\epsilon_{\theta,\mathrm{tar},n}
\right).
\]

Define

\[
\operatorname{SNR}_c
=
\frac{
\widehat T^{\mathrm{MB}}_c
}{
\max\{E_c,10^{-8}\}
}.
\]

At least 10 of 16 development cells must have

\[
\operatorname{SNR}_c\ge3.
\]

The earlier decision gate required mixed-response SNR at least three in 60% of development cells and half-radius rank correlation at least 0.90. 

## Baseline calibration

For each baseline \(b\), fit on development cells

\[
T_c\approx\alpha_b+\beta_b S^b_c,
\qquad
\alpha_b,\beta_b\ge0
\]

using `scipy.optimize.nnls` on the two-column design `[1, S_b]`. There is no ridge penalty.

Development comparison uses 16-fold leave-one-cell-out calibration. After the development decision, fit each baseline once on all surviving development cells and freeze \((\alpha_b,\beta_b)\).

The mixed predictor receives no fitted intercept, slope, or normalization.

## Development decision

All architecture, numerical, donor, conditioning, and SNR gates must pass first.

Let

\[
\operatorname{RMSE}_{\mathrm{MB,dev}}
\]

be the raw mixed predictor RMSE and

\[
\operatorname{RMSE}_{\mathrm{best,dev}}
=
\min_b
\operatorname{RMSE}^{\mathrm{LOOCV}}_{b,\mathrm{dev}}.
\]

Define development gain

\[
\Delta_{\mathrm{dev}}
=
1-
\frac{
\operatorname{RMSE}_{\mathrm{MB,dev}}
}{
\operatorname{RMSE}_{\mathrm{best,dev}}
}.
\]

Decision:

- \(\Delta_{\mathrm{dev}}<0.05\): terminate the oral line;
- \(0.05\le\Delta_{\mathrm{dev}}<0.10\): poster-level result; do not open confirmation;
- \(\Delta_{\mathrm{dev}}\ge0.10\): freeze all files and open confirmation.

This reproduces the preregistered development decision boundary rather than using confirmation as a model-selection set. 

## Confirmation RMSE analysis

For each surviving confirmation cell, evaluate the mixed predictor and four frozen baseline predictions.

Define the best baseline RMSE as the minimum over the four complete fixed prediction vectors. The baseline identity is not reselected per cell.

Run 100,000 paired bootstrap replicates with NumPy `PCG64`, seed `20260805`, resampling cells with replacement separately within the near and far bins. Do not refit calibrations.

For each replicate, compute

\[
\Delta_{\mathrm{rel}}
=
1-
\frac{
\operatorname{RMSE}_{\mathrm{MB}}
}{
\min_b\operatorname{RMSE}_b
},
\]

and

\[
\Delta_{\mathrm{abs}}
=
\min_b\operatorname{RMSE}_b
-
\operatorname{RMSE}_{\mathrm{MB}}.
\]

The bootstrap replicate takes the minimum baseline RMSE after resampling, so uncertainty includes which fixed baseline is best.

Overall success requires:

\[
\Delta_{\mathrm{rel}}\ge0.20,
\]

\[
\operatorname{percentile}_{2.5\%}
(\Delta_{\mathrm{rel}})
\ge0.10,
\]

and

\[
\Delta_{\mathrm{abs}}\ge0.01
\]

logit-margin units.

Within each distance bin:

\[
\Delta_{\mathrm{rel,bin}}\ge0.10,
\]

the 95% lower bound must exceed zero, and

\[
\Delta_{\mathrm{abs,bin}}\ge0.005.
\]

These strengthen and operationalize the repository’s frozen requirement of at least 20% RMSE improvement, a lower confidence bound of at least 10%, and success in both distance bins. 

## Cancellation-subset AUROC

For tensor cell \(c\), average the PIE main effects and define

\[
\Delta B_{x,c}
=
B_{x,\mathrm{pat},c}
-
B_{x,\mathrm{tar},c},
\]

\[
\Delta B_{z,c}
=
B_{z,\mathrm{pat},c}
-
B_{z,\mathrm{tar},c}.
\]

The cancellation subset is frozen as cells satisfying

\[
\Delta B_{x,c}\Delta B_{z,c}<0
\]

and

\[
\min\{|\Delta B_{x,c}|,|\Delta B_{z,c}|\}\ge0.05.
\]

The high-mismatch label is

\[
\mathbf1\{T_c\ge0.10\}.
\]

The subset must contain:

- at least eight cells;
- at least three positives;
- at least three negatives;
- at least three cells from each distance bin.

Use raw \(\widehat T^{\mathrm{MB}}_c\) as the detection score. Success requires

\[
\operatorname{AUROC}\ge0.80
\]

and a 95% lower bound of at least

\[
0.70.
\]

Bootstrap within bin-by-class strata, preserving stratum counts. 

## Half-radius robustness

Recompute the complete mixed cell predictor using the raw full-radius inverse and the raw half-radius inverse, without Richardson extrapolation.

Overall and within each distance bin:

\[
\operatorname{Spearman}
\left(
\widehat T^{1},
\widehat T^{1/2}
\right)
\ge0.90.
\]

The median symmetric relative change must be at most \(20\%\), with denominator

\[
\max\left\{
\frac{
|\widehat T^1|+|\widehat T^{1/2}|
}{2},
0.05
\right\}.
\]

Meeting only the point RMSE threshold or only the radius-correlation threshold is not success. 

## Decisive failure

The oral result fails upon any of the following:

- overall confirmatory gain below 20%;
- overall lower confidence bound below 10%;
- failure in either distance bin;
- absolute RMSE reduction below the frozen floor;
- fewer than 29 conditioned/surviving confirmation cells;
- cancellation-subset size or class-balance failure;
- cancellation AUROC or lower-bound failure;
- half-radius rank or magnitude-stability failure;
- a baseline matching or beating the mixed predictor within uncertainty;
- any post hoc change to model, hook, layer, gate set, basis dimension, corruption, target, radius, cell, threshold, or calibration;
- target implementation depending on predictor code or endpoints;
- structural signal disappearing after bypass control;
- unresolved manual-tail or model-equivalence discrepancy.

After failure there is no alternate corruption, new layer, larger model, radius sweep, gate replacement, or exploratory server branch.

# Compute and Early-Stop Budget

## Exact forward-equivalent accounting

One “tail evaluation” means one system/item/intervention condition returning the 100 selected logits from the cached block-10 anchor. One JVP invocation is counted as two tail-forward equivalents. Full-model calls are listed separately and conservatively costed as six tail equivalents for the hard-cap calculation.

### Scientific core

| Component | Tail evaluations | JVP invocations | Full-model evaluations | Effective units |
|---|---:|---:|---:|---:|
| 512 basis-donor pairs + 512 radius-donor pairs, clean/corrupt anchors | 0 | 0 | 2,048 | 2,048 |
| 384 tensor items: target/corrupt/patched anchors | 0 | 0 | 1,152 | 1,152 |
| 384 tensor items: mixed estimator, equal-budget FO baseline, shared PIE/single cache | 1,297,920 | 0 | 0 | 1,297,920 |
| 384 energy items: target/corrupt/patched anchors | 0 | 0 | 1,152 | 1,152 |
| 384 energy items: four target endpoints for each of three systems | 4,608 | 0 | 0 | 4,608 |
| 384 energy items: JVP audit for three systems | 0 | 1,152 | 0 | 2,304 |
| 16 leave-one-basis-noun-out bases × 16 development cells × two systems × two full-radius signs | 1,024 | 0 | 0 | 1,024 |
| **Core total** | **1,303,552** | **1,152** | **4,352** | **1,310,208** |

The 384 tensor-item unique tail count is

\[
384\times
(1682_{\mathrm{mixed}}
+
1682_{\mathrm{FO}}
+
16_{\mathrm{factorial}})
=
1,297,920.
\]

Behavioral centers are shared, and the single-direction endpoints are contained in the 16 factorial endpoints.

### Exact audit overhead

| Audit | Tail evaluations | Full-model equivalents |
|---|---:|---:|
| Hugging Face comparison on 32 planned prompts | 0 | 32 |
| TransformerLens no-op hook audit | 0 | 32 |
| Manual-tail/full-hook comparison on 32 conditions | 32 | 32 |
| Pre-confirm duplicate audit | 32 | 64 |
| Confirmation duplicate audit | 32 | 32 |
| **Audit total** | **96** | **192** |

### Total

\[
\boxed{
1,303,648\ \text{tail evaluations}
}
\]

\[
\boxed{
1,152\ \text{JVP invocations}
}
\]

\[
\boxed{
4,544\ \text{full-model or full-model-equivalent evaluations}
}
\]

Raw invocation count:

\[
\boxed{
1,309,344
}
\]

Tail-equivalent count with JVP costed at two:

\[
\boxed{
1,310,496
}
\]

Conservative hard-budget units with every full evaluation costed at six tail evaluations:

\[
\boxed{
1,333,216
}
\]

Development/donor phase:

\[
439,008
\]

effective units.

Untouched confirmation phase:

\[
871,488
\]

effective units.

## Hardware contract

### RTX 4090, 24 GB

- manual-tail batch target: 512 conditions;
- full-model batch target: 64 prompts;
- target JVP batch target: 64;
- peak allocated-memory ceiling: 20 GB;
- planning wall-clock range: 8–16 GPU hours;
- hard cap: 24 GPU hours.

### A40-class, 48 GB

- manual-tail batch target: 1024 conditions;
- full-model batch target: 128 prompts;
- target JVP batch target: 128;
- peak allocated-memory ceiling: 32 GB;
- planning wall-clock range: 14–28 GPU hours;
- hard cap: 40 GPU hours.

The wall-clock ranges are planning estimates, not guarantees. A deterministic 2% operation-mixture preflight is drawn by lowest SHA-256 rank from the development plan. Its outputs are retained and reused. Proceed only if measured throughput and memory extrapolate below the relevant hard cap.

The complete run is far below the repository’s earlier ten-GPU-day ceiling, which explicitly required stopping rather than reallocating failed stages to exploratory variants. 

## Ordered early stops

1. Environment or revision mismatch.
2. Tokenization-contract failure.
3. Model-configuration mismatch.
4. Hugging Face/TransformerLens discrepancy.
5. Hook/no-op discrepancy.
6. Manual-tail discrepancy.
7. Missing donor quota.
8. Basis spectral or leave-one-noun stability failure.
9. Radius floor or radius-stability failure.
10. Two-percent throughput or memory failure.
11. Development duplicate-noise failure.
12. Fewer than 15 development cells survive.
13. Fewer than 15 development cells are target-conditioned.
14. Mixed SNR gate failure.
15. Half-radius development stability failure.
16. Development RMSE gain below 10%.
17. Manifest/hash/freeze failure.
18. Confirmation duplicate-noise failure.
19. Confirmation survival/conditioning failure.
20. Any decisive confirmatory threshold failure.

A stop at any point leaves later operations unexecuted.

# Claim Boundary

A successful run would support exactly the following claim:

> In the pinned GPT-2-small Greater-Than setup, output-response interventions at a four-dimensional block-10 residual subspace, combined with matched-bypass controls at ten actual MLP-10 GELU gates, non-tautologically identify a selected local path tensor. That tensor predicts an independently measured finite-radius, edge-isolated path-specific mismatch between a clean target system and an MLP-8-patched corrupted system.

The theorem itself applies to the explicit matched-bypass gate structural class, including arbitrary smooth downstream computation. The empirical instantiation is restricted to:

- GPT-2 small at the pinned revision;
- the frozen year-span prompt family;
- the final prompt position;
- the four-dimensional donor-derived block-10 `resid_mid` subspace;
- the ten declared actual MLP-10 gates;
- the 100 two-digit year logits;
- the declared clean/corrupt/MLP-8-patched systems;
- the frozen local radii and finite cell population.

The following claims remain prohibited:

- complete GPT-2 circuit identification;
- identification of head 9.1, MLP 8, or MLP 9 as simultaneous downstream mediators;
- identification of all 3072 MLP-10 gates;
- identification outside the four-dimensional residual subspace;
- global or large-intervention causal equivalence;
- natural-path uniqueness without the declared clamp policy;
- attention-probability identification;
- recovery of a universal circuit shared by other prompts, models, or tasks;
- proof that MLP-8 patching restores the entire Greater-Than mechanism;
- proof that mixed responses are more computationally efficient than white-box automatic differentiation;
- an “IRS certificate” valid without explicit intervention, target, and probe laws;
- an oral-level empirical claim unless every confirmatory condition passes.

The published Greater-Than evidence is used only to preregister a transformer-relevant topology, MLP-8 comparator, and MLP-10 gate set. It is not treated as proof that the frozen new experiment will pass. The paper itself emphasizes serial MLP dependencies, MLP-10 direct and via-MLP-11 routes, and distributed computation across multiple MLP-10 neurons. 

# Binding Execution Checklist

## Implementation obligations

- [ ] Check out repository commit `126556f`.
- [ ] Do not modify the completed ASG-RDAG estimator or its tests.
- [ ] Add the exact frozen manifest below before any confirmation computation.
- [ ] Pin and hash the model revision, tokenizer, TransformerLens commit, environment lockfile, and repository state.
- [ ] Materialize all donor, development, tensor, energy, and confirmation splits before model responses.
- [ ] Hash every split and deterministic direction matrix.
- [ ] Validate all 100 suffix token IDs.
- [ ] Validate prompt final-position metadata.
- [ ] Validate Hugging Face/TransformerLens logits.
- [ ] Validate block-8 no-op patches.
- [ ] Validate manual tails against full hooks.
- [ ] Build \(U\) only from basis donors.
- [ ] Build radii only from radius donors.
- [ ] Preserve actual MLP-10 gate coordinates.
- [ ] Implement one-gate path and matched-control responses exactly.
- [ ] Implement the independent finite path target in a code-isolated module.
- [ ] Compute the architecture-derived \(A^{\mathrm{WB}}\) only for audits.
- [ ] Never feed \(A^{\mathrm{WB}}\) into the primary mixed predictor.
- [ ] Run the deterministic 2% throughput preflight and reuse its outputs.
- [ ] Write development outputs before reading confirmation prompts.
- [ ] Freeze baseline coefficients and analysis hashes before confirmation.
- [ ] Execute confirmation once.
- [ ] Do not replace invalid cells.
- [ ] Do not change hooks, gates, radii, corruption, bins, target, or thresholds after development.
- [ ] Apply every early-stop condition mechanically.
- [ ] Produce a terminal result JSON containing either success or the first failed gate.

## Required source files

```text
src/green_bridge_spec.py
src/green_bridge_dataset.py
src/matched_bypass_gate.py
src/green_bridge_tail.py
src/green_bridge_path_target.py
src/exp_green_bridge_gpt2.py
src/analyze_green_bridge.py
src/test_green_bridge_contract.py
```

`src/test_green_bridge_contract.py` must include:

- DAG-order tests;
- exact hook-name and shape tests;
- untouched-coordinate tests;
- path/control center-equality tests;
- control severance tests;
- residual-bypass preservation tests;
- target residual-subtraction tests;
- target import-firewall tests;
- finite-design rank tests;
- deterministic split/hash tests;
- pair orientation/quota tests;
- tensor/energy disjointness tests;
- donor/evaluation disjointness tests;
- forward-count tests;
- baseline budget tests;
- calibration no-refit tests;
- confirmation-lock tests;
- terminal-verdict tests.

## Required artifacts

```text
requirements-green-bridge.lock
outputs/green_bridge/manifest.json
outputs/green_bridge/model_fingerprint.json
outputs/green_bridge/splits.json
outputs/green_bridge/donor_basis.npz
outputs/green_bridge/radii.json
outputs/green_bridge/hook_audit.json
outputs/green_bridge/tail_audit.json
outputs/green_bridge/noise_audit_dev.json
outputs/green_bridge/dev_tensor_scores.parquet
outputs/green_bridge/dev_energy_targets.parquet
outputs/green_bridge/dev_result.json
outputs/green_bridge/frozen_analysis.json
outputs/green_bridge/noise_audit_confirm.json
outputs/green_bridge/confirm_tensor_scores.parquet
outputs/green_bridge/confirm_energy_targets.parquet
outputs/green_bridge/result.json
outputs/green_bridge/sha256sums.txt
```

## Complete frozen manifest

```yaml
schema_version: "green-bridge-v1"
repository:
  url: "https://github.com/ScottBlizzard/idle_1"
  branch: "main"
  commit: "126556f"
  output_document: "analysis/GPTPRO_GREEN_BRIDGE_20260805.md"

decision:
  verdict: "GREEN"
  authorized_experiments: 1
  exploratory_followups_after_failure: false
  confirmation_retries: 0

theorem:
  name: "Matched-Bypass Gate Identification"
  upstream_site: "blocks.10.hook_resid_mid"
  gate_pre_site: "blocks.10.mlp.hook_pre"
  gate_post_site: "blocks.10.mlp.hook_post"
  downstream_anchor: "blocks.10.hook_resid_post"
  identity: "H_path - H_control = C * A"
  inverse: "A_i = dot(C, H_path_i - H_control_i) / dot(C, C)"
  path_tensor: "P[:, i, j] = G[:, j] * A[j, i]"
  direct_bypass: "D[:, i] = J_path[:, i, j] - P[:, i, j]"
  hidden_positive_kappa_assumed: false
  complete_cut_assumed: false
  bypass_absence_assumed: false
  faithfulness_assumed: false
  gate_nondegeneracy:
    observable: "C"
    failure_action: "gate invalid unless certified target-null"

environment:
  python: "3.11.13"
  torch: "2.7.1+cu126"
  cuda_runtime: "12.6"
  transformer_lens: "3.6.0"
  transformer_lens_commit: "4a4dc26"
  transformers: "5.13.0"
  numpy: "2.2.6"
  scipy: "1.15.3"
  pandas: "2.2.3"
  pyarrow: "19.0.1"
  dtype_model: "float32"
  dtype_finite_difference: "float64"
  autocast: false
  tf32: false
  deterministic_algorithms: true
  resolver_substitution_allowed: false

model:
  hf_id: "openai-community/gpt2"
  revision: "607a30d783dfa663caf39e06633721c8d4cfcd7e"
  transformer_lens_loader: "from_pretrained_no_processing"
  default_prepend_bos: false
  padding_side: "right"
  expected:
    n_layers: 12
    d_model: 768
    n_heads: 12
    d_mlp: 3072
    normalization: "LN"
    layer_norm_epsilon: 0.00001
  load_processing:
    fold_ln: false
    center_writing_weights: false
    center_unembed: false
    refactor_factored_attn_matrices: false
    fold_value_biases: false
  hash:
    model_state_dict: true
    config: true
    tokenizer_json: true
    vocab: true
    merges: true
    lockfile: true

task:
  prompt: "<|endoftext|> The {noun} lasted from the year {cc:02d}{y:02d} to the year {cc:02d}"
  add_special_tokens: false
  prediction_position: "last token"
  suffix_min: 5
  suffix_max: 94
  output_suffixes: 100
  output_dimension_k: 100
  logit_contrast:
    correct: "mean logits q > clean_y"
    incorrect: "mean logits q <= clean_y"
    use_clean_y_for_all_systems: true
  tokenization:
    suffix_must_be_one_token: true
    suffix_decode_must_be_exact: true
    suffix_ids_must_be_unique: true
    full_year_must_be_two_tokens: true
    full_year_second_token_must_equal_suffix_id: true
    century_prefix_must_be_one_token: true
    clean_corrupt_length_must_match: true

systems:
  target:
    prompt: "clean"
  corrupt:
    prompt: "replace only two-digit suffix y with paired y_prime"
    same_century: true
    preserve_all_non_year_content: true
  patched:
    prompt: "corrupt"
    hook: "blocks.8.hook_mlp_out"
    position: "last"
    patch_value: "clean block-8 MLP output"
    other_positions_patched: false

dimensions:
  residual_dimension: 768
  residual_subspace_r1: 4
  mlp_width: 3072
  selected_gates_r2: 10
  output_dimension_k: 100
  kronecker_dimension: 40

selected_gates:
  layer: 10
  coordinate_system: "actual MLP preactivation coordinates"
  indices:
    - 2326
    - 1138
    - 2287
    - 606
    - 2848
    - 2305
    - 46
    - 2659
    - 946
    - 1616
  data_dependent_replacement_allowed: false
  rotation_allowed: false

evaluation:
  nouns:
    - campaign
    - dynasty
    - reign
    - siege
    - treaty
    - warfare
    - expedition
    - kingdom
  centuries:
    - 12
    - 14
    - 16
  bins:
    near:
      min_abs_difference: 8
      max_abs_difference: 16
    far:
      min_abs_difference: 40
      max_abs_difference: 56
  cells_total: 48
  development_cells: 16
  confirmation_cells: 32
  development_rule: "noun_index mod 3 == century_index"
  tensor_items_per_cell: 8
  energy_items_per_cell: 8
  orientation_per_role:
    y_prime_greater: 4
    y_prime_less: 4
  tensor_energy_pair_overlap_allowed: false
  failed_item_replacement_allowed: false
  failed_cell_replacement_allowed: false

donors:
  nouns:
    - invasion
    - insurgency
    - rivalry
    - hostility
    - raids
    - sanctions
    - domination
    - confrontation
    - pilgrimage
    - journey
    - voyage
    - operation
    - outbreak
    - reforms
    - relationship
    - modernization
  centuries:
    - 11
    - 13
    - 15
    - 17
  bins:
    - near
    - far
  basis_pairs_per_cell: 4
  radius_pairs_per_cell: 4
  basis_pairs_total: 512
  radius_pairs_total: 512
  basis_radius_overlap_allowed: false
  donor_evaluation_overlap_allowed: false
  orientation_per_role:
    y_prime_greater: 2
    y_prime_less: 2

pair_selection:
  salt: "idle1-gt-bridge-20260805"
  ordering: "ascending SHA256 hexadecimal digest"
  pair_key: "{salt}|pair|{noun}|{cc:02d}|{bin}|{role}|{a:02d}|{b:02d}"
  orientation_key: "{salt}|orient|{noun}|{cc:02d}|{bin}|{role}|{a:02d}|{b:02d}"
  orientation_bit: "low bit of first digest byte"
  evaluation_role_order:
    - tensor
    - energy
  donor_role_order:
    - basis
    - radius
  quota_failure_action: "technical stop"

basis:
  source: "basis donor clean-corrupt block-10 resid_mid chords"
  matrix_shape:
    - 512
    - 768
  centered: false
  dtype: "float64"
  device: "CPU"
  blas_threads: 1
  algorithm: "scipy.linalg.svd"
  full_matrices: false
  lapack_driver: "gesvd"
  columns: 4
  sign_rule: "largest absolute coordinate positive; lowest index breaks ties"
  spectral_gap_min_sigma4_over_sigma5: 1.10
  min_sigma4_over_sigma1: 0.0001
  leave_one_noun_max_principal_angle_degrees: 15
  hash_matrix: true
  hash_basis: true

radii:
  residual:
    scale: "median(||U^T(clean-corrupt chord)||_2 / sqrt(4))"
    multiplier: 0.20
    symbol: "h1"
    half_radius: true
    floor: "2^-10 * median RMS(resid_mid anchor)"
  gate:
    scale: "max(1.4826*MAD(pooled anchors), median abs clean-corrupt preactivation difference)"
    multiplier: 0.20
    symbol: "h2_j"
    half_radius: true
    floor: "2^-10 * max(1, median abs preactivation anchor)"
  radius_search_allowed: false
  radius_inflation_after_floor_failure_allowed: false
  leave_one_radius_noun_relative_change_max: 0.20

hooks:
  patch:
    name: "blocks.8.hook_mlp_out"
    position: "last"
  x:
    name: "blocks.10.hook_resid_mid"
    operation: "add Ux at last position"
  z:
    name: "blocks.10.mlp.hook_pre"
    operation: "add scalar z to current gate"
  gate:
    name: "blocks.10.mlp.hook_post"
    path: "current gate live; all other gates anchored"
    control: "current gate = exact act_fn(anchor_pre + z); all other gates anchored"
    joint: "selected ten live; unselected 3062 anchored"
  target_bypass_subtraction:
    name: "blocks.10.hook_resid_post"
    operation: "subtract Ux at last position"
  untouched_entry_tolerance: 0.0000001
  invocation_count_exactly_one: true
  clone_before_mutation: true
  reset_hooks_in_finally: true

finite_differences:
  radii:
    - 1.0
    - 0.5
  extrapolation: "(4 * half - full) / 3"
  x_axes: 4
  z_axes: 10
  path_mixed_corners_per_gate_per_radius: 16
  control_mixed_corners_per_gate_per_radius: 16
  calls_per_gate_per_radius_per_system: 42
  shared_centers_per_system: 1
  calls_per_system_per_tensor_item: 841
  calls_target_plus_patched_per_tensor_item: 1682
  ridge_allowed: false
  inverse_after_richardson: true
  compute_full_and_half_raw_inverses: true

target:
  implementation: "src/green_bridge_path_target.py"
  tensor_items_used: false
  energy_items_used: true
  selected_gates_joint: true
  unselected_gates_anchored: true
  residual_bypass_subtracted: true
  primary_radius: 1.0
  half_radius: 0.5
  definition: "(g(+rho) - g(-rho)) / (2*rho)"
  units: "Greater-Than logit margin"
  cell_target: "abs(mean PSE_patched - mean PSE_target)"
  conditioning_gap: "abs(mean PSE_target - mean PSE_corrupt)"
  jvp_role: "implementation and locality audit only"
  prohibited_imports:
    - "mixed_path_identification"
    - "matched_bypass_gate"
    - "green_bridge_tail"
    - "predictor score modules"
    - "baseline modules"
  target_reads_predictor_outputs: false

baselines:
  behavioral:
    score: "abs(center margin patched - center margin target)"
  single_direction:
    residual_direction: "delta"
    gate_direction: "standardized clean-corrupt selected-gate chord zeta"
    radii:
      - 1.0
      - 0.5
    conceptual_calls: 8
    endpoints_shared_with_pie: true
  first_order:
    random_generator: "numpy PCG64"
    seed_source: "first 8 big-endian bytes of SHA256('idle1-gt-bridge-20260805:first-order')"
    residual_directions: 200
    first_four_axes: true
    random_directions: 196
    gate_axes: 10
    calls_per_item: 1682
    equal_to_mixed_budget: true
  pie:
    corners_per_system_per_radius: 4
    conceptual_calls_per_item: 16
    provides_cancellation_main_effects: true
  calibration:
    methods:
      - behavioral
      - single_direction
      - first_order
      - pie
    solver: "scipy.optimize.nnls"
    intercept_nonnegative: true
    slope_nonnegative: true
    ridge: false
    development: "16-fold leave-one-cell-out"
    confirmation: "single all-development fit, frozen"
    mixed_calibration: "identity"

numerical_audits:
  hf_vs_tl:
    prompts: 32
    max_abs_logit_error: 0.00002
  no_op_patch:
    prompts: 32
    max_abs_logit_error: 0.00002
  manual_tail:
    conditions: 32
    condition_types:
      center: 8
      x_only: 8
      z_only: 8
      path_mixed: 4
      control_mixed: 4
    max_abs_logit_error: 0.00002
    derivative_relative_error: 0.0001
  center_replay:
    rms_error: 0.000002
    max_abs_error: 0.00002
  development_duplicates:
    full_anchors: 64
    tail_corners: 32
    epsilon_floor: 0.0000001
  confirmation_duplicates:
    full_anchors: 32
    tail_corners: 32
    max_error: "max(2*epsilon_dev, 2e-6)"

structural_admissibility:
  curvature_rms_min: 0.0005
  curvature_snr_min: 20
  gate_response_rms_min: 0.0005
  gate_response_snr_min: 20
  factorization_relative_residual_max: 0.15
  white_box_A_relative_error_max: 0.05
  white_box_A_small_absolute_error_max: 0.0001
  tensor_full_half_cosine_min: 0.95
  tensor_full_half_symmetric_relative_change_max: 0.25
  richardson_correction_relative_max: 0.25
  tensor_snr_min: 20
  bypass_cross_gate_relative_rms_max: 0.15
  certified_null_gate_response_noise_multiple: 5
  certified_null_logit_bound_max: 0.005
  active_gates_per_system_per_item_min: 3
  valid_tensor_items_per_cell_min: 6
  valid_energy_items_per_cell_min: 6
  projected_chord_min_sigma_x: 0.10

target_audits:
  jvp_absolute_error_max: 0.01
  jvp_relative_error_max: 0.05
  jvp_relative_denominator_floor: 0.05
  full_radius_locality_absolute_floor: 0.02
  full_radius_locality_relative_max: 0.25
  basis_leave_one_noun_spearman_min: 0.90
  basis_leave_one_noun_median_relative_change_max: 0.20

survival:
  development_cells_min: 15
  total_cells_technical_min: 40
  confirmation_cells_technical_min: 28
  confirmation_cells_oral_min: 29
  confirmation_cells_per_bin_min: 14
  conditioned_development_cells_min: 15
  conditioned_confirmation_cells_min: 29

conditioning:
  absolute_gap_logit: 0.10
  alternative_gap_dev_sd: 0.25

development:
  mixed_snr_min: 3
  cells_meeting_snr_min: 10
  raw_mixed_gain_stop_below: 0.05
  raw_mixed_gain_poster_below: 0.10
  confirmation_open_gain_min: 0.10
  confirmation_data_read_before_freeze_allowed: false

confirmation:
  bootstrap:
    replicates: 100000
    rng: "numpy PCG64"
    seed: 20260805
    paired: true
    stratify_by_distance_bin: true
    refit_baselines: false
    interval: "percentile 95%"
  overall:
    relative_rmse_gain_min: 0.20
    relative_rmse_gain_lower_bound_min: 0.10
    absolute_rmse_reduction_min: 0.01
  per_bin:
    relative_rmse_gain_min: 0.10
    relative_rmse_gain_lower_bound_min: 0.0
    lower_bound_strictly_greater: true
    absolute_rmse_reduction_min: 0.005
  cancellation:
    product_must_be_negative: true
    main_effect_abs_min: 0.05
    subset_size_min: 8
    positives_min: 3
    negatives_min: 3
    cells_per_bin_min: 3
    high_mismatch_threshold: 0.10
    auroc_min: 0.80
    auroc_lower_bound_min: 0.70
    bootstrap_strata:
      - distance_bin
      - target_class
  half_radius:
    spearman_overall_min: 0.90
    spearman_per_bin_min: 0.90
    median_symmetric_relative_change_overall_max: 0.20
    median_symmetric_relative_change_per_bin_max: 0.20
    denominator_floor: 0.05

compute:
  tensor_items_total: 384
  energy_items_total: 384
  donor_pairs_total: 1024
  tail_evaluations_core: 1303552
  tail_evaluations_audit: 96
  tail_evaluations_total: 1303648
  jvp_invocations_total: 1152
  full_model_evaluations_core: 4352
  full_model_evaluations_audit: 192
  full_model_evaluations_total: 4544
  raw_invocations_total: 1309344
  effective_tail_units_total: 1310496
  conservative_hard_budget_units: 1333216
  development_effective_units: 439008
  confirmation_effective_units: 871488
  preflight_fraction: 0.02
  preflight_reused: true
  rtx4090:
    memory_gb: 24
    tail_batch_target: 512
    full_batch_target: 64
    jvp_batch_target: 64
    peak_memory_ceiling_gb: 20
    expected_hours_min: 8
    expected_hours_max: 16
    hard_hours_max: 24
  a40:
    memory_gb: 48
    tail_batch_target: 1024
    full_batch_target: 128
    jvp_batch_target: 128
    peak_memory_ceiling_gb: 32
    expected_hours_min: 14
    expected_hours_max: 28
    hard_hours_max: 40

implementation_files:
  - "src/green_bridge_spec.py"
  - "src/green_bridge_dataset.py"
  - "src/matched_bypass_gate.py"
  - "src/green_bridge_tail.py"
  - "src/green_bridge_path_target.py"
  - "src/exp_green_bridge_gpt2.py"
  - "src/analyze_green_bridge.py"
  - "src/test_green_bridge_contract.py"

required_outputs:
  - "requirements-green-bridge.lock"
  - "outputs/green_bridge/manifest.json"
  - "outputs/green_bridge/model_fingerprint.json"
  - "outputs/green_bridge/splits.json"
  - "outputs/green_bridge/donor_basis.npz"
  - "outputs/green_bridge/radii.json"
  - "outputs/green_bridge/hook_audit.json"
  - "outputs/green_bridge/tail_audit.json"
  - "outputs/green_bridge/noise_audit_dev.json"
  - "outputs/green_bridge/dev_tensor_scores.parquet"
  - "outputs/green_bridge/dev_energy_targets.parquet"
  - "outputs/green_bridge/dev_result.json"
  - "outputs/green_bridge/frozen_analysis.json"
  - "outputs/green_bridge/noise_audit_confirm.json"
  - "outputs/green_bridge/confirm_tensor_scores.parquet"
  - "outputs/green_bridge/confirm_energy_targets.parquet"
  - "outputs/green_bridge/result.json"
  - "outputs/green_bridge/sha256sums.txt"

terminal_rules:
  first_failed_gate_recorded: true
  post_failure_model_search: false
  post_failure_hook_search: false
  post_failure_gate_search: false
  post_failure_corruption_search: false
  post_failure_radius_search: false
  post_failure_threshold_change: false
  post_failure_confirmation_retry: false
```

BRIDGE GREEN — SERVER EXECUTION AUTHORIZED