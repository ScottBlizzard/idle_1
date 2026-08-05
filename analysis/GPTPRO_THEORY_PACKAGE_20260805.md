# Theory Gate Verdict

**Audited object.** This package resolves the formal gate for repository `idle_1`, branch `main`, commit `3553809d4e0f7241169b1139cc2376871a2b0bd9`. The Round 2 entry point explicitly requires the first-round verdict, the binding theory prompt, the P0 theory note, and the relevant implementation/tests to be read in that order, and prohibits further GPU experiments until a non-tautological identification theorem with a converse exists.

**Binding status: AMBER.**

A non-tautological positive theorem can be proved for a restricted but genuinely transformer-relevant class: an acyclic residual graph in which an upstream mediator block enters known, independently intervenable elementwise gates; the gate outputs enter the measured output linearly; and every remaining bypass is independent of the gate intervention. In that class:

\[
\text{central first-order responses}
+
\text{central mixed-second-order responses}
\]

identify a fixed-basis **reduced local path-gain tensor**, together with the direct upstream bypass gain and downstream gate gain. The structural inverse, its constants, probe-completeness conditions, finite-radius error, finite-design error, and finite-probe concentration bounds are derived below rather than assumed.

The result is not a theorem about arbitrary transformer block outputs. In particular, it does not validate the currently proposed Greater-Than grouping

\[
M_1=
\{\text{heads }5.1,5.5,6.9,7.10,8.8,8.11,9.1\},
\qquad
M_2=
\{\text{MLPs }8,9,10,11\}.
\]

That grouping fails three load-bearing conditions:

1. it is not a two-block acyclic cut because head \(9.1\) lies downstream of MLP \(8\) but upstream of MLP \(9\);
2. the proposed \(M_2\) variables are MLP residual writes, not known elementwise gate-input coordinates with an observable activation-curvature ratio;
3. the downstream computation contains serial MLP-to-MLP routes, direct residual routes, attention routes, layer-normalization effects, and omitted mixed bypasses whose contributions to the total mixed derivative are not separated by the proposed measurements.

The Greater-Than analysis itself reports that MLPs rely on upstream MLPs, that MLP \(8\) relies on the listed earlier heads, that MLP \(9\) relies on head \(9.1\), and that several listed heads also contribute directly to the logits.  It further characterizes MLP \(8\) as mainly indirect, MLP \(9\) as having substantial direct and indirect contributions, MLP \(10\) as mainly direct, and MLP \(11\) as direct.  Those facts are incompatible with treating the four MLP outputs as one parallel separable gate block.

The current P0 theory correctly stops at local response agreement and explicitly disclaims circuit identification.  The implementation likewise contains forward and symmetric first-order signatures but no four-corner mixed estimator, no curvature correction, no structural inverse, and no probe-complete tensor recovery.  Its tests verify first-order linear recovery and quadratic cancellation, not structural identification.

Accordingly:

- the restricted-class theorem below is valid;
- a poster-level fixed-basis local-path identification claim survives;
- arbitrary block-output path or circuit identification does not survive;
- **the execution agent is not authorized to run the proposed Greater-Than GPU program in its present \(M_1/M_2\) form**;
- CPU-only analytic implementation tests specified below are authorized because they verify the proved estimator rather than substitute experiments for missing theory.

The unresolved lemma preventing GREEN is:

> **Greater-Than structural-inverse lemma.** For the exact GPT-2-small computational DAG and the exact proposed intervention hooks, derive an architecture-level injective map from the total first and mixed responses at the proposed \(M_1\) and raw-MLP-output \(M_2\) cuts to the declared \(M_1\!\to M_2\!\to\) margin path effect, with every direct, serial, interleaved, normalization, attention, and residual bypass term either observed, algebraically removed, or proved zero.

No such map is present, and Section 6 proves that without a complete-cut restriction of precisely this kind the desired path effect is not identifiable even from perfect, noiseless, probe-complete local response tensors.

# Formal Objects and Equivalence Relations

## Common fixed-basis setting

Let \(s\in\{p,\star\}\) index a patched system and its target system. All comparisons occur in a common, predeclared coordinate basis.

Let

\[
x\in\mathbb R^{d_1},
\qquad
z\in\mathbb R^{d_2}
\]

denote additive interventions on two declared mediator blocks, and let

\[
f_s:\mathcal N\subseteq\mathbb R^{d_1}\times\mathbb R^{d_2}
\longrightarrow
\mathbb R^k
\]

be the downstream output map around the system-specific center, translated so that the center is \((x,z)=(0,0)\).

All vector norms are Euclidean. Matrix and tensor norms without a subscript are Frobenius norms. For a multilinear map \(T\), \(\|T\|_{\mathrm{op}}\) denotes its induced Euclidean operator norm.

For

\[
P\in\mathbb R^{k\times d_1\times d_2},
\]

define

\[
P[x,z]_\alpha
=
\sum_{i=1}^{d_1}\sum_{j=1}^{d_2}
P_{\alpha i j}x_i z_j
\]

and define the contraction over the \(M_2\) index

\[
\mathsf S(P)_{\alpha i}
=
\sum_{j=1}^{d_2}P_{\alpha i j}.
\]

For a vector \(\rho\in\mathbb R^{d_2}\), define mode-\(3\) scaling

\[
(\mathsf R_\rho H)_{\alpha i j}
=
\rho_j H_{\alpha i j}.
\]

Let \(Q_1\) and \(Q_2\) be probe laws on the declared \(M_1\) and \(M_2\) coordinates. Their uncentered second-moment operators are

\[
M_1=\mathbb E_{u\sim Q_1}[uu^\top],
\qquad
M_2=\mathbb E_{v\sim Q_2}[vv^\top].
\]

When a probe law is centrally symmetric, these are also its covariance matrices. The theorems require the second moments, not zero probe means.

## Zero-order behavioral equivalence

For a fixed linear task readout \(\ell\in\mathbb R^k\), define

\[
f_p\equiv_{0,\ell} f_\star
\quad\Longleftrightarrow\quad
\ell^\top f_p(0,0)
=
\ell^\top f_\star(0,0).
\]

Vector-valued zero-order equivalence is

\[
f_p\equiv_0 f_\star
\quad\Longleftrightarrow\quad
f_p(0,0)=f_\star(0,0).
\]

Equality of one task margin is strictly weaker than equality of the full output vector.

## Probe-indexed first-order local equivalence

Define

\[
J_{1,s}=D_1f_s(0,0)\in\mathbb R^{k\times d_1},
\qquad
J_{2,s}=D_2f_s(0,0)\in\mathbb R^{k\times d_2}.
\]

Then

\[
f_p\equiv_{1,Q_1,Q_2} f_\star
\]

means

\[
J_{1,p}u=J_{1,\star}u
\quad Q_1\text{-almost surely},
\]

and

\[
J_{2,p}v=J_{2,\star}v
\quad Q_2\text{-almost surely}.
\]

This relation is indexed by the probe laws. If \(M_1\) or \(M_2\) is singular, it says nothing about directions outside the corresponding covered subspace.

## Probe-indexed first-plus-mixed-second-order local equivalence

Define the total mixed derivative tensor

\[
H_s=D_{12}^2f_s(0,0)
\in\mathbb R^{k\times d_1\times d_2}.
\]

Raw first-plus-mixed equivalence is

\[
f_p\equiv_{2,Q_1,Q_2}f_\star
\]

when first-order equivalence holds and

\[
H_p[u,v]=H_\star[u,v]
\quad
(Q_1\otimes Q_2)\text{-almost surely}.
\]

Raw equality of \(H_p\) and \(H_\star\) is not generally structural equality. The positive class below supplies known system-specific curvature corrections

\[
P_s=\mathsf R_{\rho_s}H_s.
\]

The structurally calibrated local relation is therefore

\[
f_p\equiv_{2,\rho,Q_1,Q_2}f_\star
\]

when

\[
J_{1,p}u=J_{1,\star}u,
\qquad
J_{2,p}v=J_{2,\star}v,
\qquad
P_p[u,v]=P_\star[u,v]
\]

almost surely under the corresponding laws.

## Reduced structural and path equivalence

The independently defined structural object will be

\[
\theta_s=(D_s,G_s,P_s),
\]

where

\[
D_s\in\mathbb R^{k\times d_1}
\]

is the direct \(M_1\)-to-output bypass gain,

\[
G_s\in\mathbb R^{k\times d_2}
\]

is the local \(M_2\)-gate-to-output gain, and

\[
P_s\in\mathbb R^{k\times d_1\times d_2}
\]

is the mediator-resolved product of edge derivatives along the declared paths

\[
M_{1,i}\longrightarrow M_{2,j}
\longrightarrow Y_\alpha.
\]

Define

\[
\theta_p\equiv_{\mathrm{str}}\theta_\star
\quad\Longleftrightarrow\quad
D_p=D_\star,\quad
G_p=G_\star,\quad
P_p=P_\star.
\]

Path equivalence alone is

\[
\theta_p\equiv_{\mathrm{path}}\theta_\star
\quad\Longleftrightarrow\quad
P_p=P_\star.
\]

The identified detailed-model object is the quotient

\[
[(A,C,b)]_\theta
=
\left\{
(A',C',b'):
\theta(A',C',b')=\theta(A,C,b)
\right\}.
\]

Thus the theorem identifies a reduced local structural equivalence class. It does not silently promote that quotient to complete parameter identity.

## Relations explicitly not identified

The following are different relations.

### Full weight identity

This requires equality of every underlying weight, bias, normalization parameter, and internal coordinate implementation. The theorem does not imply it.

### Hidden-basis or reparameterization identity

Two detailed systems can implement the same function under

\[
A'=RA,
\qquad
C'=CR^{-1}
\]

for an invertible hidden transformation \(R\). Response tensors cannot distinguish such parameterizations unless the hidden basis is fixed by intervention semantics.

### Circuit identity

Circuit identity would require a unique graph, unique edge support, and a unique allocation of causal contribution to graph edges. The reduced tensor \(P\) records local effective gains in a declared graph; it does not prove that no alternative graph computes the same local map.

### Global algorithm identity

Global identity requires

\[
f_p(x,z)=f_\star(x,z)
\]

on a declared global domain, or an equivalence of algorithms over an input distribution. Equality of finite derivatives at one center does not imply this.

## Logical hierarchy

Without the structural assumptions in Section 3, none of the following implications is valid:

\[
\equiv_0
\centernot\Longrightarrow
\equiv_1,
\]

\[
\equiv_1
\centernot\Longrightarrow
\equiv_2,
\]

\[
\equiv_2
\centernot\Longrightarrow
\equiv_{\mathrm{path}},
\]

\[
\equiv_{\mathrm{path}}
\centernot\Longrightarrow
\text{full weight, circuit, or global identity}.
\]

Within the positive class, complete calibrated first-plus-mixed responses identify \(\theta\). The class assumptions, not the derivative notation itself, establish that implication.

# Explicit Multi-Mediator Structural Class

## Anchored Separable-Gate Residual DAG

The positive class is called the **Anchored Separable-Gate Residual DAG**, abbreviated ASG-RDAG.

For each system \(s\), let

\[
x\in\mathbb R^{d_1}
\]

be the upstream mediator intervention, and let

\[
z\in\mathbb R^{d_2}
\]

be an additive intervention at a fixed set of downstream gate-input coordinates.

The graph contains the variables

\[
X_i,\quad
Z_j,\quad
W_j,\quad
Y_\alpha
\]

with edges

\[
X_i\to Z_j,
\qquad
Z_j\to W_j,
\qquad
W_j\to Y_\alpha,
\]

together with an unrestricted \(X\)-only residual bypass

\[
X\to Y.
\]

The structural equations are

\[
Z_{s,j}(x,z)
=
a_{s,j}
+
A_{s,j:}x
+
z_j,
\]

\[
W_{s,j}(x,z)
=
\psi_j\!\left(Z_{s,j}(x,z)\right),
\]

and

\[
f_s(x,z)
=
y_s
+
b_s(x)
+
\sum_{j=1}^{d_2}
c_{s,j}
\left[
\psi_j\!\left(a_{s,j}+A_{s,j:}x+z_j\right)
-
\psi_j(a_{s,j})
\right].
\tag{3.1}
\]

Here:

- \(y_s\in\mathbb R^k\) is the center output;
- \(b_s:\mathbb R^{d_1}\to\mathbb R^k\) is the direct residual bypass, with \(b_s(0)=0\);
- \(A_s\in\mathbb R^{d_2\times d_1}\) is the upstream-to-gate edge matrix;
- \(a_s\in\mathbb R^{d_2}\) is the observed gate-input anchor;
- \(c_{s,j}\in\mathbb R^k\) is the output direction of gate \(j\);
- \(\psi_j:\mathbb R\to\mathbb R\) is a known elementwise activation.

The intervention \(do(z_j=\zeta)\) is an additive intervention at the preactivation coordinate \(Z_j\), after all upstream contributions have been computed. It does not rotate, align, or relearn the gate basis.

## Load-bearing assumptions

### ASG-1: fixed coordinates

The \(x\), \(z\), and output coordinates are fixed before comparing systems. No system-specific Procrustes map, canonical-correlation map, or learned invertible alignment is allowed.

A fixed orthonormal basis fitted on an independent donor split may define the \(x\)-subspace. The \(z\)-coordinates must remain actual gate coordinates, or a coordinate selection and signed permutation that preserves the elementwise activation structure.

### ASG-2: acyclic block order

Every declared \(X_i\) is upstream of every declared gate \(Z_j\). There is no edge from a declared \(Z_j\), \(W_j\), or downstream variable back into the declared \(X\) block.

### ASG-3: separable known gates

Conditional on \(x\), each \(z_j\) enters exactly one known scalar activation:

\[
z_j\mapsto\psi_j(a_{s,j}+A_{s,j:}x+z_j).
\]

There are no terms such as

\[
\psi_{j\ell}(Z_j,Z_\ell)
\]

and no unknown mixing of gate coordinates before the activation.

### ASG-4: linear gate-output readout

After activation, each gate contributes through the fixed output vector \(c_{s,j}\). There is no downstream nonlinearity that jointly mixes \(x\) and \(z\) before the measured output.

A known fixed linear map can be absorbed into the columns \(c_{s,j}\).

### ASG-5: complete mixed cut

The bypass \(b_s\) is independent of \(z\). Equivalently,

\[
D_{12}^2 b_s(x,z)=0.
\]

All \(x\)-\(z\) mixed dependence in the measured output is generated by the declared gates. This is the complete-cut condition. Section 6 proves that its removal destroys path identification.

### ASG-6: known anchors and nonzero curvature

The gate anchors \(a_{s,j}\) are observed from the forward pass, and the activation functions are known from the architecture.

For every declared gate,

\[
\psi_j''(a_{s,j})\neq 0.
\tag{3.2}
\]

Define the known curvature ratio

\[
\rho_{s,j}
=
\frac{\psi_j'(a_{s,j})}
     {\psi_j''(a_{s,j})}.
\tag{3.3}
\]

For stable finite-radius inference, report

\[
\gamma_s
=
\min_j
|\psi_j''(a_{s,j})|
>0
\]

and

\[
\rho_{\max,s}
=
\max_j|\rho_{s,j}|.
\]

No unspecified positive \(\kappa\) is assumed. The only scalar inverse factor is the explicit, architecture-derived ratio in (3.3).

### ASG-7: smoothness and locality

For exact derivative identification, \(b_s\) is \(C^1\) and the gates are \(C^2\) at the anchor.

For the central finite-radius bounds, \(b_s\) and the activations are \(C^4\) on the entire intervention rectangle used by the estimator.

### ASG-8: no hidden genericity assumptions

The theorem assumes no:

- sparsity of \(A_s\);
- sign pattern;
- monotonicity;
- full rank of \(A_s\);
- full rank of the output columns \(c_{s,j}\);
- noncancellation or generic faithfulness;
- uniqueness of the complete network weights.

Probe rank is required. Structural edge rank is not.

## Independently meaningful structural parameters

Define

\[
D_s
=
Db_s(0)
\in\mathbb R^{k\times d_1}.
\tag{3.4}
\]

Define the gate-to-output edge gain

\[
G_{s,\alpha j}
=
c_{s,j,\alpha}\psi_j'(a_{s,j}).
\tag{3.5}
\]

Define the reduced path-gain tensor

\[
P_{s,\alpha i j}
=
c_{s,j,\alpha}
\psi_j'(a_{s,j})
A_{s,ji}.
\tag{3.6}
\]

Equation (3.6) is not a definition in terms of observed response derivatives. It is the product of structural edge derivatives along the graph path

\[
X_i
\longrightarrow
Z_j
\longrightarrow
W_j
\longrightarrow
Y_\alpha:
\]

\[
\frac{\partial Z_j}{\partial X_i}
\,
\frac{\partial W_j}{\partial Z_j}
\,
\frac{\partial Y_\alpha}{\partial W_j}
=
A_{s,ji}
\psi_j'(a_{s,j})
c_{s,j,\alpha}.
\]

Thus \(P\) has an independent path interpretation.

## Observable derivative map

Differentiating (3.1) at \((0,0)\) gives

\[
J_{1,s}
=
D_s+\mathsf S(P_s),
\tag{3.7}
\]

\[
J_{2,s}
=
G_s,
\tag{3.8}
\]

and

\[
H_{s,\alpha i j}
=
c_{s,j,\alpha}
\psi_j''(a_{s,j})
A_{s,ji}.
\tag{3.9}
\]

Combining (3.3), (3.6), and (3.9),

\[
P_s
=
\mathsf R_{\rho_s}H_s.
\tag{3.10}
\]

Therefore the explicit structural inverse is

\[
\boxed{
\begin{aligned}
P_s&=\mathsf R_{\rho_s}H_s,\\
G_s&=J_{2,s},\\
D_s&=J_{1,s}-\mathsf S(P_s).
\end{aligned}}
\tag{3.11}
\]

This is the load-bearing inverse. It is derived from the substantive gate graph rather than postulated.

## Optional active-channel edge recovery

The reduced tensor \(P\) is identifiable even when the detailed edges are not.

For a gate \(j\) satisfying

\[
\psi_j'(a_{s,j})\neq 0,
\]

the output column is

\[
c_{s,j}
=
\frac{G_{s,:,j}}
     {\psi_j'(a_{s,j})}.
\tag{3.12}
\]

Because

\[
P_{s,:, :,j}
=
G_{s,:,j} A_{s,j:},
\]

if additionally

\[
\|G_{s,:,j}\|_2>0,
\]

then

\[
A_{s,j:}
=
\frac{
G_{s,:,j}^{\top}P_{s,:,:,j}
}{
\|G_{s,:,j}\|_2^2
}.
\tag{3.13}
\]

Thus the detailed local edges are recovered on active output-visible channels.

If \(G_{s,:,j}=0\), then \(P_{s,:,:,j}=0\), and \(A_{s,j:}\) is output-null and unidentifiable. This does not compromise reduced path-gain identification: every path through that gate has zero first-order output gain.

## Concrete activation example

For the exact Gaussian GELU

\[
\psi(t)=t\Phi(t),
\]

where \(\Phi\) and \(\phi\) are the standard normal CDF and density,

\[
\psi'(t)=\Phi(t)+t\phi(t),
\]

\[
\psi''(t)=(2-t^2)\phi(t).
\]

Hence

\[
\rho(t)
=
\frac{\Phi(t)+t\phi(t)}
     {(2-t^2)\phi(t)},
\]

and the curvature condition fails exactly at

\[
t=\pm\sqrt 2.
\]

This illustrates that the condition is an explicit check on known activations and observed anchors, not an unnamed faithfulness constant. For any implemented GELU approximation, the exact code-level derivatives must be used instead.

## Class-specific finite-radius derivative constants

Let \(A_{s,j:}\) denote row \(j\) of \(A_s\). On the probed neighborhood define

\[
B_{3,s}
=
\sup_x
\|D^3b_s(x)\|_{\mathrm{op}}.
\]

If

\[
L_{\ell,s,j}
=
\sup_{\text{probed interval}}
|\psi_j^{(\ell)}|,
\]

then the following valid vector-output bounds are available:

\[
M_{300,s}
\le
B_{3,s}
+
\sum_{j=1}^{d_2}
\|c_{s,j}\|_2
L_{3,s,j}
\|A_{s,j:}\|_2^3,
\tag{3.14}
\]

\[
M_{030,s}
\le
\sum_{j=1}^{d_2}
\|c_{s,j}\|_2
L_{3,s,j},
\tag{3.15}
\]

\[
M_{310,s}
\le
\sum_{j=1}^{d_2}
\|c_{s,j}\|_2
L_{4,s,j}
\|A_{s,j:}\|_2^3,
\tag{3.16}
\]

\[
M_{130,s}
\le
\sum_{j=1}^{d_2}
\|c_{s,j}\|_2
L_{4,s,j}
\|A_{s,j:}\|_2.
\tag{3.17}
\]

The subscripts count derivatives with respect to block \(1\) and block \(2\). These are coarse but explicit constants. Because the norm is an \(\mathbb R^k\)-valued operator norm, no output-dimension factor has been omitted. If only a coordinatewise derivative bound \(\bar M\) is available, the corresponding vector bound is at most \(\sqrt{k}\bar M\).

# Main Identification Theorem

## Theorem 1: exact fixed-basis path-gain identification

Let \(d_1,d_2,k\ge 1\). Let \(f_p,f_\star\) be two ASG-RDAG systems satisfying ASG-1 through ASG-6.

For \(s\in\{p,\star\}\), define

\[
\theta_s=(D_s,G_s,P_s)
\]

and equip the structural space with

\[
\|\theta_s\|_\Theta^2
=
\|D_s\|_F^2
+
\|G_s\|_F^2
+
\|P_s\|_F^2.
\]

Define the calibrated local response tuple

\[
L_s
=
(J_{1,s},J_{2,s},P_s)
\]

with norm

\[
\|L_s\|_{\mathcal L}^2
=
\|J_{1,s}\|_F^2
+
\|J_{2,s}\|_F^2
+
\|P_s\|_F^2.
\]

Let

\[
\Delta\theta=\theta_p-\theta_\star,
\qquad
\Delta L=L_p-L_\star.
\]

Then:

### Exact structural inverse

\[
P_s=\mathsf R_{\rho_s}H_s,
\qquad
G_s=J_{2,s},
\qquad
D_s=J_{1,s}-\mathsf S(P_s).
\tag{4.1}
\]

### Derived structural-map conditioning

\[
\frac{1}{2d_2+1}
\|\Delta\theta\|_\Theta^2
\le
\|\Delta L\|_{\mathcal L}^2
\le
(2d_2+1)
\|\Delta\theta\|_\Theta^2.
\tag{4.2}
\]

Thus the squared lower inverse constant is

\[
\kappa_{\mathrm{ASG}}^2
=
\frac{1}{2d_2+1},
\]

derived from the graph rather than assumed.

### Product-probe response energy

Let \(Q_1\otimes Q_2\) be a product law with second moments satisfying

\[
\lambda_1^-I_{d_1}
\preceq
M_1
\preceq
\lambda_1^+I_{d_1},
\]

\[
\lambda_2^-I_{d_2}
\preceq
M_2
\preceq
\lambda_2^+I_{d_2},
\]

where

\[
\lambda_1^-,\lambda_2^->0.
\]

Define

\[
\mathcal E_Q(\Delta L)
=
\mathbb E_{u\sim Q_1}
\|\Delta J_1u\|_2^2
+
\mathbb E_{v\sim Q_2}
\|\Delta J_2v\|_2^2
+
\mathbb E_{u\sim Q_1,v\sim Q_2}
\|\Delta P[u,v]\|_2^2.
\tag{4.3}
\]

Set

\[
c_Q
=
\min\left\{
\lambda_1^-,
\lambda_2^-,
\lambda_1^-\lambda_2^-
\right\},
\tag{4.4}
\]

\[
C_Q
=
\max\left\{
\lambda_1^+,
\lambda_2^+,
\lambda_1^+\lambda_2^+
\right\}.
\tag{4.5}
\]

Then

\[
c_Q\|\Delta L\|_{\mathcal L}^2
\le
\mathcal E_Q(\Delta L)
\le
C_Q\|\Delta L\|_{\mathcal L}^2.
\tag{4.6}
\]

Combining (4.2) and (4.6),

\[
\boxed{
\frac{c_Q}{2d_2+1}
\|\Delta\theta\|_\Theta^2
\le
\mathcal E_Q(\Delta L)
\le
C_Q(2d_2+1)
\|\Delta\theta\|_\Theta^2.
}
\tag{4.7}
\]

Consequently,

\[
\boxed{
\|\Delta\theta\|_\Theta^2
\le
\frac{2d_2+1}{c_Q}
\mathcal E_Q(\Delta L).
}
\tag{4.8}
\]

Moreover,

\[
\mathcal E_Q(\Delta L)=0
\quad\Longleftrightarrow\quad
\Delta\theta=0.
\tag{4.9}
\]

Equation (4.9) is valid only on the declared probe subspaces with strictly positive second-moment eigenvalues.

## Theorem 2: central finite-radius and finite-design recovery

Assume ASG-7. Let

\[
u_a\in\mathbb R^{d_1},
\quad
\|u_a\|_2=1,
\quad
a=1,\ldots,m_1,
\]

and

\[
v_b\in\mathbb R^{d_2},
\quad
\|v_b\|_2=1,
\quad
b=1,\ldots,m_2.
\]

Let \(r_a>0\) and \(t_b>0\) be probe-specific radii.

Define the central first responses

\[
Y^{(1)}_{s,a}
=
\frac{
f_s(r_au_a,0)-f_s(-r_au_a,0)
}{
2r_a
},
\tag{4.10}
\]

\[
Y^{(2)}_{s,b}
=
\frac{
f_s(0,t_bv_b)-f_s(0,-t_bv_b)
}{
2t_b
},
\tag{4.11}
\]

and the four-corner mixed responses

\[
\begin{aligned}
Y^{(12)}_{s,ab}
=
\frac{1}{4r_at_b}
\big[
&
f_s(r_au_a,t_bv_b)
-
f_s(r_au_a,-t_bv_b)\\
&
-
f_s(-r_au_a,t_bv_b)
+
f_s(-r_au_a,-t_bv_b)
\big].
\end{aligned}
\tag{4.12}
\]

Let \(U\in\mathbb R^{m_1\times d_1}\) and \(V\in\mathbb R^{m_2\times d_2}\) have rows \(u_a^\top\) and \(v_b^\top\). Assume

\[
\mu_1
=
\lambda_{\min}\!\left(\frac{U^\top U}{m_1}\right)
>0,
\qquad
\mu_2
=
\lambda_{\min}\!\left(\frac{V^\top V}{m_2}\right)
>0.
\tag{4.13}
\]

Arrange \(Y_s^{(1)}\in\mathbb R^{m_1\times k}\) and \(Y_s^{(2)}\in\mathbb R^{m_2\times k}\) by rows. For output coordinate \(\alpha\), let

\[
Y^{(12)}_{s,\alpha}
\in
\mathbb R^{m_1\times m_2}
\]

have entry \(Y^{(12)}_{s,ab,\alpha}\).

Define the ordinary least-squares estimators

\[
\widehat J_{1,s}
=
(Y_s^{(1)})^\top
U
(U^\top U)^{-1},
\tag{4.14}
\]

\[
\widehat J_{2,s}
=
(Y_s^{(2)})^\top
V
(V^\top V)^{-1},
\tag{4.15}
\]

and

\[
\widehat H_{s,\alpha}
=
(U^\top U)^{-1}
U^\top
Y^{(12)}_{s,\alpha}
V
(V^\top V)^{-1}.
\tag{4.16}
\]

Set

\[
\widehat P_s
=
\mathsf R_{\rho_s}\widehat H_s,
\tag{4.17}
\]

\[
\widehat G_s
=
\widehat J_{2,s},
\tag{4.18}
\]

\[
\widehat D_s
=
\widehat J_{1,s}
-
\mathsf S(\widehat P_s).
\tag{4.19}
\]

Let

\[
\overline e_{1,s}^2
=
\frac1{m_1}
\sum_{a=1}^{m_1}
\left(
\frac{M_{300,s}r_a^2}{6}
\right)^2,
\tag{4.20}
\]

\[
\overline e_{2,s}^2
=
\frac1{m_2}
\sum_{b=1}^{m_2}
\left(
\frac{M_{030,s}t_b^2}{6}
\right)^2,
\tag{4.21}
\]

and

\[
\overline e_{12,s}^2
=
\frac1{m_1m_2}
\sum_{a=1}^{m_1}
\sum_{b=1}^{m_2}
\left(
\frac{
M_{310,s}r_a^2
+
M_{130,s}t_b^2
}{6}
\right)^2.
\tag{4.22}
\]

Then

\[
\|\widehat J_{1,s}-J_{1,s}\|_F
\le
\epsilon_{1,s}
:=
\sqrt{\frac{\overline e_{1,s}^2}{\mu_1}},
\tag{4.23}
\]

\[
\|\widehat J_{2,s}-J_{2,s}\|_F
\le
\epsilon_{2,s}
:=
\sqrt{\frac{\overline e_{2,s}^2}{\mu_2}},
\tag{4.24}
\]

\[
\|\widehat H_s-H_s\|_F
\le
\epsilon_{H,s}
:=
\sqrt{
\frac{
\overline e_{12,s}^2
}{
\mu_1\mu_2
}
},
\tag{4.25}
\]

\[
\|\widehat P_s-P_s\|_F
\le
\epsilon_{P,s}
:=
\rho_{\max,s}\epsilon_{H,s},
\tag{4.26}
\]

and

\[
\|\widehat D_s-D_s\|_F
\le
\epsilon_{1,s}
+
\sqrt{d_2}\epsilon_{P,s}.
\tag{4.27}
\]

Therefore

\[
\boxed{
\|\widehat\theta_s-\theta_s\|_\Theta
\le
\epsilon_{\theta,s}
}
\tag{4.28}
\]

with

\[
\epsilon_{\theta,s}
=
\left[
\left(
\epsilon_{1,s}
+
\sqrt{d_2}\epsilon_{P,s}
\right)^2
+
\epsilon_{2,s}^2
+
\epsilon_{P,s}^2
\right]^{1/2}.
\tag{4.29}
\]

If \(\rho_s\) is numerically approximated by \(\widehat\rho_s\), then (4.26) becomes

\[
\|\widehat P_s-P_s\|_F
\le
\|\widehat\rho_s\|_\infty
\|\widehat H_s-H_s\|_F
+
\|\widehat\rho_s-\rho_s\|_\infty
\|H_s\|_F.
\tag{4.30}
\]

In the noiseless exact-derivative limit, full column rank of \(U\) and \(V\) yields exact recovery from finitely many probes. For a quadratic gate model, (4.10)–(4.12) are already exact at nonzero radius.

## Finite-radius structural-energy interval

Let

\[
\eta_1
=
\epsilon_{1,p}+\epsilon_{1,\star},
\qquad
\eta_2
=
\epsilon_{2,p}+\epsilon_{2,\star},
\qquad
\eta_P
=
\epsilon_{P,p}+\epsilon_{P,\star}.
\]

Let \(\widehat{\mathcal E}_Q\) denote (4.3) with the estimated tensors. Define

\[
B_{\mathrm{rad}}^2
=
\lambda_1^+\eta_1^2
+
\lambda_2^+\eta_2^2
+
\lambda_1^+\lambda_2^+\eta_P^2.
\tag{4.31}
\]

Then

\[
\left|
\sqrt{\widehat{\mathcal E}_Q}
-
\sqrt{\mathcal E_Q}
\right|
\le
B_{\mathrm{rad}},
\tag{4.32}
\]

and hence

\[
\boxed{
\left(
\sqrt{\widehat{\mathcal E}_Q}
-
B_{\mathrm{rad}}
\right)_+^2
\le
\mathcal E_Q
\le
\left(
\sqrt{\widehat{\mathcal E}_Q}
+
B_{\mathrm{rad}}
\right)^2.
}
\tag{4.33}
\]

This is preferable to an unspecified additive \(O(r^2)\): it gives the exact norm-level propagation of the finite-radius tensor errors.

## Theorem 3: finite-probe coverage and energy concentration

### Bounded i.i.d. probe coverage

Let \(x_1,\ldots,x_m\in\mathbb R^d\) be independent draws with

\[
\|x_\ell\|_2\le L
\quad\text{almost surely}.
\]

Let

\[
M=\mathbb E[xx^\top],
\qquad
\widehat M=\frac1m\sum_{\ell=1}^m x_\ell x_\ell^\top.
\]

Then with probability at least \(1-\delta\),

\[
\boxed{
\|\widehat M-M\|_{\mathrm{op}}
\le
2L^2
\sqrt{
\frac{
\log(2\cdot 9^d/\delta)
}{
2m
}
}.
}
\tag{4.34}
\]

Consequently, if the right side is at most \(\lambda_{\min}(M)/2\),

\[
\lambda_{\min}(\widehat M)
\ge
\frac12\lambda_{\min}(M).
\tag{4.35}
\]

A descriptive effective rank

\[
r_{\mathrm{eff}}(M)
=
\frac{\operatorname{tr}M}{\|M\|_{\mathrm{op}}}
\]

does not replace (4.35). Identification requires a positive minimum eigenvalue on the declared subspace.

### Bounded i.i.d. energy sampling

Freeze the estimated tensors using an independent fitting set. Draw independent evaluation pairs

\[
(u_\ell,v_\ell)
\overset{\mathrm{iid}}{\sim}
Q_1\otimes Q_2.
\]

Define

\[
X_\ell
=
\|\Delta\widehat J_1u_\ell\|_2^2
+
\|\Delta\widehat J_2v_\ell\|_2^2
+
\|\Delta\widehat P[u_\ell,v_\ell]\|_2^2.
\tag{4.36}
\]

If

\[
0\le X_\ell\le B_X,
\]

then

\[
\widehat{\mathcal E}_m
=
\frac1m\sum_{\ell=1}^m X_\ell
\]

satisfies, with probability at least \(1-\delta\),

\[
\left|
\widehat{\mathcal E}_m
-
\widehat{\mathcal E}_Q
\right|
\le
\xi_m
:=
B_X
\sqrt{
\frac{\log(2/\delta)}{2m}
}.
\tag{4.37}
\]

A computable valid bound is

\[
B_X
=
L_1^2\|\Delta\widehat J_1\|_F^2
+
L_2^2\|\Delta\widehat J_2\|_F^2
+
L_1^2L_2^2\|\Delta\widehat P\|_F^2
\tag{4.38}
\]

when \(\|u\|\le L_1\) and \(\|v\|\le L_2\).

Combining (4.33) and (4.37),

\[
\boxed{
\left(
\sqrt{(\widehat{\mathcal E}_m-\xi_m)_+}
-
B_{\mathrm{rad}}
\right)_+^2
\le
\mathcal E_Q
\le
\left(
\sqrt{\widehat{\mathcal E}_m+\xi_m}
+
B_{\mathrm{rad}}
\right)^2.
}
\tag{4.39}
\]

Therefore the structural discrepancy obeys

\[
\boxed{
\|\Delta\theta\|_\Theta^2
\le
\frac{2d_2+1}{c_Q}
\left(
\sqrt{\widehat{\mathcal E}_m+\xi_m}
+
B_{\mathrm{rad}}
\right)^2.
}
\tag{4.40}
\]

It also obeys the lower bound

\[
\boxed{
\|\Delta\theta\|_\Theta^2
\ge
\frac{
\left(
\sqrt{(\widehat{\mathcal E}_m-\xi_m)_+}
-
B_{\mathrm{rad}}
\right)_+^2
}{
C_Q(2d_2+1)
}.
}
\tag{4.41}
\]

### Sub-Gaussian probe energy

Suppose \(u\) and \(v\) are independent and satisfy, for all \(p\ge2\),

\[
\left(
\mathbb E|\langle a,u\rangle|^p
\right)^{1/p}
\le
K_1\sqrt p
\left(
a^\top M_1a
\right)^{1/2},
\tag{4.42}
\]

\[
\left(
\mathbb E|\langle b,v\rangle|^p
\right)^{1/p}
\le
K_2\sqrt p
\left(
b^\top M_2b
\right)^{1/2}.
\tag{4.43}
\]

Let

\[
\mu_1
=
\mathbb E\|\Delta\widehat J_1u\|^2,
\quad
\mu_2
=
\mathbb E\|\Delta\widehat J_2v\|^2,
\quad
\mu_{12}
=
\mathbb E\|\Delta\widehat P[u,v]\|^2.
\]

Then the variance \(\sigma_X^2=\operatorname{Var}(X)\) is finite and satisfies

\[
\sigma_X^2
\le
3\left[
16K_1^4k\mu_1^2
+
16K_2^4k\mu_2^2
+
256K_1^4K_2^4k
\min(d_1,d_2)
\mu_{12}^2
\right].
\tag{4.44}
\]

Partition \(m=b\ell\) independent evaluation pairs into \(b\) equal blocks. Let \(\widetilde{\mathcal E}_{\mathrm{MOM}}\) be the median of the block means. If

\[
b\ge8\log(1/\delta),
\]

then

\[
\boxed{
\left|
\widetilde{\mathcal E}_{\mathrm{MOM}}
-
\widehat{\mathcal E}_Q
\right|
\le
2\sigma_X\sqrt{\frac bm}
}
\tag{4.45}
\]

with probability at least \(1-\delta\).

The constant \(\sigma_X\), or a certified upper bound on it, is required for a numerical confidence interval. If no such certificate exists, confirmatory probes must be normalized or clipped so that the bounded theorem applies.

### Sampling without replacement from a frozen donor pool

Let a frozen finite population contain \(N\) scalar energy values

\[
x_1,\ldots,x_N\in[a,b],
\qquad
\mu_N=\frac1N\sum_{i=1}^N x_i.
\]

Draw \(m\le N\) values uniformly without replacement. Define

\[
\rho_m
=
\begin{cases}
1-\dfrac{m-1}{N},&m\le N/2,\\[6pt]
\left(1-\dfrac mN\right)\left(1+\dfrac1m\right),&m>N/2.
\end{cases}
\tag{4.46}
\]

Then, with probability at least \(1-\delta\),

\[
\boxed{
|\overline X_m-\mu_N|
\le
(b-a)
\sqrt{
\frac{
\rho_m\log(2/\delta)
}{
2m
}
}.
}
\tag{4.47}
\]

The same result applies to each scalar quadratic form

\[
(q^\top x_i)^2
\]

in a finite direction pool. A \(1/4\)-net union bound yields

\[
\boxed{
\|\widehat M-M_N\|_{\mathrm{op}}
\le
2L^2
\sqrt{
\frac{
\rho_m\log(2\cdot9^d/\delta)
}{
2m
}
}
}
\tag{4.48}
\]

when all pool vectors have norm at most \(L\). The finite-population correction and its piecewise \(\rho_m\) follow the Hoeffding–Serfling result.

These bounds are conditional on the frozen donor pool. They do not by themselves justify generalization to a larger superpopulation.

# Complete Proofs

## Proof of the structural derivative identities

Fix one system and suppress the subscript \(s\).

From (3.1),

\[
f(x,z)
=
y+b(x)
+
\sum_{j=1}^{d_2}
c_j
\left[
\psi_j(a_j+A_{j:}x+z_j)-\psi_j(a_j)
\right].
\]

For \(h\in\mathbb R^{d_1}\),

\[
D_1f(0,0)[h]
=
Db(0)[h]
+
\sum_{j=1}^{d_2}
c_j\psi_j'(a_j)A_{j:}h.
\]

In coordinates,

\[
(J_1)_{\alpha i}
=
D_{\alpha i}
+
\sum_{j=1}^{d_2}
c_{j,\alpha}\psi_j'(a_j)A_{ji}.
\]

By the structural definition (3.6),

\[
P_{\alpha i j}
=
c_{j,\alpha}\psi_j'(a_j)A_{ji}.
\]

Therefore

\[
J_1=D+\mathsf S(P).
\]

For \(w\in\mathbb R^{d_2}\),

\[
D_2f(0,0)[w]
=
\sum_{j=1}^{d_2}
c_j\psi_j'(a_j)w_j.
\]

Thus

\[
(J_2)_{\alpha j}
=
c_{j,\alpha}\psi_j'(a_j)
=
G_{\alpha j},
\]

so

\[
J_2=G.
\]

For \(h\in\mathbb R^{d_1}\) and \(w\in\mathbb R^{d_2}\),

\[
D_{12}^2f(0,0)[h,w]
=
\sum_{j=1}^{d_2}
c_j\psi_j''(a_j)
(A_{j:}h)w_j.
\]

Hence

\[
H_{\alpha i j}
=
c_{j,\alpha}\psi_j''(a_j)A_{ji}.
\]

Multiplying by the known ratio

\[
\rho_j=\frac{\psi_j'(a_j)}{\psi_j''(a_j)}
\]

gives

\[
\rho_jH_{\alpha i j}
=
c_{j,\alpha}\psi_j'(a_j)A_{ji}
=
P_{\alpha i j}.
\]

Therefore

\[
P=\mathsf R_\rho H.
\]

Substituting into the first identity gives

\[
D=J_1-\mathsf S(P),
\]

while \(G=J_2\). This proves (4.1).

The derivation also proves that \(P\) is the product of the three edge derivatives along the declared graph path. It was not obtained by renaming \(H\).

## Proof of the structural-map conditioning bound

For every \(P\in\mathbb R^{k\times d_1\times d_2}\),

\[
\|\mathsf S(P)\|_F^2
=
\sum_{\alpha=1}^k
\sum_{i=1}^{d_1}
\left(
\sum_{j=1}^{d_2}P_{\alpha i j}
\right)^2.
\]

By Cauchy–Schwarz,

\[
\left(
\sum_{j=1}^{d_2}P_{\alpha i j}
\right)^2
\le
d_2
\sum_{j=1}^{d_2}
P_{\alpha i j}^2.
\]

Summing over \(\alpha,i\),

\[
\|\mathsf S(P)\|_F
\le
\sqrt{d_2}\|P\|_F.
\tag{5.1}
\]

For \(\theta=(D,G,P)\), its image is

\[
L=(D+\mathsf S(P),G,P).
\]

Using

\[
\|A+B\|_F^2
\le
2\|A\|_F^2+2\|B\|_F^2
\]

and (5.1),

\[
\begin{aligned}
\|L\|_{\mathcal L}^2
&=
\|D+\mathsf S(P)\|_F^2
+\|G\|_F^2
+\|P\|_F^2\\
&\le
2\|D\|_F^2
+
2d_2\|P\|_F^2
+
\|G\|_F^2
+
\|P\|_F^2\\
&=
2\|D\|_F^2
+
\|G\|_F^2
+
(2d_2+1)\|P\|_F^2\\
&\le
(2d_2+1)
\left(
\|D\|_F^2+\|G\|_F^2+\|P\|_F^2
\right).
\end{aligned}
\]

Therefore

\[
\|L\|_{\mathcal L}^2
\le
(2d_2+1)\|\theta\|_\Theta^2.
\tag{5.2}
\]

Conversely,

\[
D=J_1-\mathsf S(P),
\qquad
G=J_2.
\]

Thus

\[
\begin{aligned}
\|\theta\|_\Theta^2
&=
\|J_1-\mathsf S(P)\|_F^2
+
\|J_2\|_F^2
+
\|P\|_F^2\\
&\le
2\|J_1\|_F^2
+
2d_2\|P\|_F^2
+
\|J_2\|_F^2
+
\|P\|_F^2\\
&\le
(2d_2+1)
\left(
\|J_1\|_F^2
+
\|J_2\|_F^2
+
\|P\|_F^2
\right)\\
&=
(2d_2+1)\|L\|_{\mathcal L}^2.
\end{aligned}
\]

Rearranging,

\[
\|L\|_{\mathcal L}^2
\ge
\frac1{2d_2+1}
\|\theta\|_\Theta^2.
\tag{5.3}
\]

Apply (5.2) and (5.3) to the differences between two systems. This proves (4.2).

## Proof of the product-probe energy identity and bounds

For a matrix \(A\in\mathbb R^{k\times d}\),

\[
\begin{aligned}
\mathbb E\|Au\|_2^2
&=
\mathbb E[u^\top A^\top Au]\\
&=
\operatorname{tr}
\left(
A^\top A\mathbb E[uu^\top]
\right)\\
&=
\operatorname{tr}(A^\top AM).
\end{aligned}
\tag{5.4}
\]

Equivalently,

\[
\mathbb E\|Au\|_2^2
=
\|AM^{1/2}\|_F^2.
\]

If

\[
\lambda^-I\preceq M\preceq\lambda^+I,
\]

then

\[
\lambda^-\|A\|_F^2
\le
\mathbb E\|Au\|_2^2
\le
\lambda^+\|A\|_F^2.
\tag{5.5}
\]

For a tensor \(T\), write \(T_\alpha\in\mathbb R^{d_1\times d_2}\) for its output-\(\alpha\) slice. Then

\[
T[u,v]_\alpha=u^\top T_\alpha v.
\]

By independence of \(u\) and \(v\),

\[
\begin{aligned}
\mathbb E[T[u,v]_\alpha^2]
&=
\mathbb E_u
\mathbb E_v
\left[
u^\top T_\alpha vv^\top T_\alpha^\top u
\right]\\
&=
\mathbb E_u
\left[
u^\top T_\alpha M_2T_\alpha^\top u
\right]\\
&=
\operatorname{tr}
\left(
T_\alpha M_2T_\alpha^\top M_1
\right)\\
&=
\left\|
M_1^{1/2}T_\alpha M_2^{1/2}
\right\|_F^2.
\end{aligned}
\]

Summing over \(\alpha\),

\[
\mathbb E\|T[u,v]\|_2^2
=
\sum_{\alpha=1}^k
\left\|
M_1^{1/2}T_\alpha M_2^{1/2}
\right\|_F^2.
\tag{5.6}
\]

The eigenvalue bounds imply

\[
\lambda_1^-\lambda_2^-\|T\|_F^2
\le
\mathbb E\|T[u,v]\|_2^2
\le
\lambda_1^+\lambda_2^+\|T\|_F^2.
\tag{5.7}
\]

Apply (5.5) to \(\Delta J_1,\Delta J_2\), apply (5.7) to \(\Delta P\), and sum. Taking the smallest lower coefficient and largest upper coefficient yields (4.6). Combining with (4.2) yields (4.7) and (4.8).

If \(\mathcal E_Q=0\), all three nonnegative terms in (4.3) vanish. Because both second moments are positive definite, (5.5) and (5.7) imply

\[
\Delta J_1=0,
\qquad
\Delta J_2=0,
\qquad
\Delta P=0.
\]

The structural inverse gives

\[
\Delta G=0,
\qquad
\Delta D
=
\Delta J_1-\mathsf S(\Delta P)
=
0.
\]

Thus \(\Delta\theta=0\). The reverse implication is immediate. This proves (4.9).

## Proof of the central first-order error

Fix \(s\), suppress its subscript, and consider

\[
g(q)=f(qu,0)
\]

for a unit vector \(u\).

Taylor's theorem around \(q=0\) gives

\[
g(r)
=
g(0)
+
rg'(0)
+
\frac{r^2}{2}g''(0)
+
R_+,
\]

\[
g(-r)
=
g(0)
-
rg'(0)
+
\frac{r^2}{2}g''(0)
+
R_-,
\]

where, using the vector-valued third-derivative operator norm,

\[
\|R_+\|_2
\le
\frac{M_{300}r^3}{6},
\qquad
\|R_-\|_2
\le
\frac{M_{300}r^3}{6}.
\]

Subtracting,

\[
g(r)-g(-r)
=
2rg'(0)+R_+-R_-.
\]

Therefore

\[
\left\|
\frac{g(r)-g(-r)}{2r}
-
g'(0)
\right\|_2
\le
\frac{\|R_+\|+\|R_-\|}{2r}
\le
\frac{M_{300}r^2}{6}.
\tag{5.8}
\]

Because

\[
g'(0)=J_1u,
\]

this proves the pointwise \(M_1\) bound. Replacing \(M_{300}\) by \(M_{030}\), \(u\) by \(v\), and \(r\) by \(t\) proves the \(M_2\) bound.

## Proof of the four-point mixed error

Define

\[
F(q,w)=f(qu,wv).
\]

By applying the fundamental theorem of calculus first in \(q\) and then in \(w\),

\[
\begin{aligned}
&
f(ru,tv)
-
f(ru,-tv)
-
f(-ru,tv)
+
f(-ru,-tv)\\
&\qquad
=
\int_{-r}^{r}
\int_{-t}^{t}
D_{12}^2f(qu,wv)[u,v]
\,dw\,dq.
\end{aligned}
\]

Thus the four-point estimator equals

\[
\frac1{4rt}
\int_{-r}^{r}
\int_{-t}^{t}
D_{12}^2f(qu,wv)[u,v]
\,dw\,dq.
\tag{5.9}
\]

Let

\[
h(q,w)
=
D_{12}^2f(qu,wv)[u,v].
\]

For fixed \(w\), the second derivative of \(h\) with respect to \(q\) is

\[
\frac{\partial^2h}{\partial q^2}
=
D_{1112}^4f(qu,wv)[u,u,u,v].
\]

Its norm is at most \(M_{310}\). The symmetric-average identity for a twice differentiable function gives

\[
\left\|
\frac1{2r}\int_{-r}^r h(q,w)\,dq
-
h(0,w)
\right\|_2
\le
\frac{M_{310}r^2}{6}.
\tag{5.10}
\]

To verify the constant, Taylor-expand \(h(q,w)\) around \(q=0\). The linear term integrates to zero, and

\[
\frac1{2r}\int_{-r}^r\frac{q^2}{2}\,dq
=
\frac{r^2}{6}.
\]

Now consider \(h(0,w)\). Its second derivative with respect to \(w\) is

\[
\frac{\partial^2h(0,w)}{\partial w^2}
=
D_{1222}^4f(0,wv)[u,v,v,v],
\]

whose norm is at most \(M_{130}\). Therefore

\[
\left\|
\frac1{2t}\int_{-t}^t h(0,w)\,dw
-
h(0,0)
\right\|_2
\le
\frac{M_{130}t^2}{6}.
\tag{5.11}
\]

Combining (5.9)–(5.11) by the triangle inequality,

\[
\left\|
Y^{(12)}(u,v)-H[u,v]
\right\|_2
\le
\frac{
M_{310}r^2+M_{130}t^2
}{6}.
\tag{5.12}
\]

This proves the pointwise mixed bound used in (4.22).

## Proof of the least-squares error bounds

For the first block, write

\[
Y^{(1)}
=
UJ_1^\top+E_1.
\]

The estimator satisfies

\[
\widehat J_1^\top
=
(U^\top U)^{-1}U^\top Y^{(1)}
=
J_1^\top
+
(U^\top U)^{-1}U^\top E_1.
\]

Therefore

\[
\widehat J_1-J_1
=
E_1^\top U(U^\top U)^{-1}.
\]

The operator norm of

\[
U(U^\top U)^{-1}
\]

is

\[
\frac1{\sigma_{\min}(U)}.
\]

Hence

\[
\|\widehat J_1-J_1\|_F
\le
\frac{\|E_1\|_F}{\sigma_{\min}(U)}.
\]

Because

\[
\sigma_{\min}(U)^2
=
m_1\mu_1
\]

and

\[
\|E_1\|_F^2
\le
m_1\overline e_1^2,
\]

we obtain

\[
\|\widehat J_1-J_1\|_F
\le
\sqrt{
\frac{\overline e_1^2}{\mu_1}
}.
\]

This proves (4.23). The same argument proves (4.24).

For output coordinate \(\alpha\),

\[
Y_\alpha^{(12)}
=
UH_\alpha V^\top+E_{12,\alpha}.
\]

Thus

\[
\widehat H_\alpha-H_\alpha
=
(U^\top U)^{-1}U^\top
E_{12,\alpha}
V(V^\top V)^{-1}.
\]

Therefore

\[
\|\widehat H_\alpha-H_\alpha\|_F
\le
\frac{
\|E_{12,\alpha}\|_F
}{
\sigma_{\min}(U)\sigma_{\min}(V)
}.
\]

Squaring and summing over \(\alpha\),

\[
\|\widehat H-H\|_F^2
\le
\frac{
\|E_{12}\|_F^2
}{
\sigma_{\min}(U)^2
\sigma_{\min}(V)^2
}.
\]

Since

\[
\sigma_{\min}(U)^2=m_1\mu_1,
\qquad
\sigma_{\min}(V)^2=m_2\mu_2,
\]

and

\[
\|E_{12}\|_F^2
\le
m_1m_2\overline e_{12}^2,
\]

we obtain (4.25).

Mode-\(3\) scaling gives

\[
\|\mathsf R_\rho(\widehat H-H)\|_F
\le
\|\rho\|_\infty
\|\widehat H-H\|_F,
\]

which proves (4.26).

Finally,

\[
\widehat D-D
=
(\widehat J_1-J_1)
-
\mathsf S(\widehat P-P).
\]

Using (5.1),

\[
\|\widehat D-D\|_F
\le
\epsilon_1+\sqrt{d_2}\epsilon_P.
\]

Combining the three structural components in direct-sum norm proves (4.28) and (4.29).

For an estimated curvature ratio,

\[
\widehat P-P
=
\mathsf R_{\widehat\rho}(\widehat H-H)
+
\mathsf R_{\widehat\rho-\rho}H.
\]

Taking norms gives (4.30).

## Proof of the finite-radius energy interval

Define the exact direct-sum response difference

\[
Z(u,v)
=
\left(
\Delta J_1u,
\Delta J_2v,
\Delta P[u,v]
\right)
\in\mathbb R^{3k}
\]

and its estimated counterpart \(\widehat Z(u,v)\).

Then

\[
\mathcal E_Q
=
\|Z\|_{L^2(Q_1\otimes Q_2)}^2,
\]

\[
\widehat{\mathcal E}_Q
=
\|\widehat Z\|_{L^2(Q_1\otimes Q_2)}^2.
\]

The reverse triangle inequality in the Hilbert space \(L^2\) gives

\[
\left|
\|\widehat Z\|_{L^2}
-
\|Z\|_{L^2}
\right|
\le
\|\widehat Z-Z\|_{L^2}.
\tag{5.13}
\]

The tensor-estimation errors for a two-system difference are at most \(\eta_1,\eta_2,\eta_P\). By the upper probe-energy bounds,

\[
\|\widehat Z-Z\|_{L^2}^2
\le
\lambda_1^+\eta_1^2
+
\lambda_2^+\eta_2^2
+
\lambda_1^+\lambda_2^+\eta_P^2
=
B_{\mathrm{rad}}^2.
\]

Substituting into (5.13) proves (4.32). Squaring both sides while respecting nonnegativity yields (4.33).

## Proof of the bounded covariance certificate

Let

\[
A=\widehat M-M.
\]

For any fixed unit vector \(q\),

\[
q^\top\widehat Mq
=
\frac1m
\sum_{\ell=1}^m
(q^\top x_\ell)^2.
\]

Because

\[
0\le(q^\top x_\ell)^2\le L^2,
\]

Hoeffding's inequality gives

\[
\Pr\left(
|q^\top Aq|\ge t
\right)
\le
2\exp\left(
-\frac{2mt^2}{L^4}
\right).
\tag{5.14}
\]

Let \(\mathcal N\) be a \(1/4\)-net of the unit sphere. A standard volume argument gives

\[
|\mathcal N|\le9^d.
\]

For a symmetric matrix \(A\),

\[
\|A\|_{\mathrm{op}}
\le
2\max_{q\in\mathcal N}|q^\top Aq|.
\tag{5.15}
\]

To verify (5.15), let \(x\) be a unit vector attaining \(|x^\top Ax|=\|A\|_{\mathrm{op}}\), and choose \(q\in\mathcal N\) with \(\|x-q\|\le1/4\). Then

\[
|x^\top Ax-q^\top Aq|
\le
|(x-q)^\top Ax|
+
|q^\top A(x-q)|
\le
\frac12\|A\|_{\mathrm{op}}.
\]

Thus

\[
\frac12\|A\|_{\mathrm{op}}
\le
|q^\top Aq|.
\]

Apply (5.14) to every \(q\in\mathcal N\), take a union bound, and set

\[
t
=
L^2
\sqrt{
\frac{
\log(2\cdot9^d/\delta)
}{
2m
}
}.
\]

With probability at least \(1-\delta\),

\[
\max_{q\in\mathcal N}|q^\top Aq|
\le t.
\]

Equation (5.15) gives (4.34).

Weyl's inequality gives

\[
\lambda_{\min}(\widehat M)
\ge
\lambda_{\min}(M)
-
\|\widehat M-M\|_{\mathrm{op}}.
\]

If the error is at most \(\lambda_{\min}(M)/2\), then (4.35) follows.

## Proof of bounded energy concentration and the combined interval

Conditional on the tensor-fitting set, \(X_1,\ldots,X_m\) are independent scalar variables in \([0,B_X]\). Hoeffding's inequality gives

\[
\Pr\left(
\left|
\widehat{\mathcal E}_m
-
\widehat{\mathcal E}_Q
\right|
\ge\xi
\right)
\le
2\exp\left(
-\frac{2m\xi^2}{B_X^2}
\right).
\]

Setting the right side to \(\delta\) gives (4.37).

On the same event,

\[
\widehat{\mathcal E}_Q
\le
\widehat{\mathcal E}_m+\xi_m
\]

and

\[
\widehat{\mathcal E}_Q
\ge
(\widehat{\mathcal E}_m-\xi_m)_+.
\]

Apply (4.33) to these upper and lower values. This yields (4.39). Substituting the upper endpoint into (4.8) proves (4.40), and substituting the lower endpoint into the lower direction of (4.7) proves (4.41).

The evaluation pairs must be independent pairs. A Cartesian collection formed by reusing \(m_1\) draws of \(u\) and \(m_2\) draws of \(v\) does not create \(m_1m_2\) independent energy observations. Cartesian reuse is valid for deterministic tensor fitting, but not for an i.i.d. Hoeffding sample-size claim.

## Proof of the sub-Gaussian moment and median-of-means bounds

Let \(A\in\mathbb R^{k\times d_1}\), with rows \(a_\alpha^\top\), and define

\[
X_A=\|Au\|_2^2
=
\sum_{\alpha=1}^k
(a_\alpha^\top u)^2.
\]

Using

\[
\left(
\sum_{\alpha=1}^k z_\alpha^2
\right)^2
\le
k\sum_{\alpha=1}^kz_\alpha^4,
\]

and (4.42) with \(p=4\),

\[
\begin{aligned}
\mathbb E X_A^2
&\le
k\sum_{\alpha=1}^k
\mathbb E(a_\alpha^\top u)^4\\
&\le
16K_1^4k
\sum_{\alpha=1}^k
(a_\alpha^\top M_1a_\alpha)^2\\
&\le
16K_1^4k
\left(
\sum_{\alpha=1}^k
a_\alpha^\top M_1a_\alpha
\right)^2\\
&=
16K_1^4k
\left(
\mathbb E\|Au\|^2
\right)^2.
\end{aligned}
\tag{5.16}
\]

The same argument applies to the \(M_2\) first-order term.

For the mixed tensor \(T\), condition on \(v\). For output coordinate \(\alpha\),

\[
T[u,v]_\alpha
=
\langle T_\alpha v,u\rangle.
\]

Applying (5.16) conditionally,

\[
\mathbb E_u
\left[
\|T[u,v]\|^4
\mid v
\right]
\le
16K_1^4k
\left(
\mathbb E_u[
\|T[u,v]\|^2
\mid v]
\right)^2.
\tag{5.17}
\]

There exists a positive semidefinite \(B\in\mathbb R^{d_2\times d_2}\) such that

\[
\mathbb E_u[
\|T[u,v]\|^2
\mid v]
=
v^\top Bv.
\]

Specifically,

\[
B
=
\sum_{\alpha=1}^k
T_\alpha^\top M_1T_\alpha.
\]

Apply the first-order fourth-moment bound to \(B^{1/2}v\), whose output dimension is at most \(d_2\):

\[
\mathbb E_v(v^\top Bv)^2
\le
16K_2^4d_2
\left(
\mathbb E_vv^\top Bv
\right)^2.
\tag{5.18}
\]

Combining (5.17) and (5.18),

\[
\mathbb E\|T[u,v]\|^4
\le
256K_1^4K_2^4kd_2
\left(
\mathbb E\|T[u,v]\|^2
\right)^2.
\]

Conditioning in the opposite order gives the same inequality with \(d_1\), so the minimum of the two bounds is valid:

\[
\mathbb E\|T[u,v]\|^4
\le
256K_1^4K_2^4k
\min(d_1,d_2)
\left(
\mathbb E\|T[u,v]\|^2
\right)^2.
\tag{5.19}
\]

Now write

\[
X=X_1+X_2+X_{12}.
\]

Because

\[
(X_1+X_2+X_{12})^2
\le
3(X_1^2+X_2^2+X_{12}^2),
\]

(5.16) and (5.19) imply (4.44).

For the median-of-means result, partition the \(m=b\ell\) observations into \(b\) independent blocks. Let \(\overline X_j\) be block \(j\)'s mean. Then

\[
\operatorname{Var}(\overline X_j)
=
\frac{\sigma_X^2}{\ell}.
\]

By Chebyshev,

\[
\Pr\left(
|\overline X_j-\mathbb EX|
>
\frac{2\sigma_X}{\sqrt\ell}
\right)
\le
\frac14.
\]

Call such a block bad. If the median differs from \(\mathbb EX\) by more than \(2\sigma_X/\sqrt\ell\), at least \(b/2\) blocks are bad.

Hoeffding's inequality for the independent bad-block indicators gives

\[
\Pr\left(
\#\text{bad blocks}\ge b/2
\right)
\le
\exp\left(
-2b(1/2-1/4)^2
\right)
=
e^{-b/8}.
\]

For

\[
b\ge8\log(1/\delta),
\]

this probability is at most \(\delta\). Since \(\ell=m/b\),

\[
\frac{2\sigma_X}{\sqrt\ell}
=
2\sigma_X\sqrt{\frac bm}.
\]

This proves (4.45).

## Proof of the without-replacement application

For a fixed scalar finite population \(x_i\in[a,b]\), the Hoeffding–Serfling inequality gives the one-sided deviation with the finite-population factor \(\rho_m\). Applying it to \(x_i\) and \(-x_i\) with failure probability \(\delta/2\) yields (4.47).

For covariance estimation, fix a unit vector \(q\). The finite population

\[
z_i=(q^\top x_i)^2
\]

lies in \([0,L^2]\). Applying (4.47) with failure probability \(\delta/9^d\) to each member of a \(1/4\)-net and then using (5.15) yields (4.48).

# Converses and Counterexamples

## Strongest finite-probe if-and-only-if completeness theorem

Let all declared gates satisfy

\[
\psi_j'(a_j)\neq0,
\qquad
\psi_j''(a_j)\neq0,
\]

so that every curvature ratio is nonzero.

Let \(U\in\mathbb R^{m_1\times d_1}\) contain the first-block directions, \(V\in\mathbb R^{m_2\times d_2}\) contain the second-block directions, and let

\[
(u_n,v_n),
\qquad n=1,\ldots,m_{12},
\]

be arbitrary paired mixed probes.

Define

\[
W
=
\begin{bmatrix}
(v_1\otimes u_1)^\top\\
\vdots\\
(v_{m_{12}}\otimes u_{m_{12}})^\top
\end{bmatrix}
\in
\mathbb R^{m_{12}\times d_1d_2}.
\]

The complete first-plus-mixed measurement map is injective on the reduced ASG structural class if and only if

\[
\operatorname{rank}(U)=d_1,
\tag{6.1}
\]

\[
\operatorname{rank}(V)=d_2,
\tag{6.2}
\]

and

\[
\operatorname{rank}(W)=d_1d_2.
\tag{6.3}
\]

For a Cartesian mixed design containing every pair \((u_a,v_b)\),

\[
W=V\otimes U
\]

up to row ordering, and

\[
\operatorname{rank}(W)
=
\operatorname{rank}(V)
\operatorname{rank}(U).
\]

Therefore a Cartesian design is complete if and only if \(U\) and \(V\) each have full column rank.

### Sufficiency

Full rank of \(U\) uniquely determines \(J_1\) from \(UJ_1^\top\). Full rank of \(V\) uniquely determines \(J_2\). Full rank of \(W\) uniquely determines every vectorized output slice of \(H\). The known curvature ratios then determine \(P\), and the structural inverse determines \(D\) and \(G\).

### Necessity of \(U\)-rank

If \(U\) is rank deficient, choose nonzero

\[
a\in\ker U.
\]

For any nonzero \(c\in\mathbb R^k\), let two systems differ only in

\[
\Delta D=ca^\top.
\]

Then

\[
\Delta J_1u_a=0
\]

for every measured direction, while

\[
\Delta\theta\neq0.
\]

### Necessity of \(V\)-rank

If \(V\) is rank deficient, choose nonzero

\[
b\in\ker V.
\]

Set \(A=0\), so \(P=H=0\). Choose a difference in output columns that produces

\[
\Delta G=cb^\top.
\]

Then every measured \(J_2v_b\) is unchanged despite \(\Delta G\neq0\).

### Necessity of Kronecker rank

Suppose \(W\) is rank deficient. Choose a nonzero matrix

\[
K\in\mathbb R^{d_1\times d_2}
\]

with

\[
W\operatorname{vec}(K)=0.
\]

Use scalar output \(k=1\). Fix nonzero output coefficients \(c_j\) and fixed known activations. Let the target have \(A_\star=0\), and set

\[
A_{p,j:}
=
\frac{K_{:,j}^\top}
     {c_j\psi_j''(a_j)}.
\]

Then

\[
H_p-H_\star=K.
\]

Every measured mixed response is zero because

\[
u_n^\top Kv_n=0.
\]

Set

\[
D_p-D_\star
=
-\mathsf S(P_p-P_\star).
\]

Then

\[
J_{1,p}=J_{1,\star}.
\]

Keep \(c_j\) fixed, so

\[
J_{2,p}=J_{2,\star}.
\]

Because every curvature ratio is nonzero,

\[
P_p-P_\star
=
\mathsf R_\rho K
\neq0.
\]

Thus the structural parameters differ while all measured first and paired mixed responses agree.

This proves the if-and-only-if theorem. Merely having more than one direction is not completeness.

## Rank-deficient population covariance

Let \(a\neq0\) lie in the nullspace of \(M_1\). Then

\[
a^\top u=0
\quad Q_1\text{-almost surely}.
\]

For nonzero \(b\in\mathbb R^{d_2}\) and \(c\in\mathbb R^k\), define

\[
T[u,v]
=
c(a^\top u)(b^\top v).
\]

Then

\[
T\neq0
\]

but

\[
\mathbb E\|T[u,v]\|^2=0.
\]

The analogous construction applies when \(M_2\) is singular. Hence positive covariance on the actual declared subspace is necessary.

## First-order cancellation that mixed probes resolve

Take

\[
d_1=1,\qquad d_2=2,\qquad k=1,
\]

with

\[
\psi_1(t)=\psi_2(t)=t+\frac12t^2,
\qquad
a_1=a_2=0,
\qquad
c_1=c_2=1.
\]

For the target, set

\[
A_\star=
\begin{bmatrix}
1\\
-1
\end{bmatrix},
\qquad
D_\star=0.
\]

For the patched system, set

\[
A_p=
\begin{bmatrix}
0\\
0
\end{bmatrix},
\qquad
D_p=0.
\]

Both have the same zero-order output. Since \(\psi'(0)=1\),

\[
J_{1,\star}
=
1+(-1)
=
0,
\]

\[
J_{1,p}=0.
\]

Also

\[
J_{2,\star}=J_{2,p}=[1,1].
\]

Thus every first-order response agrees.

But \(\psi''(0)=1\), so

\[
H_\star=[1,-1],
\qquad
H_p=[0,0].
\]

The mixed response distinguishes the systems. This is a genuine path cancellation rather than a rank-deficient-probe artifact.

## Unknown activation-curvature ratio

Remove the assumption that the activation function and anchor are known.

Consider scalar systems with one gate.

System A:

\[
\psi_A(t)=t+\frac12t^2,
\qquad
A_A=1,
\qquad
c_A=1,
\qquad
D_A=-1.
\]

At zero,

\[
J_{1,A}=0,
\qquad
J_{2,A}=1,
\qquad
H_A=1,
\qquad
P_A=1.
\]

System B:

\[
\psi_B(t)=t+\frac14t^2,
\qquad
A_B=2,
\qquad
c_B=1,
\qquad
D_B=-2.
\]

At zero,

\[
\psi_B'(0)=1,
\qquad
\psi_B''(0)=\frac12.
\]

Hence

\[
J_{1,B}
=
-2+1\cdot1\cdot2
=
0,
\]

\[
J_{2,B}=1,
\]

\[
H_B
=
1\cdot\frac12\cdot2
=
1,
\]

but

\[
P_B
=
1\cdot1\cdot2
=
2.
\]

Thus the complete raw response tuple \((J_1,J_2,H)\) is identical while the reduced path gain differs. The known ratio \(\psi'/\psi''\) is necessary.

## Zero activation curvature

Let

\[
\psi(t)=t,
\qquad
c=1.
\]

For any scalar \(A\), set

\[
D=-A.
\]

Then

\[
J_1=D+c\psi'(0)A=0,
\]

\[
J_2=1,
\]

\[
H=c\psi''(0)A=0,
\]

for every \(A\), while

\[
P=A.
\]

Therefore no first-plus-mixed response theorem can identify \(P\) at a locally linear gate without some additional intervention or structural information. Nonzero known curvature is load-bearing.

## Hidden reparameterization and full-weight nonidentification

Consider a linear hidden computation

\[
f(x)=CAx.
\]

For every invertible \(R\),

\[
A'=RA,
\qquad
C'=CR^{-1}
\]

gives

\[
C'A'
=
CR^{-1}RA
=
CA.
\]

The two systems agree on every input and therefore on every derivative of every order, yet their individual hidden weights differ.

This proves that complete response tensors do not identify arbitrary internal weights under hidden-basis transformations. The fixed-basis reduced path gain \(CA\) may be identified; \(A\) and \(C\) separately are not, absent additional fixed-gate information.

## Output-null mediator channel

In the ASG class, let

\[
c_j=0.
\]

Then for every \(A_{j:}\),

\[
G_{:,j}=0,
\qquad
H_{:,:,j}=0,
\qquad
P_{:,:,j}=0.
\]

The upstream edge \(A_{j:}\) is not identifiable. It is also irrelevant to the measured output through the declared path. This is why the theorem's unconditional target is the reduced path gain, not every latent edge.

## Omitted mixed bypass

Suppose the true output is decomposed into a declared path contribution and an omitted bypass:

\[
f(u,v)
=
f_{\mathrm{path}}(u,v)
+
f_{\mathrm{bypass}}(u,v).
\]

Take scalar variables and define system A by

\[
f_{\mathrm{path},A}(u,v)=uv,
\]

\[
f_{\mathrm{bypass},A}(u,v)=-uv.
\]

Then

\[
f_A(u,v)=0
\]

for every \(u,v\), even though the declared path has mixed gain \(1\).

Define system B by

\[
f_{\mathrm{path},B}(u,v)=0,
\qquad
f_{\mathrm{bypass},B}(u,v)=0.
\]

All zero-, first-, mixed-, and higher-order responses of the total output agree exactly, but the declared path effects differ.

More generally, for any decomposition

\[
f=p+b
\]

and any smooth \(q\),

\[
p'=p+q,
\qquad
b'=b-q
\]

produce the same total map. Therefore no observation of the total downstream response, even with infinitely many perfect local probes, can identify the path/bypass allocation unless the graph or an edge intervention rules out such reallocations.

ASG-5 does exactly that by requiring the bypass to be independent of \(z\).

## Full marginal spans but incomplete paired mixed probes

Let

\[
d_1=d_2=2
\]

and use only the paired probes

\[
(u_1,v_1)=(e_1,e_1),
\qquad
(u_2,v_2)=(e_2,e_2).
\]

The \(u\)-marginals span \(\mathbb R^2\), and the \(v\)-marginals span \(\mathbb R^2\). Nevertheless,

\[
W=
\begin{bmatrix}
(e_1\otimes e_1)^\top\\
(e_2\otimes e_2)^\top
\end{bmatrix}
\]

has rank \(2<4\).

The tensor with only

\[
T_{1,2}=1
\]

nonzero satisfies

\[
T[e_1,e_1]=T[e_2,e_2]=0.
\]

Thus full marginal spans do not imply mixed completeness for arbitrary pairings.

## Finite-radius failure without smoothness control

For any fixed radius \(r>0\), define

\[
g(t)=a\sin\left(\frac{\pi t}{r}\right).
\]

Then

\[
g(r)=g(-r)=0,
\]

so the central finite difference at radius \(r\) is zero. But

\[
g'(0)=\frac{a\pi}{r},
\]

which can be arbitrarily large.

For a mixed example, define

\[
h(u,v)
=
a
\sin\left(\frac{\pi u}{r}\right)
\sin\left(\frac{\pi v}{t}\right).
\]

All four corners \((\pm r,\pm t)\) equal zero, but

\[
\frac{\partial^2h}{\partial u\,\partial v}(0,0)
=
\frac{a\pi^2}{rt}.
\]

Therefore a finite-radius measurement does not identify a derivative without a uniform smoothness bound or a limiting-radius argument.

## Local agreement does not imply global algorithm identity

Let

\[
f_0(x,z)=0
\]

and

\[
f_1(x,z)=\gamma x_1^3.
\]

At the center, both systems have the same output, all first derivatives agree, and all \(x\)-\(z\) mixed second derivatives agree. They differ away from the center.

No finite collection of derivatives below the first nonzero order proves global algorithm identity.

# Path-Specific Target

## Exact nested path intervention in the ASG graph

The path of interest is

\[
X
\longrightarrow
Z
\longrightarrow
W
\longrightarrow
Y.
\]

Let \(\delta\in\mathbb R^{d_1}\) be an upstream displacement.

The gate state naturally induced by \(\delta\) is

\[
Z_s(\delta,0)
=
a_s+A_s\delta.
\]

To isolate the declared path, hold the direct \(X\)-to-\(Y\) bypass at its baseline value while setting the gates to the state induced by \(\delta\). The exact vector path-specific effect is

\[
\boxed{
\tau_s^{\mathrm{path}}(\delta)
=
\sum_{j=1}^{d_2}
c_{s,j}
\left[
\psi_j(a_{s,j}+A_{s,j:}\delta)
-
\psi_j(a_{s,j})
\right].
}
\tag{7.1}
\]

Equivalently,

\[
\tau_s^{\mathrm{path}}(\delta)
=
\left[
f_s(\delta,0)-f_s(0,0)
\right]
-
b_s(\delta).
\]

This is an absolute output-space effect, not a normalized recovery ratio.

## Relation to the identified tensor

Differentiating (7.1) at \(\delta=0\),

\[
D\tau_s^{\mathrm{path}}(0)
=
\mathsf S(P_s).
\tag{7.2}
\]

The identified local path effect is therefore

\[
\boxed{
\tau_{s,\mathrm{lin}}^{\mathrm{path}}(\delta)
=
\mathsf S(P_s)\delta.
}
\tag{7.3}
\]

The individual tensor entry

\[
P_{s,\alpha i j}
\]

is the contribution of upstream coordinate \(i\), through gate \(j\), to output coordinate \(\alpha\) at first order.

## Finite-displacement path-effect bound

Suppose

\[
|\psi_j''(t)|
\le
L_{2,s,j}
\]

on every interval between \(a_{s,j}\) and \(a_{s,j}+A_{s,j:}\delta\).

Taylor's theorem gives

\[
\left|
\psi_j(a_{s,j}+A_{s,j:}\delta)
-
\psi_j(a_{s,j})
-
\psi_j'(a_{s,j})A_{s,j:}\delta
\right|
\le
\frac{L_{2,s,j}}2
(A_{s,j:}\delta)^2.
\]

Thus

\[
\begin{aligned}
\|
\tau_s^{\mathrm{path}}(\delta)
-
\tau_{s,\mathrm{lin}}^{\mathrm{path}}(\delta)
\|_2
&\le
\frac12
\sum_{j=1}^{d_2}
\|c_{s,j}\|_2
L_{2,s,j}
(A_{s,j:}\delta)^2\\
&\le
\frac12
\left[
\sum_{j=1}^{d_2}
\|c_{s,j}\|_2
L_{2,s,j}
\|A_{s,j:}\|_2^2
\right]
\|\delta\|_2^2.
\end{aligned}
\tag{7.4}
\]

This explicitly delimits the local approximation.

## Exact nonlinear path effect from active-edge recovery

If, for every path-relevant gate,

\[
\psi_j'(a_{s,j})\neq0
\]

and

\[
\|G_{s,:,j}\|_2>0,
\]

then (3.12) and (3.13) recover \(c_{s,j}\) and \(A_{s,j:}\). Substituting them into (7.1) recovers the exact nonlinear path effect throughout the neighborhood on which the ASG structural equation is valid.

For an estimated active edge, let

\[
g_j=\|G_{:,j}\|_2
\]

and suppose

\[
\|\widehat G_{:,j}-G_{:,j}\|_2
\le
\frac{g_j}{2}.
\]

Then \(\|\widehat G_{:,j}\|\ge g_j/2\), and

\[
\|\widehat A_{j:}-A_{j:}\|_2
\le
\frac{2}{g_j}
\left[
\|\widehat P_{:,:,j}-P_{:,:,j}\|_F
+
\|\widehat G_{:,j}-G_{:,j}\|_2
\|A_{j:}\|_2
\right].
\tag{7.5}
\]

Detailed edge recovery is therefore explicitly conditioned on an output-visibility floor. Reduced \(P\) recovery is not.

## Scalar Greater-Than margin target

For cell \(c\), let

\[
\ell_c\in\mathbb R^k
\]

be a fixed linear contrast between the correct and incorrect two-digit-token logits.

The exact scalar path-specific effect is

\[
\operatorname{PSE}_{s,c}(\delta_c)
=
\ell_c^\top
\tau_s^{\mathrm{path}}(\delta_c).
\tag{7.6}
\]

The identified local version is

\[
\operatorname{PSE}_{s,c}^{\mathrm{lin}}(\delta_c)
=
\ell_c^\top
\mathsf S(P_s)\delta_c.
\tag{7.7}
\]

The absolute structural mismatch target is

\[
\boxed{
T_c
=
\left|
\operatorname{PSE}_{p,c}
-
\operatorname{PSE}_{\star,c}
\right|.
}
\tag{7.8}
\]

Its local approximation is

\[
T_c^{\mathrm{lin}}
=
\left|
\ell_c^\top
\left[
\mathsf S(P_p)-\mathsf S(P_\star)
\right]
\delta_c
\right|.
\tag{7.9}
\]

No denominator is required. Units are logit-margin units.

## Why the proposed proxies are not substitutes

### Attention probability

An attention probability is one softmax coefficient. It omits:

- the corresponding value vector;
- the output projection;
- cancellation across heads;
- downstream MLP transformations;
- residual bypasses;
- the task readout.

Equal attention probabilities can therefore coexist with different path gains.

### Activation cosine

Cosine similarity discards magnitude:

\[
\cos(x,\alpha x)=1
\]

for every \(\alpha>0\), while path effects scale with \(\alpha\). It is also coordinate-dependent and cannot determine edge products.

### Normalized recovery with a near-zero denominator

For

\[
R=\frac{n}{d},
\]

a numerator perturbation \(\varepsilon\) changes the score by

\[
\frac{\varepsilon}{d}.
\]

As \(d\to0\), the sensitivity is unbounded. A recovery ratio therefore cannot be treated as a structural ground truth unless the denominator has a preregistered positive lower bound.

### Final behavioral restoration

Final restoration measures a zero-order equality such as

\[
\ell^\top f_p(0,0)
=
\ell^\top f_\star(0,0).
\]

The cancellation and zero-order counterexamples in Section 6 show that it does not determine \(J_1\), \(J_2\), \(H\), or \(P\).

# Greater-Than Mapping

## Published circuit and proposed variables

The published Greater-Than analysis identifies late MLPs and a set of attention heads as important components. It reports that MLP \(8\) relies on heads \(8.11,8.8,7.10,6.9,5.5,5.1\), that MLP \(9\) relies on head \(9.1\), and that the MLPs rely heavily on upstream MLPs.  The circuit diagram also contains direct head-to-logit, head-to-MLP, MLP-to-MLP, and MLP-to-logit routes.

The proposed raw mapping is:

| Theory object | Proposed GPT-2 object |
|---|---|
| \(M_1\) | Residual-stream writes of heads \(5.1,5.5,6.9,7.10,8.8,8.11,9.1\) at the answer/end position |
| \(M_2\) | Residual-stream writes of MLPs \(8,9,10,11\) at the answer/end position |
| \(Y\) | Greater-Than output logits or a correct-versus-incorrect logit margin |
| \(Q_1,Q_2\) | Disjoint held-out clean donor states under a template/century/distance-bin conditional product law |
| Desired target | Absolute path effect through \(M_1\to M_2\to Y\) |

The source analysis takes the relevant component representations at the end position.

## Recommended output vector definition

The estimator should operate on a vector of eligible two-digit-token logits,

\[
y_s\in\mathbb R^k,
\]

rather than collapsing to one scalar during tensor fitting.

For cell \(c\), let \(\mathcal V_c^+\) be eligible tokens representing years greater than the start-year suffix and \(\mathcal V_c^-\) the eligible incorrect tokens. Define

\[
(\ell_c)_t
=
\begin{cases}
|\mathcal V_c^+|^{-1},&t\in\mathcal V_c^+,\\
-|\mathcal V_c^-|^{-1},&t\in\mathcal V_c^-,\\
0,&\text{otherwise}.
\end{cases}
\tag{8.1}
\]

Then

\[
m_{s,c}
=
\ell_c^\top y_s
\tag{8.2}
\]

is an absolute mean-correct-minus-mean-incorrect logit margin.

Keeping \(k>1\) during tensor estimation preserves output-dimension factors and permits mediator-slice factorization checks. The scalar margin is applied afterward as a fixed linear functional.

## Topological failure of the proposed two-block cut

The proposed \(M_1\) block is not wholly upstream of the proposed \(M_2\) block.

In GPT-2 block order:

\[
\text{attention at layer }8
\to
\text{MLP }8
\to
\text{attention at layer }9
\to
\text{MLP }9.
\]

Therefore head \(9.1\) is downstream of MLP \(8\) but upstream of MLP \(9\). The published path-patching result reflects this ordering: MLP \(8\) depends on earlier heads, while MLP \(9\) depends on head \(9.1\).

Consequently, there is no acyclic two-block structural graph in which all proposed \(M_1\) coordinates precede all proposed \(M_2\) coordinates. A factorial intervention on the concatenated blocks cannot be interpreted as one \(M_1\to M_2\) mixed derivative.

## Gate-coordinate failure

The ASG inverse requires \(M_2\) to be an additive intervention at known elementwise gate inputs:

\[
a_j+A_{j:}x+z_j.
\]

The proposed \(M_2\) variables are full MLP residual outputs. At those outputs:

- there is no known scalar activation \(\psi_j\) applied coordinatewise after the intervention;
- no observed anchor \(a_j\) supplies a ratio \(\psi'_j(a_j)/\psi''_j(a_j)\);
- the coordinates have already been linearly mixed by the MLP output projection;
- subsequent transformer blocks mix them through layer normalization, attention, further MLPs, and residual additions.

Thus the raw mixed derivative with respect to head writes and MLP writes is a total downstream Hessian. It is not an invertible curvature-weighted version of the desired path gain.

## Complete-cut failure

The Greater-Than circuit contains:

- direct head-to-logit contributions;
- head-to-MLP contributions;
- MLP-to-MLP contributions;
- direct MLP-to-logit contributions;
- contributions from components outside the declared circuit;
- interleaved residual and normalization paths.

The source analysis explicitly reports direct and indirect contributions of different magnitudes for MLPs \(8\)–\(11\).

Therefore the total mixed derivative can be written schematically as

\[
H_{\mathrm{total}}
=
H_{\mathrm{desired\ path}}
+
H_{\mathrm{serial\ MLP}}
+
H_{\mathrm{direct\ residual}}
+
H_{\mathrm{normalization}}
+
H_{\mathrm{attention}}
+
H_{\mathrm{other\ bypass}}.
\]

The proposed measurements observe only the sum. Section 6 proves that no decomposition of this sum is identifiable without either a complete-cut theorem or edge-specific interventions that separately observe the terms.

## Probe-dimension lower bound

If the seven head writes are represented as separate \(768\)-dimensional residual vectors, the raw \(M_1\) dimension is

\[
d_{1,\mathrm{raw}}=7\cdot768=5376.
\]

Four MLP residual writes give

\[
d_{2,\mathrm{raw}}=4\cdot768=3072.
\]

Identification of an unrestricted mixed tensor on these raw spaces requires Kronecker rank

\[
d_{1,\mathrm{raw}}d_{2,\mathrm{raw}}
=
16{,}515{,}072.
\]

No small donor-probe design is probe-complete in that space. Restricting to lower-dimensional predeclared subspaces is mathematically permissible, but the theorem and every claim must then be restricted to those subspaces. Effective rank or “many directions” does not turn a rank-deficient design into global circuit identification.

For the gate theorem, an arbitrary PCA rotation of \(M_2\) is also invalid because it destroys the coordinatewise activation-curvature relation. A gate-side reduction must preserve actual gate coordinates.

## Assumptions that are directly testable

The following can be checked from code, hooks, and logged arrays:

- exact topological order of every mediator;
- whether the intervention is at a gate input or a residual output;
- exact activation function and anchor;
- \(\psi_j'(a_j)\), \(\psi_j''(a_j)\), and \(\rho_j\);
- probe design rank and singular values;
- donor split disjointness;
- product sampling of block donors;
- finite-radius stability;
- whether each recovered mediator slice approximately satisfies
  \[
  P_{:,:,j}=G_{:,j}A_{j:};
  \]
- output-vector and logit-contrast definitions.

## Assumptions that remain modeling assumptions

These cannot be certified merely by observing a small mixed residual:

- that the declared graph contains every mixed bypass;
- that no unmeasured path cancels the declared path;
- that a local path gain captures a global algorithm;
- that the chosen subspace contains every semantically relevant direction;
- that an apparent circuit is unique among behaviorally equivalent parameterizations.

## Binding Greater-Than conclusion

The current proposed hooks do not instantiate the positive theorem. In particular:

\[
\boxed{
\text{raw head-output/MLP-output mixed responses}
\not\Rightarrow
\text{identified }M_1\to M_2\to\text{margin path gain}.
}
\]

This is an identification failure, not a request for more statistical power. No amount of GPU sampling repairs it.

# Implementation Contract

## Required new module boundary

The structural estimator must be implemented separately from the existing first-order IRS utilities, for example as:

```text
src/mixed_path_identification.py
src/test_mixed_path_identification.py
src/run_mixed_path_synthetic.py
```

The existing implementation computes forward or symmetric first differences with arrays shaped `[item, probe, output]` and then averages squared discrepancies over both the probe and output axes.  Its synthetic runner tests known gradient gaps rather than a mixed structural inverse.  Those functions may remain as local-response diagnostics, but their outputs must not be relabeled as the structural estimator specified here.

## Canonical index order

Use:

- \(s\): system, patched or target;
- \(c\): semantic cell;
- \(n\): prompt/item nested within cell;
- \(a\): \(M_1\) design direction;
- \(b\): \(M_2\) design direction;
- \(\alpha\): output coordinate;
- \(i\): declared \(M_1\) coordinate;
- \(j\): declared \(M_2\) gate coordinate;
- \(\sigma,\tau\in\{-1,+1\}\): corner signs.

Per center, the canonical arrays are:

| Array | Shape |
|---|---|
| `U` | `[m1, r1]` |
| `V` | `[m2, r2]` |
| `radius1` | `[m1]` |
| `radius2` | `[m2]` |
| `rho` | `[r2]` |
| `first1_corner` | `[m1, 2, k]` |
| `first2_corner` | `[m2, 2, k]` |
| `mixed_corner` | `[m1, m2, 2, 2, k]` |
| `Y1` | `[m1, k]` |
| `Y2` | `[m2, k]` |
| `Y12` | `[m1, m2, k]` |
| `J1` | `[k, r1]` |
| `J2` | `[k, r2]` |
| `H` | `[k, r1, r2]` |
| `P` | `[k, r1, r2]` |
| `D` | `[k, r1]` |
| `G` | `[k, r2]` |

Batched execution may prepend

```text
[system, cell, item]
```

to every array.

## Basis requirements

Let

\[
B_1\in\mathbb R^{d_{1,\mathrm{raw}}\times r_1}
\]

be a fixed orthonormal upstream basis. It may be:

- a coordinate selector;
- a fixed orthonormal basis fitted only on the donor/basis split.

Let

\[
B_2\in\mathbb R^{d_{2,\mathrm{raw}}\times r_2}
\]

be the \(M_2\) basis. For the positive theorem it must be a gate-coordinate selector or a signed permutation of selected gate coordinates. Arbitrary rotations are prohibited.

For a donor displacement \(\delta\), define its projected coordinate vector

\[
q=B^\top\delta.
\]

If \(q\neq0\), store

\[
r=\|q\|_2,
\qquad
u=q/r.
\]

The actual intervention is

\[
\pm rBu.
\]

Log:

- donor ID;
- raw displacement norm;
- projected norm;
- projection residual norm;
- unit direction;
- actual radius;
- endpoint-support diagnostics.

## Exact first-order estimator

For every direction \(u_a\) and radius \(r_a\):

```text
y_plus  = evaluate(center_M1 + r_a * B1 @ u_a, center_M2)
y_minus = evaluate(center_M1 - r_a * B1 @ u_a, center_M2)

Y1[a] = (y_plus - y_minus) / (2 * r_a)
```

For \(M_2\):

```text
y_plus  = evaluate(center_M1, center_M2 + t_b * B2 @ v_b)
y_minus = evaluate(center_M1, center_M2 - t_b * B2 @ v_b)

Y2[b] = (y_plus - y_minus) / (2 * t_b)
```

The same directions, radii, donor identities, output coordinates, and corner definitions must be used for patched and target systems.

## Exact mixed estimator

For every Cartesian pair \((a,b)\):

```text
y_pp = evaluate(+r_a * u_a, +t_b * v_b)
y_pm = evaluate(+r_a * u_a, -t_b * v_b)
y_mp = evaluate(-r_a * u_a, +t_b * v_b)
y_mm = evaluate(-r_a * u_a, -t_b * v_b)

Y12[a, b] = (y_pp - y_pm - y_mp + y_mm) / (4 * r_a * t_b)
```

The sign order must be exactly

\[
(+,+)-(+,-)-(-,+)+(-,-).
\]

The denominator is the product of the two actual probe radii. A single nominal radius must not replace variable donor-chord lengths.

## Tensor fitting

Use QR or SVD-backed least squares. Do not explicitly invert matrices in production code.

Mathematically:

```text
J1_T = solve(U.T @ U, U.T @ Y1)
J1   = J1_T.T

J2_T = solve(V.T @ V, V.T @ Y2)
J2   = J2_T.T

for alpha in 0..k-1:
    H[alpha] =
        solve(U.T @ U,
              U.T @ Y12[:, :, alpha] @ V)
        @ inverse(V.T @ V)
```

The final implementation should use two linear solves rather than constructing either inverse.

Then:

```text
for j in 0..r2-1:
    P[:, :, j] = rho[j] * H[:, :, j]

G = J2
D = J1 - sum(P, axis=M2_coordinate)
```

No ridge penalty is permitted in the confirmatory estimator. Ridge can return a number under rank deficiency, but it cannot restore identification. A deficient design must fail the gate.

## Arbitrary paired mixed probes

For non-Cartesian paired probes, construct

\[
W_{n,:}
=
(v_n\otimes u_n)^\top.
\]

For each output coordinate,

\[
\operatorname{vec}(\widehat H_\alpha)
=
(W^\top W)^{-1}W^\top y_\alpha.
\]

Require

\[
\operatorname{rank}(W)=r_1r_2.
\]

Full marginal ranks of \(U\) and \(V\) are insufficient.

## Probe coverage diagnostics

For each cell and block, compute

\[
\widehat M_1=\frac{U^\top U}{m_1},
\qquad
\widehat M_2=\frac{V^\top V}{m_2}.
\]

Report:

- algebraic rank;
- all singular values;
- \(\lambda_{\min}\);
- \(\lambda_{\max}\);
- condition number;
- effective rank;
- bounded or finite-population lower confidence bound on \(\lambda_{\min}\);
- rank of the paired Kronecker design.

Numerical rank in float64 is defined by

\[
\sigma_i
>
10^3\epsilon_{\mathrm{mach}}
\max(m,d)\sigma_{\max}.
\]

Scientific stability additionally requires protocol-frozen lower-eigenvalue and condition-number thresholds selected using donor/development data before confirmatory cells. If the lower confidence bound on \(\lambda_{\min}\) is nonpositive, no probe-complete claim is allowed.

## Curvature diagnostics

For every gate and center, log

\[
a_j,
\quad
\psi_j'(a_j),
\quad
\psi_j''(a_j),
\quad
\rho_j.
\]

Fail structural identification if:

\[
\psi_j''(a_j)=0,
\]

the activation/hook does not supply a known gate coordinate, or \(\rho_j\) is nonfinite.

Report

\[
\gamma=\min_j|\psi_j''(a_j)|,
\qquad
\rho_{\max}=\max_j|\rho_j|.
\]

Large \(\rho_{\max}\) must appear explicitly in the finite-radius error bound. It must not be hidden by normalization.

## Factorization diagnostic

The positive class implies, for each gate \(j\),

\[
P_{:,:,j}
=
G_{:,j}A_{j:},
\]

so each nonzero gate slice is rank one across output and upstream coordinates.

Define

\[
R_j^{\mathrm{fact}}
=
\min_{a\in\mathbb R^{r_1}}
\frac{
\|P_{:,:,j}-G_{:,j}a^\top\|_F
}{
\max(\|P_{:,:,j}\|_F,\tau_{\mathrm{num}})
}.
\tag{9.1}
\]

The minimizer is

\[
a^\top
=
\frac{G_{:,j}^\top P_{:,:,j}}
     {\|G_{:,j}\|_2^2}
\]

when \(G_{:,j}\neq0\).

A large residual falsifies the separable-gate model. A small residual is necessary but does not prove that no omitted bypass exists.

## Finite-radius and half-radius checks

For every probe direction and donor identity, evaluate the same estimator at:

\[
(r,t),
\qquad
(r/2,t/2).
\]

Preferably also evaluate

\[
(r/4,t/4)
\]

on development cells.

For any estimated tensor \(T\), define

\[
\Delta_r(T)
=
\|\widehat T(r,t)-\widehat T(r/2,t/2)\|_F.
\]

The primary confirmatory stability rule inherited from the planned protocol is:

- cell-score Spearman correlation between full and half radius at least \(0.90\);
- median absolute cell-score change no more than \(20\%\).

These are falsification diagnostics, not proofs of smoothness.

With three radii and a leading \(r^2+t^2\) error, the ratio

\[
\frac{\Delta_r(T)}
     {\Delta_{r/2}(T)}
\]

should approach \(4\) when both radii are scaled together and the leading coefficient is nonzero. Failure of this pattern falsifies the assumed local regime but success does not establish the complete-cut assumption.

Richardson extrapolation

\[
T_{\mathrm R}
=
\frac{
4\widehat T(r/2,t/2)-\widehat T(r,t)
}{3}
\]

may be used only when sufficient fifth- and sixth-order smoothness exists to justify the next even-order term.

## Absolute response energy

Use an independent evaluation-probe set after fitting the tensors.

For each evaluation pair:

```text
z1  = delta_J1 @ u
z2  = delta_J2 @ v
z12 = contract(delta_P, u, v)

X = squared_norm(z1) + squared_norm(z2) + squared_norm(z12)
```

Report

\[
\widehat{\mathcal E}
=
\operatorname{mean}(X).
\]

The theorem uses a sum over output coordinates. If an implementation reports per-output MSE,

\[
\operatorname{MSE}_{\mathrm{per\ output}}
=
\widehat{\mathcal E}/k,
\]

it must label that normalization explicitly. It must not silently equate per-output RMSE with the Frobenius structural energy.

## Target-energy and denominator admissibility

The primary score is absolute and has no denominator.

If a normalized response score is additionally reported, define the target energy

\[
S_{\star,Q}
=
\mathbb E
\left[
\|J_{1,\star}u\|^2
+
\|J_{2,\star}v\|^2
+
\|P_\star[u,v]\|^2
\right].
\]

Use an independent evaluation sample and compute a lower confidence bound

\[
\underline S_{\star,Q}.
\]

A normalized score is admissible only if a threshold \(\tau_{\mathrm{energy}}>0\) was frozen before confirmatory evaluation and

\[
\underline S_{\star,Q}
\ge
\tau_{\mathrm{energy}}.
\]

If no such threshold is present in the locked protocol, the normalized score must not be used as a confirmatory endpoint.

For the path-specific target, use the previously declared absolute conditioning criterion:

\[
|\operatorname{PSE}_{\mathrm{clean}}
-
\operatorname{PSE}_{\mathrm{corrupt}}|
\ge0.10
\]

logit-margin units, or at least \(0.25\) development-cell standard deviations. This criterion must be evaluated without using the patched-target mismatch outcome.

## Path-specific target implementation

For the local target:

```text
serial_gain = sum(P, axis=M2_coordinate)     # [k, r1]
path_vector = serial_gain @ delta_upstream   # [k]
pse         = dot(logit_contrast, path_vector)
```

For two systems:

```text
T_cell_item = abs(pse_patched - pse_target)
```

For an exact nonlinear ASG target, first recover active \(A_j,c_j\), then evaluate

```text
path_vector =
    sum_j c_j * (
        psi_j(a_j + A_j @ delta_upstream)
        - psi_j(a_j)
    )
```

Do not evaluate the nonlinear formula on inactive or curvature-degenerate channels.

## Data splits

The minimum split structure is:

1. **Donor/basis split \(D\):** conditional donor pools and any fixed \(M_1\) basis.
2. **Development cells:** radius, numerical thresholds, admissibility thresholds, and code debugging.
3. **Confirmatory cells:** untouched semantic cells.
4. **Within-cell tensor-design probes:** used to fit \(J_1,J_2,H\).
5. **Within-cell energy-evaluation probes:** independent of tensor-design probes.
6. **Optional support/calibration split:** used only for declared endpoint-support diagnostics, never to repair identification.

No confirmatory outcome may be used to change:

- mediator sites;
- gate coordinates;
- output tokens;
- basis rank;
- radius;
- curvature threshold;
- rank threshold;
- donor condition;
- cell definition.

## Cell-level aggregation and uncertainty

Prompts and probes are nested observations. The independent scientific unit is the semantic cell.

For each cell \(c\):

1. estimate item-level tensors;
2. aggregate item-level absolute path effects using the protocol-frozen mean or median;
3. obtain one cell-level structural target \(T_c\);
4. obtain one cell-level method score.

For comparisons between methods:

- resample cells, not prompts, probes, heads, or layers;
- keep all nested observations together when a cell is resampled;
- use paired cell bootstraps because every method sees the same cells and probes;
- report finite-probe analytic uncertainty separately from across-cell uncertainty.

Overlapping donor pools across cells invalidate a claim that donor draws are independent scientific replicates. Conditional finite-probe intervals remain valid within their frozen pools; confirmatory inferential resampling remains at the cell level.

## Numerical invariants

The implementation must enforce:

- model evaluation mode with stochastic layers disabled;
- identical center replay before every intervention batch;
- float64 accumulation for responses, normal equations, norms, and diagnostics;
- paired directions and radii across systems;
- paired sign corners;
- deterministic unshuffling after randomized corner-execution order;
- no zero radii;
- no NaN or infinity;
- exact basis and output-vocabulary hashes in artifacts;
- exact donor IDs in artifacts;
- no ridge fallback;
- failure on rank deficiency;
- failure on unknown gate activation or curvature;
- separation of tensor-design and energy-evaluation probes;
- preservation of the output axis until a fixed \(\ell_c\) is applied.

# Synthetic Verification Specification

## Base analytic ASG model

Use

\[
d_1=2,
\qquad
d_2=2,
\qquad
k=2.
\]

Let

\[
\psi(t)=t+\frac12t^2
\]

coordinatewise, with anchor \(a=0\). Then

\[
\psi'(0)=1,
\qquad
\psi''(0)=1,
\qquad
\rho=(1,1).
\]

Set

\[
A=
\begin{bmatrix}
1 & -2\\
0.5 & 1
\end{bmatrix},
\]

\[
C=
\begin{bmatrix}
2 & -1\\
1 & 3
\end{bmatrix},
\]

\[
D=
\begin{bmatrix}
0.25 & -0.5\\
1 & 0.75
\end{bmatrix},
\]

\[
y_0=
\begin{bmatrix}
0.1\\
-0.2
\end{bmatrix}.
\]

Define

\[
f(u,v)
=
y_0
+
Du
+
C\left[
\psi(Au+v)-\psi(0)
\right].
\tag{10.1}
\]

The ground-truth gate gain is

\[
G=C
=
\begin{bmatrix}
2 & -1\\
1 & 3
\end{bmatrix}.
\]

The path tensor slices are

\[
P_{:,:,1}
=
\begin{bmatrix}
2 & -4\\
1 & -2
\end{bmatrix},
\]

\[
P_{:,:,2}
=
\begin{bmatrix}
-0.5 & -1\\
1.5 & 3
\end{bmatrix}.
\]

Therefore

\[
\mathsf S(P)
=
\begin{bmatrix}
1.5 & -5\\
2.5 & 1
\end{bmatrix},
\]

and

\[
J_1
=
D+\mathsf S(P)
=
\begin{bmatrix}
1.75 & -5.5\\
3.5 & 1.75
\end{bmatrix}.
\]

Also,

\[
J_2=C,
\qquad
H=P.
\]

## Exact finite-radius recovery test

Use

\[
U=I_2,
\qquad
V=I_2,
\]

and any positive radii, for example

\[
r_1=r_2=t_1=t_2=0.37.
\]

Because (10.1) is quadratic:

- central first differences recover \(J_1,J_2\) exactly;
- the four-point contrast recovers \(H\) exactly;
- curvature correction recovers \(P\) exactly;
- subtraction recovers \(D\) exactly.

Required float64 tolerance:

```text
max_abs(J1_hat - J1) < 1e-12
max_abs(J2_hat - J2) < 1e-12
max_abs(H_hat  - H ) < 1e-12
max_abs(P_hat  - P ) < 1e-12
max_abs(D_hat  - D ) < 1e-12
```

## Non-coordinate full-rank design test

Use an overdetermined fixed design such as

\[
U=
\begin{bmatrix}
1&0\\
0&1\\
1/\sqrt2&1/\sqrt2\\
1/\sqrt2&-1/\sqrt2
\end{bmatrix},
\]

and the same form for \(V\).

The recovered tensors must equal the analytic ground truth to \(10^{-12}\). This verifies orientation and least-squares equations rather than only coordinate probing.

## Active-edge inverse test

For gate \(j\), compute

\[
\widehat c_j
=
\widehat G_{:,j},
\]

because \(\psi'(0)=1\), and

\[
\widehat A_{j:}
=
\frac{
\widehat G_{:,j}^\top
\widehat P_{:,:,j}
}{
\|\widehat G_{:,j}\|^2
}.
\]

Require

```text
max_abs(C_hat - C) < 1e-12
max_abs(A_hat - A) < 1e-12
```

## Structural response-energy identity test

Enumerate the finite centrally symmetric direction laws

\[
u\in
\left\{
\frac{(\pm1,\pm1)}{\sqrt2}
\right\},
\]

\[
v\in
\left\{
\frac{(\pm1,\pm1)}{\sqrt2}
\right\}.
\]

Then

\[
M_1=M_2=\frac12I_2.
\]

For a chosen second system, enumerate all \(16\) product pairs and verify

\[
\mathcal E_Q
=
\frac12\|\Delta J_1\|_F^2
+
\frac12\|\Delta J_2\|_F^2
+
\frac14\|\Delta P\|_F^2
\]

to \(10^{-12}\).

Also verify both sides of (4.7) numerically.

## First-order cancellation test

Implement the scalar cancellation example:

\[
\psi(t)=t+\frac12t^2,
\]

\[
c=[1,1],
\]

\[
A_\star=[1,-1]^\top,
\qquad
A_p=[0,0]^\top,
\]

\[
D_\star=D_p=0.
\]

Required assertions:

```text
zero_order_gap == 0
max_abs(J1_star - J1_patch) < 1e-12
max_abs(J2_star - J2_patch) < 1e-12
first_order_energy < 1e-24
mixed_energy > 0
P_star == [1, -1]
P_patch == [0, 0]
```

## Rank-deficient first-block test

Use

\[
U=
\begin{bmatrix}
1&0\\
1&0
\end{bmatrix}.
\]

Create a direct-gain difference only in the second coordinate:

\[
\Delta D=
\begin{bmatrix}
0&1
\end{bmatrix}.
\]

All measured first responses are zero. The implementation must:

- report rank \(1<2\);
- refuse structural recovery;
- never return a ridge-based certificate.

## Paired-Kronecker incompleteness test

Use only

\[
(e_1,e_1),
\qquad
(e_2,e_2).
\]

Require:

```text
rank(U_marginal) == 2
rank(V_marginal) == 2
rank(W_kronecker) == 2
required_rank_W == 4
structural_identification_status == FAIL
```

Use a tensor with only an off-diagonal entry to verify that all measured mixed responses vanish.

## Unknown-ratio counterexample test

Implement the two scalar systems:

```text
System A:
    psi(t) = t + 0.5 * t^2
    A = 1
    C = 1
    D = -1

System B:
    psi(t) = t + 0.25 * t^2
    A = 2
    C = 1
    D = -2
```

Require:

```text
J1_A == J1_B == 0
J2_A == J2_B == 1
H_A  == H_B  == 1
P_A  == 1
P_B  == 2
```

Any implementation that infers equal path gains from equal raw \(H\) without using the activation ratio must fail this test.

## Zero-curvature test

Use

\[
\psi(t)=t.
\]

Require:

- curvature diagnostic equals zero;
- structural inverse status is `FAIL_CURVATURE`;
- no \(P\) estimate is issued;
- the code demonstrates identical \((J_1,J_2,H)\) for two different \(A\) values after setting \(D=-A\).

## Omitted-bypass test

Implement:

```text
System A:
    path(u, v)   =  u * v
    bypass(u, v) = -u * v

System B:
    path(u, v)   = 0
    bypass(u, v) = 0
```

Require:

```text
total_output_A == total_output_B for all tested points
all_total_response_tensors_match == True
declared_path_effects_match == False
complete_cut_diagnostic == FAIL
```

This test must be documented as an impossibility check, not as a recoverable case.

## Exact finite-radius scaling test

Use

\[
\psi_{\gamma,\eta}(t)
=
t+\frac12t^2+\frac{\gamma}{6}t^3+\frac{\eta}{24}t^4.
\]

For scalar preactivation

\[
q=a^\top u+v,
\]

the central first error is exactly

\[
\widehat J_1^{(r)}[u]-J_1[u]
=
\frac{\gamma r^2}{6}(a^\top u)^3.
\tag{10.2}
\]

The mixed four-point error is exactly

\[
\widehat H^{(r,t)}[u,v]-H[u,v]
=
\frac{\eta}{6}
\left[
r^2(a^\top u)^3v
+
t^2(a^\top u)v^3
\right].
\tag{10.3}
\]

The implementation must verify (10.2) and (10.3) to \(10^{-12}\) for multiple radii.

When both radii are halved and the corresponding coefficient is nonzero, the error magnitude must divide by \(4\).

## Output-dimension accounting test

Replicate a scalar output \(k\) times.

The Frobenius response energy must multiply by \(k\):

\[
\mathcal E_k=k\mathcal E_1.
\]

A per-output MSE may remain unchanged, but it must be explicitly labeled as

\[
\mathcal E_k/k.
\]

This detects silent averaging over the output axis.

## Without-replacement concentration test

Create a finite pool with small \(N\), such as

\[
N=8,
\qquad
m=4.
\]

Enumerate every subset of size \(m\). Compute the exact violation frequency of the interval in (4.47) for several \(\delta\) values. The empirical violation probability must not exceed the bound, up to exact floating-point comparison.

This validates the implementation of \(\rho_m\), the two-sided \(\log(2/\delta)\) factor, and finite-population centering.

## Required test suite names

```text
test_quadratic_asg_exact_recovery
test_noncoordinate_full_rank_recovery
test_active_edge_inverse
test_product_energy_identity
test_first_order_cancellation_requires_mixed
test_rank_deficient_first_block_fails
test_marginal_span_not_kronecker_complete
test_unknown_curvature_ratio_nonidentification
test_zero_curvature_fails
test_omitted_bypass_impossibility
test_central_first_r2_scaling
test_mixed_four_point_r2_t2_scaling
test_output_dimension_energy_scaling
test_without_replacement_serfling_interval
test_corner_sign_order
test_variable_radius_denominator
test_center_replay
test_no_ridge_fallback
```

All deterministic analytic tests must pass before any model execution. They require no GPU.

# Assumption-to-Test Matrix

| Assumption or condition | Mathematical role | Direct diagnostic | Falsification condition | Current Greater-Than status |
|---|---|---|---|---|
| Common fixed basis | Prevents hidden alignment from changing the identified object | Hash and compare \(B_1,B_2\) across systems and cells | Any system-specific learned alignment | Not yet implemented |
| All \(M_1\) nodes precede all \(M_2\) nodes | Supplies a two-block acyclic graph | Static architecture/hook-order audit | Any \(M_1\) node downstream of an \(M_2\) node | **Fails:** head \(9.1\) follows MLP \(8\) |
| \(M_2\) intervention is at gate input | Makes \(z_j\) an independently intervenable scalar gate coordinate | Hook-name and tensor-location audit | Hook is residual output, post-projection activation, or arbitrary subspace | **Fails:** proposed variables are MLP outputs |
| Known activation \(\psi_j\) | Defines edge derivative and curvature correction | Code-level activation identity | Unknown or learned black-box transformation after intervention | **Fails at raw MLP outputs** |
| Known anchor \(a_j\) | Makes \(\rho_j\) observable | Log preactivation before intervention | Anchor cannot be read at the hook | **Fails at raw MLP outputs** |
| Nonzero curvature | Makes \(H\mapsto P\) invertible | Compute \(\min_j|\psi_j''(a_j)|\) | Any declared gate has zero curvature | Not defined for proposed hooks |
| Bounded curvature ratio | Controls error amplification | Report \(\rho_{\max}\) | Nonfinite or protocol-excessive \(\rho_{\max}\) | Not defined for proposed hooks |
| Separable elementwise gates | Ensures one mixed slice per gate | Rank-one residual \(R_j^{\mathrm{fact}}\) | Large factorization residual | Unproved and structurally implausible for raw outputs |
| Linear gate-output readout | Prevents downstream nonlinear mixed terms | Architecture decomposition or immediate linear-output hook | Nonlinear downstream map jointly depends on both blocks | **Fails for MLPs 8–11 to final logits** |
| Complete mixed cut | Assigns all mixed dependence to the declared path | Edge-isolated decomposition and omitted-bypass audit | Residual mixed term not assigned to measured paths | **Unproved; circuit contains multiple bypasses** |
| No arbitrary \(M_2\) rotation | Preserves coordinatewise \(\rho_j\) correction | Verify coordinate selector/signed permutation | PCA or dense learned gate rotation | Would fail if PCA used |
| Full \(M_1\) design rank | Identifies \(J_1\) and direct gain | SVD of \(U\) | \(\operatorname{rank}(U)<r_1\) | Not yet established |
| Full \(M_2\) design rank | Identifies \(J_2\) | SVD of \(V\) | \(\operatorname{rank}(V)<r_2\) | Not yet established |
| Full Kronecker rank | Identifies the mixed tensor | SVD of \(W\) or Cartesian rank identity | \(\operatorname{rank}(W)<r_1r_2\) | Raw dimensions make completeness infeasible |
| Positive population coverage | Gives lower response-energy bound | Lower confidence bound on \(\lambda_{\min}(M_i)\) | Lower bound nonpositive | Not yet established |
| Product probe law | Factorizes mixed energy into \(M_1,M_2\) moments | Independent block donor IDs and sampler audit | Coupled or adaptively chosen donor pairs | Implementable, not established |
| Disjoint donor/basis split | Prevents adaptive probe design | Dataset-ID intersection checks | Any overlap with confirmatory cells | Implementable, not established |
| Paired systems and corners | Removes avoidable Monte Carlo variation | Exact probe/corner IDs | Different directions or radii across systems | Must be newly implemented |
| \(C^4\) local smoothness | Justifies \(O(r^2+t^2)\) finite-radius error | Full/half/quarter-radius diagnostics | Instability or nonconvergent radius pattern | Testable but not yet run |
| Vector-output preservation | Prevents lost \(k\)-factors and permits factorization tests | Shape and energy-scaling tests | Output axis averaged before fitting | Current comparison averages output-axis squares |
| Active gate gain for edge recovery | Permits \(A_j,c_j\) inverse | Lower bound on \(\|G_{:,j}\|\) | Output-null or near-null gate | Unknown |
| Absolute path target | Avoids ill-conditioned recovery ratios | Unit and formula audit | Division by clean-corrupt near-zero gap | Must be enforced |
| Independent tensor-fit and energy probes | Makes energy concentration conditional and valid | Probe-set intersection check | Same probes reused and treated as fresh | Not present |
| Bounded or certified sub-Gaussian probes | Supplies finite-probe intervals | Norm bounds or certified \(K,\sigma_X\) | Unbounded probes with Hoeffding interval | Must be specified |
| Frozen finite donor population for Serfling | Defines without-replacement target | Pool hash, \(N,m\), sampler audit | Pool changes after outcomes | Implementable |
| Cell-level statistical unit | Avoids pseudo-replication | Aggregation and bootstrap audit | Heads, layers, prompts, or probes treated as independent cells | Must be enforced |
| Locality of the scientific claim | Prevents global-algorithm overclaim | Claim-text and radius-domain audit | “Global mechanism” or “algorithm identity” language | Must remain restricted |
| Uniqueness of the path decomposition | Needed for circuit identity beyond ASG | Cannot be inferred from response fit alone | Alternative path/bypass allocation exists | **Not established** |

The directly testable rows can falsify the theorem's applicability. Passing them does not prove the modeling assumptions that the cut is semantically complete or globally unique.

# Binding Go/No-Go Checklist

## Formal mathematics

| Gate | Status |
|---|---|
| Structural object defined independently of response derivatives | **PASS** |
| Explicit acyclic multi-mediator class | **PASS** |
| Exact derivative-to-structure map | **PASS** |
| Explicit structural inverse | **PASS** |
| No assumed unspecified \(\kappa\) | **PASS** |
| Derived inverse constant | **PASS:** \((2d_2+1)^{-1/2}\) |
| Product-probe lower and upper energy bounds | **PASS** |
| Vector-valued output factors retained | **PASS** |
| Central first-order finite-radius bound | **PASS** |
| Four-point mixed finite-radius bound | **PASS** |
| Deterministic finite-design tensor bounds | **PASS** |
| Bounded i.i.d. probe concentration | **PASS** |
| Sub-Gaussian finite-variance/MOM result | **PASS** |
| Without-replacement finite-population bound | **PASS** |
| Strong if-and-only-if probe-completeness theorem | **PASS** |
| Constructive cancellation converse | **PASS** |
| Constructive rank-deficiency converse | **PASS** |
| Constructive reparameterization converse | **PASS** |
| Constructive omitted-bypass impossibility | **PASS** |
| Local path-specific causal target and approximation bound | **PASS** |

## Implementation readiness for the restricted class

| Gate | Status |
|---|---|
| Exact array shapes | **PASS** |
| Exact central and mixed equations | **PASS** |
| Exact least-squares equations | **PASS** |
| Curvature correction | **PASS** |
| Probe-rank diagnostics | **PASS** |
| Radius diagnostics | **PASS** |
| Absolute target definition | **PASS** |
| Cell-level uncertainty contract | **PASS** |
| Analytic synthetic ground truth | **PASS** |
| Invariant test suite | **PASS** |
| Existing repository code already satisfies contract | **FAIL** |
| CPU implementation may proceed | **YES** |

## Applicability to the proposed Greater-Than experiment

| Gate | Status |
|---|---|
| Proposed \(M_1\) wholly upstream of proposed \(M_2\) | **FAIL** |
| Proposed \(M_2\) consists of known gate-input coordinates | **FAIL** |
| Known \(\psi'/\psi''\) ratio at proposed \(M_2\) hook | **FAIL** |
| Parallel separable-gate structure | **FAIL** |
| Complete mixed cut with no omitted bypass | **UNPROVED AND GENERICALLY FALSE** |
| Raw-space probe completeness | **FAIL BY DIMENSION** |
| Invertible map from total mixed Hessian to desired path | **ABSENT** |
| Independent absolute path target available under the same graph | **NOT YET IDENTIFIED** |
| Proposed GPU execution authorized | **NO** |

## Poster-level claim that remains valid

The strongest currently supportable claim is:

> For a predeclared fixed-basis separable-gate residual DAG with known gate activations, observed anchors, nonzero activation curvature, a complete mixed cut, and probe-complete product interventions, central first-order and four-point mixed-second-order responses identify a reduced local path-gain equivalence class with an explicit structural inverse and finite-radius and finite-probe error bounds. When probe covariance, gate curvature, fixed-basis semantics, or complete-cut assumptions fail, constructive counterexamples show that the path parameter is not identified.

Outside that class, the measurements identify only:

\[
(J_1,J_2,H)
\]

on the covered probe subspaces. They may be useful local functional diagnostics, but they are not a certificate of path, circuit, weight, reparameterization, or global algorithm identity.

## Prohibited claims

The execution or writing agent must not claim that:

- arbitrary transformer mixed Hessians are structural path tensors;
- raw MLP-output interventions satisfy the gate theorem;
- multiple donor directions imply completeness;
- effective rank replaces a positive minimum eigenvalue;
- ridge regression repairs rank deficiency;
- local response equality proves full circuit identity;
- attention probability or activation cosine is structural ground truth;
- a near-zero-denominator recovery ratio is a well-conditioned path target;
- conformal calibration repairs structural nonidentifiability;
- the current proposed Greater-Than \(M_1/M_2\) grouping is licensed by this theorem;
- additional GPU samples can resolve the absent structural inverse.

## Final binding decision

**THEORY AMBER:** A non-tautological theorem is fully established for the ASG-RDAG reduced local path-gain equivalence class, but the proposed GPT-2-small Greater-Than intervention blocks do not instantiate that class. The poster-level restricted-class identification and impossibility result is valid. The unresolved Greater-Than structural-inverse lemma prevents GREEN. The execution agent must implement and pass the CPU analytic suite, but must not run the proposed Greater-Than GPU program or claim raw block-output path identification.
