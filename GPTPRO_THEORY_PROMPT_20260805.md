# Prompt to submit to GPT Pro: formal theory gate

You are the lead theoretical researcher for an intended ICLR oral-level paper.
Work from the complete repository at:

**https://github.com/ScottBlizzard/idle_1** (branch: `main`)

Start with `GPTPRO_START_HERE.md`. Read `GPTPRO_0805.md` in full before doing
anything else, then read `analysis/IRS_THEORY_P0.md` and inspect the relevant
implementation and tests. The first review has already selected the research
direction. This is **not** another brainstorming or reviewer-verdict task. Your
job is to resolve the load-bearing formal theory gate so that an execution agent
can implement it exactly.

## Scientific objective

The proposed oral-level result is that, for a non-vacuous and
transformer-relevant restricted multi-mediator class, a probe-complete family of
central first-order and mixed-second-order local interventional responses
identifies a precisely defined fixed-basis structural or path parameter up to
finite-radius and sampling error, while rank-deficient probes or a non-injective
structural map admit explicit non-identifiability counterexamples.

The prior review correctly warns that a theorem of the form "assume the desired
structural map is injective with minimum singular value kappa" is tautological.
You must therefore **derive** injectivity and all inverse constants for an
explicit substantive model/graph class. Do not hide the hard step in an
assumption, a definition, or unexplained faithfulness language.

## Required decisions and derivations

### 1. Define the identified object exactly

Define, in a common fixed basis, all of the following as mathematical relations
rather than rhetorical evidence levels:

- zero-order behavioral equivalence;
- probe-indexed first-order local equivalence;
- probe-indexed first-plus-mixed-second-order local equivalence;
- the proposed structural/path equivalence.

State whether the identified structural object is an individual path parameter,
a reduced path-gain tensor, a graph-level equivalence class, or something else.
Explicitly distinguish it from full weight identity, circuit identity under
reparameterization, and global algorithm identity.

### 2. Construct a non-tautological positive class

Give one explicit acyclic multi-mediator structural class that is close enough
to independently intervenable transformer blocks to motivate the Greater-Than
experiment. Specify:

- structural equations and graph;
- mediator blocks and their fixed coordinates;
- intervention semantics;
- output map;
- free and known quantities;
- the exact structural/path parameter to be recovered;
- all smoothness, locality, sparsity, rank, sign, anchoring, or faithfulness
  assumptions.

Derive the map from the structural parameters to the observable first- and
mixed-second-order response tensors. Prove its injectivity for this class and
derive an explicit inverse or a computable lower bound. It is not acceptable to
define the structural parameter to be the response tensor itself unless you
also prove why that tensor equals an independently meaningful path quantity in
the stated graph.

If no transformer-relevant class can meet this requirement without vacuous
assumptions, prove or carefully establish that obstruction and issue a theory
NO-GO. Do not manufacture a positive theorem merely to keep the project alive.

### 3. State and prove the main theorem

Provide a publication-grade theorem with exact dimensions, norms, constants,
and quantifiers. It must cover:

- product probe laws on the declared mediator blocks;
- covariance/effective-rank conditions stated on the actual probed subspaces;
- central first-order estimators;
- four-point mixed-second-order estimators;
- finite-radius error under sufficient differentiability;
- vector-valued outputs without dropping output-dimension factors;
- a lower and upper response-energy bound;
- an inverse bound for the declared structural/path discrepancy;
- finite-probe sampling error, including the correct assumptions for bounded,
  sub-Gaussian, or without-replacement probes.

Show every nontrivial proof step. Clearly label which part is exact in the
restricted class and which part is a local approximation for a nonlinear
transformer.

### 4. Prove converses and impossibility boundaries

Give constructive counterexamples for at least:

- rank-deficient probe covariance hiding a structurally relevant direction;
- cancellation between first-order paths that requires a mixed-order probe;
- a non-injective structural factorization or reparameterization producing the
  same complete response tensors;
- failure when the chosen mediator cut omits a bypass variable;
- any condition in the positive theorem whose removal destroys identification.

State the strongest if-and-only-if probe-completeness result that is actually
true. Do not claim that multi-directionality alone is completeness.

### 5. Connect the theorem to a measurable path-specific target

Define an absolute, well-conditioned path-specific causal effect for the
declared mediator graph. Prove, or explicitly delimit, its relation to the
identified response object. Explain exactly why attention probability,
activation cosine, normalized recovery using a near-zero denominator, and final
behavioral restoration are not valid substitutes.

Then map the theory to the proposed GPT-2-small Greater-Than experiment:

- block M1: outputs of heads 5.1, 5.5, 6.9, 7.10, 8.8, 8.11, and 9.1;
- block M2: outputs of MLPs 8, 9, 10, and 11;
- target: an absolute path-specific effect through M1 -> M2 -> the
  correct-versus-incorrect Greater-Than logit margin;
- probes: disjoint held-out clean donor states under a fixed conditional product
  law.

State which theorem assumptions are directly testable, which are only modeling
assumptions, and what empirical diagnostic would falsify each testable one.

### 6. Deliver an implementation contract

Provide exact estimator equations and language-agnostic pseudocode for:

- paired central first-order responses;
- paired four-point mixed responses;
- probe covariance and covered-subspace diagnostics;
- target-energy/denominator admissibility;
- finite-radius and half-radius checks;
- the structural/path target;
- cell-level aggregation and uncertainty.

Specify array shapes, required data splits, numerical stability conditions,
and invariant unit tests. Include a minimal analytic synthetic model with known
ground truth that an execution agent can implement before using a GPU.

### 7. End with a binding go/no-go decision

Conclude with exactly one of:

- **THEORY GREEN:** a non-tautological positive identification theorem is fully
  established for a defensible class, and implementation may proceed;
- **THEORY AMBER:** only an equivalence-class or impossibility result is valid;
  state precisely what poster-level claim remains;
- **THEORY RED:** the proposed oral line cannot be supported without assuming
  the conclusion, so the execution agent must not run the Greater-Than program.

List any unresolved lemma that prevents GREEN. The execution agent must not be
asked to invent missing mathematics.

## Prohibited shortcuts

Do not:

1. assume an unspecified bi-Lipschitz map or positive kappa;
2. relabel first/mixed derivatives as "structure" by definition;
3. present Taylor expansion or a trace identity as the central novelty;
4. identify individual weights through quantities invariant to hidden-basis
   transformations;
5. claim global mechanism recovery from a local finite-radius result;
6. use conformal calibration to repair structural non-identifiability;
7. recommend more experiments in place of a missing proof;
8. soften a theory NO-GO because the intended venue is an oral.

## Output contract

Return one self-contained Markdown document intended to be saved verbatim as:

`analysis/GPTPRO_THEORY_PACKAGE_20260805.md`

Use exactly these top-level sections:

1. `# Theory Gate Verdict`
2. `# Formal Objects and Equivalence Relations`
3. `# Explicit Multi-Mediator Structural Class`
4. `# Main Identification Theorem`
5. `# Complete Proofs`
6. `# Converses and Counterexamples`
7. `# Path-Specific Target`
8. `# Greater-Than Mapping`
9. `# Implementation Contract`
10. `# Synthetic Verification Specification`
11. `# Assumption-to-Test Matrix`
12. `# Binding Go/No-Go Checklist`

The document must contain the complete derivation, not a proof sketch. Define
every symbol before use. Separate proved statements from conjectures and design
recommendations. Cite direct primary sources only where they are necessary to
establish prior art or the Greater-Than circuit, but keep the main effort on the
mathematics. Do not spread the answer across messages and do not write prose
outside the requested Markdown document.
