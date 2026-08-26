# GREEN v4.0.0 Execution Consistency Lock — 2026-08-26

## Authority and purpose

This lock records the conservative executable interpretation authorized by the
user after reviewing:

- `GPTPRO_GREEN_V300_TERMINAL_SCIENTIFIC_DECISION_20260826.md`;
- `GPTPRO_GREEN_V400_BINDING_CORRIGENDUM_20260826.md`.

It does not alter GREEN v3, inspect a sealed result, lower a threshold, or
authorize v4 development/confirmation.  Where the two GPTPro documents overlap,
the later specific corrigendum controls.  Where the corrigendum explicitly
preserves an earlier frozen design, that earlier design controls.  Every
remaining ambiguity is resolved toward the least outcome access.

## Binding phase scope

The current phase is a **static/theorem formal prepare**.

Allowed for real candidate/donor rows:

1. deterministic universe construction and exclusion by identifiers/hashes;
2. tokenization and shape checks;
3. frozen-model loading in evaluation mode;
4. capture and bit hashing of hook base/direction tensors needed to define a
   future certificate plan;
5. static dependency/causal-cone extraction;
6. reduced/unreduced graph construction and singleton semantic parity;
7. LayerNorm/softmax domain-margin and resource estimates;
8. creation of a certificate plan with `execution_authorized=false`.

Forbidden in this phase for any real v4 row:

- evaluation of a real Joint Witness interval or P13 result;
- a real AD target/adjudicator result;
- recovery error, endpoint-effect sign, set-SNR, certificate status, or
  Boundary Transition outcome;
- development or confirmation execution.

Consequently, the predecessor donor-result gates that require real Witness, AD,
P13 contraction, or observed boundary coverage are deferred, not silently
passed.  The only successful terminal status in the current phase is
`PREPARE_PASS_STATIC_THEOREM_ONLY`, followed by
`STOP_AFTER_FORMAL_PREPARE_RETURN_TO_GPTPRO`.  Any failed preflight emits
`PREPARE_STOP_<FIRST_FAILED_GATE>`.

## Boundary Transition precedence

The first v4 decision's outcome-blind measurement-channel experiment remains
binding:

- attenuation levels `alpha = 2^-j`, `j = 0..16`;
- subtractively dithered quantization;
- dyadic `q` candidates `2^-52, 2^-50, ..., 2^-12`;
- four deterministic dither replicates;
- effective-SNR transition/plateau estimands and their frozen development and
  confirmation gates.

The later `h_geom` amplitude ladder and logistic P13-success model are not used.
They conflict with an already frozen design and are excluded from the v4 lock.
Formal prepare may freeze the attenuation/quantization code and schemas, but it
may not select `q` from real response magnitudes or compute transition outcomes
in the current static phase.  Those actions require a later explicit authority.

## Numerical and resource constants

These constants are fixed before any v4 scientific outcome exists:

- local proof precision: 256 bits;
- official proof precision: 384 bits;
- independent audit precision: 512 bits;
- interval absolute stopping tolerance: `2^-80`;
- interval relative stopping tolerance: `2^-40`;
- maximum dyadic subdivision depth: 24;
- maximum cells per row: 262144;
- reduced/unreduced MPFR singleton requirement: interval overlap plus absolute
  midpoint difference at most `2^-180` at 256-bit preflight precision;
- TransformerLens parity tolerance: `64 * eps(dtype) * max(1, op_count) *
  max(1, abs(reference))`, evaluated from a frozen operation count; it is an
  invalidity diagnostic and never a scientific error radius;
- maximum static causal-cone nodes: 2,000,000;
- maximum projected scalar MPFR operations per row: 100,000,000;
- maximum projected proof memory per worker: 64 GiB;
- maximum formal-prepare CPU wall time: 24 hours;
- bootstrap replicates: 100000, seed 20260805;
- permutation replicates: 100000, seed 20260805;
- no donor replacement after any response/certificate quantity is computed;
  before such computation, an engineering-invalid noun is replaced only by the
  next reserve noun in the frozen SHA-256 order.

The discarded `h_geom` design means no `eta_scale` exists in this protocol.

## Candidate universe

`analysis/GREEN_V400_CANDIDATE_NOUNS_20260826.txt` is required.  It is an
immutable literal vocabulary selected before v4 model outputs are observed.
Eligibility is tested exactly as specified by GPTPro.  All nouns found in any
v1/v2/v3 donor, development, or confirmation namespace are excluded by hash
without printing sealed row content.  Eligible nouns are ranked by the frozen
v4 split salt; ranks 1–4 are static formal donors, 5–16 future development,
17–32 future confirmation, and the remainder reserve.  In the current phase
future split identifiers may be hashed and sealed, but their model responses
may not be evaluated.

## Normative Joint Witness chain

For every future authorized real row and each frozen radius, certified endpoint
and signed curvature enclosures define

`I_W(h) = w(h) - K_sec(h)/(2h)`,

with signs chosen from the exact central-secant identity so that `I_W(h)`
contains `Psi'(0)`.  The multi-radius witness is the intersection of all valid
`I_W(h)` intervals.  Empty intersections are invalid.  If
`I_W=[l_W,u_W]`, then

`B_W = max(abs(theta_hat-l_W), abs(theta_hat-u_W))`,

`B_JW = min(B_box, B_W)`,

`S_r = abs(theta_hat)/B_JW`,

and the cell statistic is the absolute mean response-only center divided by the
mean `B_JW`.  The scientific set-SNR threshold remains 4.  An interval touching
or crossing a gate boundary is unresolved.  Resource caps produce
`RESOURCE_INCONCLUSIVE`, never success.  P13 retains its predecessor numerical
definition but is not evaluated on real rows in the present phase.

## Repository and provenance

The v4 branch is created non-destructively in a new worktree from parent
`48182844a43d391439704f27aa26d513d33adaa0`.  No reset, clean, deletion, or
modification of a v3 file is permitted.  Missing GPTPro appendices are not
treated as evidence.  Formal prepare creates its own machine-derived repository
manifest, immutable closure, and literature/source ledger.

## Current authorization

- v4 implementation: authorized;
- synthetic theorem fixtures: authorized;
- static real-row donor planning described above: authorized only after all
  theorem tests pass;
- real-row Witness/AD/P13 calculation: unauthorized;
- development: unauthorized;
- confirmation: unauthorized;
- v3 confirmation: permanently sealed.
