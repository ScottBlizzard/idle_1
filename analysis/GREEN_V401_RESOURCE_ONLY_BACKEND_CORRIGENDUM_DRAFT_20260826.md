# GREEN v4.0.1 Resource-Only Backend Corrigendum — DRAFT, NOT EFFECTIVE

## Status

This document is an outcome-blind draft for project-owner review. It has no authority until the project owner explicitly approves it. No real-row certificate, development, confirmation, endpoint, derivative, P13, or q-selection action is authorized by this draft.

## Reason

The v4.0.0 formal package used a hand-written estimate of 75,000,000 scalar MPFR operations per row under a frozen cap of 100,000,000. A shape-derived audit of the actual GPT-2-small block-10-to-contrast four-branch tail gives approximately:

- 7,112,448 dense coefficient terms per branch and cell at sequence length 12;
- 28,449,792 terms per four-branch cell;
- at least 341,397,504 directed low-level MPFR operations per cell under the conservative reference accounting;
- at least 682,795,008 operations for the mandatory two-cell initial partition, before multiple radii or refinement.

Thus the old estimate is invalid and the frozen 100-million cap makes the current architecture statically infeasible. This is a resource-plan defect, not evidence against the Joint Witness estimand or theorem.

## Proposed supersession

If approved, this corrigendum supersedes only the v4.0.0 operation-cap and backend-performance clauses. It does not supersede any scientific or numerical-semantic clause.

The following remain unchanged:

1. the official scalar is `PAT_J - PAT_B - TAR_J + TAR_B`;
2. J keeps selected-gate posts live, B freezes them to the same anchor, and both keep the residual bypass;
3. `Psi'(0)=theta` remains the composition identity;
4. official interval-jet, endpoint, signed-curvature, and intersection mathematics;
5. 384-bit official and 512-bit audit precision;
6. exact dyadic radii, partition order, width tolerances, max depth, and max cells;
7. outcome, development, confirmation, AD, P13, and q-selection firewalls;
8. 24-hour projected execution and 64-GiB-per-worker hard limits;
9. every resource overflow remains `RESOURCE_INCONCLUSIVE`, never success.

The following outcome-blind changes are proposed:

1. replace the invalid hand-written operation estimate with counts derived from the replayable TensorProgram:
   - semantic coefficient terms;
   - directed MPFR primitive calls;
   - tensor kernel calls;
2. implement a compiled streaming MPFR endpoint backend that preserves exactly:
   - MPFR RNDD/RNDU endpoints;
   - the canonical scalar expansion;
   - the fixed pairwise/FMA policy;
   - partition order;
   - serialized canonical outputs;
3. require bit-identical agreement with the Python reference on the full synthetic theorem suite at both precisions;
4. determine a replacement scalar-operation hard cap only from:
   - the final replayable TensorProgram shapes;
   - tail-shaped synthetic benchmarks containing no scientific row outcome;
   - the unchanged wall-time and memory limits;
5. preserve the scalar operation-count meaning; a tensor kernel may not be counted as one scalar operation to evade the old cap;
6. keep Arb/ball arithmetic audit-only, not official;
7. exclude new divided-difference/Taylor-model enclosures from this resource-only amendment.

## Required evidence before this draft can become an execution gate

- exact program generation and shape-derived counters;
- compiled/reference bit-identity report at 384 and 512 bits;
- synthetic throughput and peak-memory report;
- proposed numeric cap derived from those reports rather than chosen ad hoc;
- independent numerical/backend audit;
- independent protocol/outcome-firewall audit;
- explicit project-owner approval of the final numeric cap and this scope.

## Project-owner approval phrase

The following phrase, if stated explicitly by the project owner after reviewing the final benchmark-derived cap, would activate the final version (not this draft):

> I approve the outcome-blind GREEN v4.0.1 Resource-Only Backend Corrigendum: preserve the scientific estimand and all certificate semantics, authorize a bit-equivalent compiled MPFR tensor backend, and replace the invalid 100-million scalar-operation cap only with the benchmark-derived hard cap documented in the final audited corrigendum.

