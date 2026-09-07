# Current state for independent strategic review

Date: 2026-09-07. Scope: review package, not a scientific result or launch authorization.

## User objective and constraints

The author has 20 days total and wants the final five days protected for manuscript
polishing. The project already has considerable breadth. The requested improvement
is theoretical and explanatory depth, a decisive incremental contribution, and a
complete paper. Additional model/task sweeps are not the default answer.
Oral is the ambition; do not turn it into a promised outcome.

The project originally sought to challenge/extend the account associated with
Grant et al., *Addressing Divergent Representations from Causal Interventions*,
described in the earlier local audit as an ICLR 2026 Oral. Verify that attribution
from primary sources. This is a scientific point of departure, not grounds to
dismiss overlapping literature.

## Evidence that already exists

| Evidence | Supported observation | Limitation |
|---|---|---|
| v4 development IOI | 1,152 total rows; 1,144 high-restoration rows, of which 899 fail the defined held-out transport criterion (78.58%) | Development diagnostic; not GREEN superiority or a universal mechanism-error rate |
| v4 development Greater-Than | 2,592 total rows; 1,767 high-restoration rows, of which 164 fail (9.28%) | Original diagnostic stratum; clean-task-validity inputs were missing from the primary route |
| Strong public-panel baselines | AUROC about 0.852 IOI and 0.961 GT; first-order, finite patching and MS-HVP scores nearly coincide at the frozen scale | These AUROCs do not substitute for the primary matched-coverage comparison |
| v3 corrected development | Accurate joint point estimates, but all 80 joint intervals cross zero | Accurate estimation did not imply useful set identification; inspect the corrected report |
| Historical support/IRS controls | Support labels changed with reference; response agreement failed as a sufficient structural-path criterion | Useful falsification evidence, not proof that GREEN solves the remaining problem |
| Repaired-backend regression | Existing report records 92 tests, zero failures/errors | Engineering evidence only; does not prove all real-case certificates or scientific utility |

The August 30 development primary analysis was blocked because site-level GREEN
statuses were absent, and GT clean-task-validity inputs were missing. The completed
baseline/endpoint run must not be marketed as a completed GREEN comparison.
See the cited development completeness decision for exact definitions and counts.

## Present repair and live computation

The old native backend could export a nonfinite value as an apparent exact zero.
A minimal attention case whose exact output is one exposed this bug. There was
also a large-integer JSON serialization limit. The numerical repair and explicit
nonfinite rejection are included in this source snapshot.

Both tasks retain 108 captures and 864 materialized direction graphs from the old
formal preparation, according to the repair handoff. The old formal workers did
not publish completed certificates. No formal P13 PASS or successor confirmation
result is asserted by this package.

First-radius diagnostic continuations reportedly passed for the two fixed repair
cases before the full acceptance jobs were launched. The raw diagnostic artifacts
remain on the server, not in this repository. Treat this as executor-reported
smoke evidence, not independently downloaded full acceptance evidence.

Latest read-only process snapshot:
```text
2026-09-07T14:33:08+08:00
    PID     ELAPSED STAT %CPU   RSS
1921903    01:15:16 Rl    102 6376680
1921987    01:15:03 Rl    102 6156096
NODE_PROGRESS full-L4 1921903 272 b85c2c207c70a4ad13c67a12e4e47c9818a37ef2f5b27e45aa803f19233d65aa pairwise_affine.v1
NODE_PROGRESS full-L4 1921903 0 70678c85d030146e880b9c300cee8047626f6652411c53588df82ca86b8df667 affine_scatter.v1
NODE_PROGRESS full-L4 1921987 246 cff84f2a81adfb4f423fb7d9f512f44c8ce109654711ed455e3cca6ebb720a29 pairwise_affine.v1
NODE_PROGRESS full-L4 1921987 258 e0141f5ab017cf6f7997896f1f2b1d6a19cf37b369954d075450361f520982ef pairwise_affine.v1
acceptance.json.started.json 372 bytes
acceptance.json.started.json 362 bytes
```

Only start markers were present: no final acceptance or failure JSON was observed
at that time. These are two engineering acceptance cases at fixed L4, 384/512-bit
precision, all three radii, with independent captured-AD overlap checks.

This snapshot does not update itself. Future output must be inspected before a
stronger status is claimed. The source is a local candidate snapshot, not a newly
adopted formal runtime. Old queues still bind the earlier runtime.

## Arithmetic workload and deadline risk

The complete certificate loop is in `src/green_v410_fixed_budget_certificate.py`.
One graph evaluation returns all five roots together. At L4, each radius calls:
six official interval evaluations (two initial cells plus two bisections), four
audit final-cell evaluations, and three point evaluations at each of two precisions.
Across three radii, this is 48 calls. Static caching affects cost, not this call count.

P13: 864 directions per task, 1,728 total.
Confirmation: 9,216 IOI directions plus up to 27,648 GT directions.
Total maximum: 38,592 directions, excluding auxiliary queues and orchestration.

Illustrative ideal lower-bound workload at 64 simultaneous workers:
- average 15 min/direction: 6.28 days;
- average 30 min/direction: 12.56 days;
- average 1 h/direction: 25.13 days;
- average 4 h/direction: 100.50 days.

A ten-day compute allocation requires average service time below 23.9 minutes
per direction at ideal utilization; the real requirement is stricter because of
phase dependencies and overhead. These are arithmetic scenarios, not measured
whole-fleet forecasts. The two currently running cases are not representative of
every layer, and repaired-runtime timings must replace old invalid-runtime timings.

The existing master plan also lists incomplete confirmation coordination modules.
Do not assume only GPU runtime remains. MPFR work is CPU-bound; empty GPUs do not
automatically make it faster. Hardware belongs to a shared school server; system
changes and other users' resources are out of scope.

## Highest-priority questions for Pro

1. **Estimand-to-endpoint gap.** The certificate bounds a selected gated-path
   contrast. The endpoint measures complete finite responses on hidden directions.
   At matching inputs, full response discrepancy decomposes into gated discrepancy
   plus bypass discrepancy. Small gated discrepancy does not alone bound their sum.
   Public-direction derivatives also do not automatically control unseen directions
   or finite-radius responses. What precise conditions, counterexamples, and useful
   propositions follow? Distinguish a deterministic implication from an empirical
   hypothesis. Do not define the endpoint to agree with the certificate.
2. **Implementation of the claimed advantage.** Does the actual five-root program
   preserve a relational dependence/cancellation that branchwise intervals lose?
   Inspect graph building, retained roots, final combination and worker record
   construction. The manuscript and some planning descriptions of shared graphs
   versus branch-isolated graphs need a precise scope audit. A function named
   Joint Witness is not evidence that its claimed mechanism exists.
3. **Beyond point estimates.** Given strong finite/AD/HVP baselines, is the benefit
   a better ranking, certified numerical assurance, a useful abstention rule, or
   some combination? What minimum fair comparison separates these explanations?
4. **Deadline feasibility.** What is the smallest scientifically complete evidence
   package that can support the strongest justified claim in 20 days? Separate
   unchanged execution of the frozen attempt from any genuinely new design.
5. **Theory depth.** Elementary non-identifiability, triangle-box sharpness and DAG
   soundness are supporting arguments. Identify a substantive additional result
   worth trying, or explain precisely why it cannot deliver the desired advance.

## Known overstatements and unresolved scope mismatches

- Earlier conversational updates incorrectly counted graph traversal restarts as
  completion of successive outputs and repeatedly predicted near completion.
  Withdraw those interpretations. Five outputs are produced on every traversal.
- The subsequent 10–14-hour remaining estimate was heuristic, not calibrated.
  Do not use it as a reliable full-project deadline.
- `iclr2027/paper.tex` still describes synthetic calibration as underway, and its
  abstract overstates completion of method/analysis relative to the repair status.
- The manuscript describes a physically separate higher-precision replay. The
  current full-acceptance function visibly calls official and audit computations
  within one process. Whether a separate formal replay route meets the intended
  scientific guarantee must be checked; do not transfer that claim from prose.
- Similar care is needed for claims of nodewise precision nesting, shared
  cancellation, and branch-independent reference bounds: check what the actual
  functions validate, not merely what earlier plans promised.
- The old label `POSTER_ONLY` in v3 is a historical assessment, not acceptance at
  any venue. No prior local or AI score is authoritative evidence of paper quality.
- The 20-day roadmap is proposed, not validated. Pro is asked to challenge it.

No source code or scientific outputs were altered to hide these issues when
preparing the review. The manuscript is supplied as-is for diagnosis, with this
status correction alongside it.

## Data-access boundary

Historical development and corrected v3 evidence are reviewable. Untouched
confirmation endpoint outcomes, keys, large activation stores and model checkpoints
are not included. Missing server artifacts must be reported as unavailable rather
than guessed. Pro should not run jobs or unlock hidden data.

Ask for a concrete theoretical and strategic judgment now; the execution agent
can later supply the two acceptance results and improved timing estimates.
