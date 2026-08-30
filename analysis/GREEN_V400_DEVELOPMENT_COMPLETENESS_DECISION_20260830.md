# GREEN v4 Development Completeness Decision

**Date:** 2026-08-30  
**Scope:** development outcomes only; confirmation remained sealed  
**Decision:** `STOP_PRIMARY_ANALYSIS_MISSING_PRESPECIFIED_GREEN_INPUTS`

## What completed successfully

The formal development prediction and endpoint fleets completed with exact
plan coverage:

- IOI: 1,152 prediction/endpoint rows; 1,152 endpoint statuses are `VALID`.
- Greater-Than: 2,592 prediction/endpoint rows; 2,592 endpoint statuses are
  `VALID`.
- Packet commitments and model-session bindings validate on every row.
- No confirmation artifact, identifier, queue, or outcome was accessed.

The no-clobber server audit artifacts are:

- `/mnt/sdb/ccj/iclr_1_runs/green_v400_development_completeness_audit_20260830_v5/ioi_audit.json`
  (`sha256=9d54f9b4cb60b392a22dbf289eb1c21f676628817e6a3c3d767531a749cdb2d4`)
- `/mnt/sdb/ccj/iclr_1_runs/green_v400_development_completeness_audit_20260830_v5/gt_audit.json`
  (`sha256=ad39e312ba045d0f235b9549d0c7fab7ee991a7ad2dbe726813f1e1ea4e2dbd8`)

## Scientifically useful development signal

The run establishes a strong silent-failure phenomenon before any GREEN
comparison is attempted.

| Task | Valid high-restoration rows | Fraction of all rows | Held-out transport failures (`error > 0.20`) | Failure fraction in the stratum |
|---|---:|---:|---:|---:|
| IOI | 1,144 / 1,152 | 99.31% | 899 | 78.58% |
| Greater-Than | 1,767 / 2,592 | 68.17% | 164 | 9.28% |

Thus, especially on IOI, behavioral restoration is not enough to establish
transport. This is directly aligned with the paper's motivating claim.

The frozen public-panel baselines are also strong predictors of the held-out
failure label. On the provisional high-restoration strata, AUROC is 0.852 for
finite activation patching/first-order/MS-HVP on IOI and approximately 0.961
on Greater-Than. The empirical four-branch interaction reaches 0.745 and
0.954, respectively. These are diagnostics only, because the primary matched
coverage comparison requires GREEN's accepted count.

MS-HVP is numerically almost identical to finite activation patching at the
frozen direction norm: the maximum absolute normalized-risk difference is
`2.46e-8` on IOI and `1.44e-8` on Greater-Than. First-order differs by at most
`2.95e-5` and `3.62e-5`, respectively. This makes the final GREEN comparison
demanding and prevents an Oral claim based on weak comparators.

## Blocking completeness defect

The frozen analyzer requires one `green_status` per site row and, for
Greater-Than, one `clean_task_valid` value per row. Neither exists in the
completed formal artifacts:

- IOI GREEN certificate statuses present: `0 / 1,152`.
- Greater-Than GREEN certificate statuses present: `0 / 2,592`.
- Greater-Than clean-task-validity records present: `0 / 2,592`.

This is not missing aggregation glue. The activated execution plans contain
prediction, Grant, numerical-replay, and endpoint queues, but no certificate
queue. Their pinned source lists omit the v4 certificate engine. The formal
worker accepts only `prediction`, `grant`, `replay`, and `endpoint`; the
prediction packet schema contains baselines but no GREEN certificate field.

The older Joint Witness implementation cannot be silently substituted:

- `CertificatePlan` rejects `execution_authorized=true`;
- `JointWitnessRowSpec` accepts only `formal_prepare_pool` or `synthetic`;
- certificate serialization is explicitly synthetic-only;
- its 2026-08-26 row universe and graph identities do not match the later IOI
  or Greater-Than site universes.

Inferring GREEN status from a baseline score, endpoint error, or restoration
would redefine the method after outcomes and violate both the firewall and the
frozen decision rule. The shared primary analyzer must therefore not run.

## Required scientific/protocol correction

A new binding correction must decide and freeze, without consulting
confirmation:

1. the exact mathematical mapping from the Joint Witness/P13 certificate to
   each IOI and Greater-Than site row;
2. the certificate row identity, graph-construction route, amplitudes,
   resource caps, precision policy, terminal-status mapping, and immutable
   output schema;
3. a plan-bound certificate queue and worker whose sources are pinned before
   execution;
4. the prespecified Greater-Than clean-competence computation and receipt;
5. whether development is formally retired as protocol-diagnostic data and a
   newly frozen untouched confirmation protocol becomes the first valid
   primary test.

Because development endpoints are now known, this cannot be described as a
retroactive completion of the already-frozen development primary analysis.
The defensible route is an explicit protocol version/corrigendum that preserves
all thresholds and treats confirmation as untouched until the corrected route
is fully frozen and audited.

## Current authorization state

- Development prediction/endpoint execution: complete.
- Frozen primary development analysis: blocked fail-closed.
- Confirmation: locked and unauthorized.
- GitHub: not pushed.
