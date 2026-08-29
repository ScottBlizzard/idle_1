# GREEN v4 development-authorization request

Date: 2026-08-29  
Requested authority: scientific binding plus development only  
Confirmation remains sealed and unauthorized

## Executive status

All currently authorized prepare-only engineering has been completed without
opening an untouched scientific outcome.  The isolated server validation suite
passes `768 passed, 51 skipped`.  Both latest sealed execution plans compile
with `execution_enabled=false` and pass the independent prepare-bundle audit.
They intentionally stop at `PLAN_COMPILED_BLOCKED_BY_BASELINES` because the
Grant activation-capture estimand is not yet uniquely bound.

Latest plan identities:

| Task | Plan SHA-256 | Queue sizes | Audit |
|---|---|---|---|
| IOI | `b48610c4f2a8bb52b3bbee0b40a1336f3a9bbffced5e6e65955d669ac85a19fa` | development prediction 1152; development Grant 9; development endpoint 1152; confirmation prediction 1152; confirmation Grant 9; confirmation endpoint 1152; replay 576 | `PASS_PREPARE_BUNDLE_AUDIT` |
| Greater-Than | `4ad78aa524bbf8f6b9db6bae3b8e326b616802a3a3ae8676033324ca11d194f1` | development prediction 2592; development Grant 9; development endpoint 2592; confirmation prediction 3456; confirmation Grant 9; confirmation endpoint 3456; replay 1728 | `PASS_PREPARE_BUNDLE_AUDIT` |

No plan in the repository or on the server authorizes real outcomes.

## Engineering closure completed after the prior decision

1. The numerical replay route is target--target reproducibility only and
   cannot define a scientific null or threshold.
2. Endpoint normalization is symmetric and uses internally computed target
   and patched RMS denominators.
3. Direction tensors, rows, payload bytes, and model constants are bound by
   hashes.  The endpoint payload is unavailable to prediction processes.
4. The frozen float32 checkpoint is evaluated in float64 for response
   differences.  Historical same-checkpoint audits showed that the earlier
   float32 discrepancy was cancellation, not a scientific effect.
5. Integrated gradients, single-point HVP, multi-step HVP, finite activation
   response, and the empirical four-branch comparator have actual frozen-model
   execution paths and historical resource measurements on both tasks.
6. The formal process entry point validates the sealed plan, model-session
   hash, source hashes, GPU 4--7 policy, `/mnt/sdb` output root, deterministic
   environment, scientific phase authorization, direction row, and
   non-overwriting atomic artifact path before execution.
7. The checked-in shared analyzer implements matched coverage, prompt-cluster
   resampling, and the simultaneous primary-comparator decision rule.

## Grant primary-source correction

The primary paper states that its transformer comparison uses natural and
intervened residual-stream vectors at the intervention position.  The author
code explicitly selects a layer and token position before computing paired MSE,
Sinkhorn EMD, matching costs, and natural--natural controls.

For GREEN's ordinary intervention, however, the entire clean `resid_post`
vector is installed into the corrupt run.  At the intervention position this
vector is exactly natural by construction.  A literal Grant diagnostic is
therefore degenerate and cannot be treated as an informative baseline.  The
repository audit
`analysis/CODEX_GREEN_V400_GRANT_CAPTURE_SEMANTICS_AUDIT_20260829.md`
documents why the previous `READY` status was premature.

### Recommended binding

Approve a clearly labelled **Grant-style downstream contextual-divergence
extension**, not an exact Grant replication:

- cohort unit: the already sealed phase-by-candidate-layer prompt cohort;
- ordinary intervention: corrupt run with the full clean center patched at the
  candidate `resid_post` site;
- natural reference: clean run for the paired prompt;
- measurement site: `blocks.10.hook_resid_post` at the same task-defined token
  position used by the candidate site;
- reason for layer 10: it is strictly downstream of every candidate layer
  0--8, common to IOI and Greater-Than, and distinct from both task-specific
  structural endpoints;
- one vector per sealed prompt/site row; no GREEN or endpoint directions enter;
- metrics: the already pinned Grant panel and deterministic natural--natural
  control;
- role: nonprimary cohort diagnostic only, never broadcast to rows, never
  counted as a GREEN win, and never used to alter selection or thresholds;
- firewall: commit every phase-layer Grant packet before opening that phase's
  endpoints; confirmation remains inaccessible until separately authorized.

Also report, as an analytic applicability statement, that literal
intervention-site divergence of the full-vector patch is zero by construction.
This preserves a fair comparison to Grant while making the genuinely new
question explicit: whether a natural vector becomes contextually divergent
after downstream computation in the corrupt context.

If this recommendation is rejected, the alternative binding should be to mark
literal Grant divergence N/A for full-residual patching and remove it from the
required execution gate.  It must not remain nominally required without a
capture definition.

## Requested decision

Please return a binding decision that does all of the following:

1. approve, amend, or reject the recommended Grant-style downstream
   contextual-divergence extension, with exact capture and pairing semantics;
2. authorize Codex to implement and outcome-blind test that route, rebuild the
   two sealed plans, and run a final independent prepare audit;
3. if that final audit passes without a scientific change, explicitly
   authorize **development execution only** for IOI and Greater-Than under the
   sealed queues, firewall, GPU 4--7 policy, and `/mnt/sdb` storage policy;
4. state that confirmation remains one-shot sealed and requires a later,
   separate authorization after the frozen development analyzer receipt;
5. state whether numerical replay may run at the start of development as the
   already-bound reproducibility gate;
6. provide any mandatory stop conditions or artifact fields for the activated
   development plan and authorization receipt.

Do not authorize confirmation, post-outcome redesign, threshold changes,
direction changes, universe changes, or reinterpretation of Grant as a row-level
winner.
