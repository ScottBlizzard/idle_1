# GREEN-V410 P13 numerical repair — execution handoff

Date: 2026-09-07. This supersedes the **launch instructions** in the September 6
LunaMax handoff. It does not change the scientific protocol or authorize a new
formal fleet. Read the status distinction below before executing anything.

## 1. Diagnosis, not a scientific verdict

The old formal certificate stage failed for both tasks. The 32 launched workers
per task all failed without publishing a completed certificate. Each task still
has 108 captures and 864 materialized direction graphs; these are not lost.
The failed supervisor PIDs were 1720283 (IOI) and 1720284 (GT).

The immediate exception was `CERTIFICATE_IMPLEMENTATION_INVALID`: the direct
Taylor remainder did not intersect the integrated second-derivative enclosure.
That intersection check is correct and remains unchanged.

An isolated GT replay established the following:

- All five value/first/second interval jets on both `[-1,0]` and `[0,1]` were
  incorrectly returned as exact zero by the old backend.
- Independently evaluated point jets gave nonzero Taylor remainders. For
  example, the positive PAT_J remainder was approximately -5.4745656681e-10,
  while the old interval curvature was `[0,0]`.
- A minimal attention example proves a native numerical bug independently of
  any real scientific outcome: query `[1000000]`, keys `[[0],[1000000]]`, values
  `[[1],[1]]`, pivot 0. Its output is identically 1. The old backend returned an
  interval decoded as `[0,0]`.
- The old fixed-pivot exponential path overflows; the serializer passes a
  nonfinite MPFR number to `mpfr_get_z_2exp`, and the resulting zero significand
  is then decoded as exact zero. This is not acceptable certificate evidence.

Do not infer either P13 success or scientific failure from these invalid runs.
Do not use the old runtime measurements as a reliable ETA for the repaired
large-domain path: the executed arithmetic is different when the safeguard is used.

## 2. Implemented changes

Local implementation worktree:

`D:\ICLR_1\.worktrees\ICLR_1_v400_impl`

1. `native/green_v400_mpfr_backend.cpp`:
   - Explicit nonfinite export markers instead of fabricated exact-zero endpoints.
   - Resident-buffer validation rejects nonfinite or reversed intervals.
   - Interval multiplication no longer discards NaNs in min/max reductions;
     invalid inputs remain invalid. Reciprocal rejects zero-crossing intervals.
   - Large softmax differences use outward monotone probability bounds and exact
     first/second derivative identities. Ordinary-domain expression order and
     primitive counts are preserved.
2. Both Python native endpoint decoders reject explicit nonfinite markers.
3. `src/green_v410_p13_runtime.py` and the P13 supervisor block an uncheckpointed
   relaunch when an earlier launch log exists, including an empty log.
4. The supervisor stops dispatching new units on a hard failure, but drains and
   accounts for its already-running children while retaining its lock. It reports
   `FAILED_DRAINING` and exact child PIDs instead of orphaning them silently.
5. The fixed-budget evaluator accepts an optional outcome-blind node-progress
   callback. This does not change numerical inputs, outputs or stopping rules.
6. Diagnostic/full-acceptance entry points are supplied in
   `analysis/green_v410_certificate_repair_probe.py`. Diagnostic outputs cannot
   be mistaken for formal P13 records. New diagnostic launches have a directory
   lock; full acceptance has an immutable start marker and will not silently retry.
7. Exact-integer JSON I/O is configured explicitly in our certificate worker,
   supervisor, status, finalizer and diagnostic **processes only**. Python 3.11's
   default 4300-digit decimal guard cannot represent some valid MPFR-derived
   rational endpoints. This is an I/O setting, not a precision change or a change
   to any school-server system configuration. The canonical JSON schema and
   exact integer values are preserved.

No radii, leaf budget, precision, threshold, directions, model weights, site
population, frozen queue, original job ID, original output directory or scientific
acceptance rule was changed. No GitHub push was made.

## 3. Why the softmax enclosure is sound

For score intervals `[L_j,U_j]`, probability `p_i` satisfies

```
1 / (1 + sum_{j!=i} exp(U_j - L_i)) <= p_i
    <= 1 / (1 + sum_{j!=i} exp(L_j - U_i)).
```

Endpoint differences are outward rounded. Each reciprocal exponential sum is
evaluated after a constant maximum shift, with an exactly-one denominator term.
All exponential arguments are nonpositive. For `x <= -p`, the implementation
uses the enclosure `[0,2^-p]`, justified by `exp(x) <= exp(-p) < 2^-p`; it does
**not** replace a small positive number by exact zero.

With `m = sum_j p_j s'_j`, derivatives are bounded by the identities

```
p'_i  = p_i (s'_i - m)
m'    = sum_j (p'_j s'_j + p_j s''_j)
p''_i = p'_i (s'_i - m) + p_i (s''_i - m').
```

The fallback trigger is a fixed numerical range test on shifted score bounds
outside [-64,64], not a scientific success threshold or an outcome-driven rule.
The graph, frozen pivot metadata and represented real function remain unchanged.
The resulting enclosures can still be wide. Soundness does not imply useful
contraction, and no test/threshold should be weakened to make them useful.

## 4. Verified tests and remaining acceptance

Final server regression suite: **92 passed in 24.39 seconds**. This includes:

- the constant-one overflow regression at both precisions and both pivots;
- independent 1024-bit sampled analytic softmax value/derivative comparisons;
- wide-domain enclosures, 384/512 nesting, resident/JSON parity;
- explicit overflow export and NaN/infinity input rejection;
- all existing compiled kernel tests, including ordinary-path primitive counts;
- complete synthetic L4, three-radius certificates at layers 0, 4 and 8;
- cache-warmed resident/independent execution comparisons;
- failure-draining and uncheckpointed-relaunch tests.
- the full engineering CLI's success/failure/no-retry behavior and exact large
  integer JSON round trips.

Machine-readable test report:

`/mnt/sdb/ccj/green_v410_certificate_repair_20260907/repair_regression_final.xml`

The first repaired GT replay completed all native evaluations, but publishing
the two interval jets hit Python's 4300-digit integer conversion limit. Its three
point files were saved. The conversion bug is now fixed and independently tested
with a 30,000-bit integer round trip and no-clobber republication. The original
`gt_repaired_probe.log` is retained. An explicit engineering-only continuation
fills only those two missing interval files with identical numerical inputs and
the same repaired backend; it does not redo the three saved points.

**Not yet established:** repaired real-sample first-radius consistency, complete
real L4/384/512/three-radius certificates, independent AD overlap on those
certificates, repaired resource qualification, P13 contraction, or formal
runtime adoption. Unit tests are not a substitute for those checks.

## 5. Isolated server state

```
SSH:       ccj@10.10.217.244
Python:    /home/ccj/miniconda3/envs/green_bridge_20260805/bin/python
Extra:     /mnt/sdb/ccj/green_v400_formal_prepare_runtime/site-packages
Repair:    /mnt/sdb/ccj/green_v410_certificate_repair_20260907
Candidate: /mnt/sdb/ccj/green_v410_certificate_repair_20260907/patched_source
Backend:   <Candidate>/lib/libgreen_v400_mpfr_backend_repaired.so
Old run:   /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal
```

The original source and original backend were NOT replaced. Beware:
`<Repair>/src` is a symlink to the original frozen source. **Never deploy changes
through that symlink.** Only `<Candidate>/src` is a separate editable copy.

The isolated library was built using our existing MPFR/GMP installation at
`/mnt/sdb/ccj/green_v401_compiled_mpfr_runtime/shared`; no system installation or
shared-server configuration was changed. Do not rebuild an in-use shared library.

Deterministically selected debugging cases are the first recorded worker failures,
not rows chosen for favorable contraction:

```
GT job:  537ca957ca4e03fd3de84c630f8adf224adf0e5f69eaf48a9673111c36df217c
IOI job: 8bca00bd5614c6bf11677e6b99564269c7d6a968f165ffab0471ddc62ccb47c8
Both:    direction_00
```

At handoff preparation, these CPU-only diagnostic parents were launched:

- GT old-backend replay: 1889337, completed; five jets saved in `gt_first_radius`.
- IOI old-backend replay: 1891402, completed and reproduced the same Taylor
  intersection failure; its five domain files and log are preserved.
- GT first repaired replay: 1894632, ended with the serialization error; original
  log `gt_repaired_probe.log`, three preserved point files.
- GT serialization-fixed continuation: inspect the matching command line; log
  `gt_repaired_serialization_resume.log`, only two missing interval evaluations.
- IOI first repaired replay: Python parent 1896641, wrapper shell 1896640;
  log `ioi_repaired_probe.log`. It started before the process-local integer-I/O fix.

PIDs are historical identifiers, not proof of current liveness. Check command
lines and start times before drawing conclusions. Do not kill or duplicate runs.
The GT first repaired replay predates the diagnostic lock. The continuation and
IOI repaired replay hold locks. Always verify both processes and logs, not only
lock-file presence.

## 6. LunaMax sequence — no scientific redesign

### A. Finish the already-running bounded smoke checks

Read the latest GT continuation log and the IOI repaired log; verify only one
parent per task/output directory.
Each smoke check evaluates `[-1,0]`, `[0,1]`, and the three point domains at 384
bits. Require all five JSON files and `FIRST_RADIUS_TAYLOR_PASS` in each repaired
log. That marker is only a first-radius consistency check, not P13 PASS.

The running IOI process still has the old decimal guard in its interpreter even
though the on-disk code is repaired. Let it finish; do not interrupt it. If its
**only** failure is the same `Exceeds the limit (4300 digits)` conversion error,
verify its parent and all children have exited, retain its log and saved points,
then run exactly one serialization-fixed diagnostic continuation:

```bash
PYTHONPATH=/mnt/sdb/ccj/green_v400_formal_prepare_runtime/site-packages \
PYTHONINTMAXSTRDIGITS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/ccj/miniconda3/envs/green_bridge_20260805/bin/python -u \
  /mnt/sdb/ccj/green_v410_certificate_repair_20260907/patched_source/analysis/green_v410_certificate_repair_probe.py --real \
  /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal/graphs/ioi/8bca00bd5614c6bf11677e6b99564269c7d6a968f165ffab0471ddc62ccb47c8/direction_00 \
  /mnt/sdb/ccj/green_v410_certificate_repair_20260907/patched_source/lib/libgreen_v400_mpfr_backend_repaired.so \
  /mnt/sdb/ccj/green_v410_certificate_repair_20260907/ioi_repaired_first_radius
```

Use a fresh, exclusively-created `ioi_repaired_serialization_resume.log`. This
specific diagnostic continuation is already decided and is not a formal retry.
The script skips the preserved point files. Do not use it for any other error.

If a smoke check fails for any other reason, preserve its files and traceback. Do not launch full
acceptance, select another row, relax nesting/intersections, increase L4 or shrink
the radius. Return the specific numerical failure for engineering diagnosis.

### B. Only after both smoke checks pass: full numerical acceptance

Run the following separately for the two fixed cases, at most two single-thread
CPU processes in total. The `full_gt` and `full_ioi` roots must be fresh. Inspect
them first; never truncate a prior run log or relaunch past a `.started.json`.

```bash
REPAIR=/mnt/sdb/ccj/green_v410_certificate_repair_20260907
CANDIDATE=$REPAIR/patched_source
PARENT=/mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal
PYTHON=/home/ccj/miniconda3/envs/green_bridge_20260805/bin/python
BACKEND=$CANDIDATE/lib/libgreen_v400_mpfr_backend_repaired.so
export PYTHONPATH=$CANDIDATE/src:/mnt/sdb/ccj/green_v400_formal_prepare_runtime/site-packages
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONINTMAXSTRDIGITS=0
cd "$CANDIDATE"

# GT: execute once, preferably as a detached subprocess with an exclusively-created log.
"$PYTHON" -u analysis/green_v410_certificate_repair_probe.py --full \
  "$PARENT/graphs/greater_than/537ca957ca4e03fd3de84c630f8adf224adf0e5f69eaf48a9673111c36df217c/direction_00" \
  "$BACKEND" \
  "$PARENT/capture/greater_than/537ca957ca4e03fd3de84c630f8adf224adf0e5f69eaf48a9673111c36df217c/capture_manifest.json" \
  "$REPAIR/full_gt/acceptance.json" 0

# IOI: execute once under the same conditions.
"$PYTHON" -u analysis/green_v410_certificate_repair_probe.py --full \
  "$PARENT/graphs/ioi/8bca00bd5614c6bf11677e6b99564269c7d6a968f165ffab0471ddc62ccb47c8/direction_00" \
  "$BACKEND" \
  "$PARENT/capture/ioi/8bca00bd5614c6bf11677e6b99564269c7d6a968f165ffab0471ddc62ccb47c8/capture_manifest.json" \
  "$REPAIR/full_ioi/acceptance.json" 0
```

These commands execute the existing complete certificate function, unchanged L4,
384/512 bits and all three radii. They check the four independent captured AD
derivatives and the existing resource ceilings, and publish only engineering
acceptance artifacts. They never publish a formal `p13_record.json`.

Do not silently restart a failed numerical-acceptance run. Its failure receipt
and start marker remain part of the engineering record. Any resource timeout is
not a numerical success and not a P13 pass.

### C. Report the acceptance result; do not automatically open a formal fleet

Even two numerical-acceptance passes do not establish P13 contraction across
1,728 directions. Before a formal launch, the repaired implementation requires a
documented engineering adoption and resource-eligibility decision. In particular,
the existing queue binds the old backend and source into every job identity.
Changing the backend cannot honestly preserve that runtime identity. Do NOT edit
old queues, fake their identities, bypass runtime closure, or treat an incomplete
failed attempt as an unstarted one. The September 6 launch commands are suspended.

The concrete next deliverable is the two numerical-acceptance receipts with
resource observations and any exact failure. No Pro consultation is required
merely to execute these checks. A genuinely new scientific/attempt decision must
be identified explicitly, not hidden inside an engineering restart.

## 7. Invariants

- CPU only for this handoff; no GPU is needed. If later authorized GPU work is
  performed, only devices 4–7 are eligible, never 0–3.
- Do not touch other users' processes, especially RLS_CVPR2027.
- Keep large artifacts on `/mnt/sdb/ccj`, never the server root disk.
- No GitHub push, goal mode, or recurring monitor.
- Keep prior failed formal records and diagnostic evidence intact.
- Never label an invalid enclosure, an engineering smoke result, or an unfinished
  run as a successful scientific certificate.
