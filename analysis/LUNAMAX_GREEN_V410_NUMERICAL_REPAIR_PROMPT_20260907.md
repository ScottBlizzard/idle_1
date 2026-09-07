Continue the GREEN-V410 numerical-repair execution work. Read
`D:\ICLR_1\analysis\GREEN_V410_P13_NUMERICAL_REPAIR_HANDOFF_20260907.md`
completely first. The implementation is in
`D:\ICLR_1\.worktrees\ICLR_1_v400_impl`, not the repository-root source tree.

The September 6 formal launch instructions are suspended. A native MPFR
nonfinite-to-zero bug and a Python exact-integer serialization failure have been
identified and repaired. Do not confuse unit-test success with a real certificate
or P13 success. Do not reopen the full formal fleet.

Your next actions are already specified:

1. Connect to `ccj@10.10.217.244`. Inspect the current bounded diagnostic PIDs and
   logs in `/mnt/sdb/ccj/green_v410_certificate_repair_20260907`. Do not duplicate
   or interrupt them. The GT serialization-fixed continuation is filling only
   two missing interval outputs; its three point files are preserved. The IOI
   first replay started before the integer-I/O fix. If it finishes with only the
   documented 4300-digit conversion error, execute the single preauthorized
   serialization-fixed continuation from the handoff, preserving its saved points
   and old log. Never repeat an expensive calculation simply because SSH closed.
2. Require `FIRST_RADIUS_TAYLOR_PASS` and all five domain files for both repaired
   cases. These are engineering smoke checks, not scientific passes. Stop new
   work on any different error and report its exact evidence.
3. Only after both smoke checks pass, run the two specified full numerical
   acceptance commands. They use the unchanged complete L4, 384/512-bit,
   three-radius certificate and check independent AD overlap and resource
   ceilings. At most two single-thread CPU acceptance processes may run. Launch
   them detached with exclusively-created logs so they survive SSH loss. Respect
   their locks and immutable start markers; do not silently retry.
4. Return the exact engineering acceptance receipts and resource observations.
   Do not launch the formal P13 fleet or confirmation. Correct runtime adoption
   and resource eligibility must be resolved explicitly: the original queues bind
   the defective backend into their job identities. Never edit those queues,
   bypass closure, overwrite the original backend/source, or disguise a restart.

No new theory or Pro consultation is needed to execute these steps. Never loosen
an intersection/nesting/AD check, choose a different debugging row for favorable
results, change radii/directions/thresholds/L4, or reinterpret invalid intervals
as certificates. A sound but wide interval is an honest result, not permission to
tune the scientific protocol.

Do not deploy through `<repair-root>/src`: it is a symlink into the frozen source.
Only `<repair-root>/patched_source` contains the isolated candidate. Keep large
artifacts under `/mnt/sdb/ccj`. This work is CPU-only; do not use GPUs 0–3, and do
not touch other users' processes or system configuration. Do not push to GitHub,
create a goal, or enable recurring monitoring. Give concise Chinese updates on
meaningful progress and the distinction between software acceptance and paper
results.
