# GREEN v4 hard single-process resource-lock candidate

Date: 2026-08-28

## Decision

The school host does not delegate its memory controller and its Docker daemon
is unavailable. A user `systemd` unit can be created, but a live probe showed
that the process remains in the parent memory cgroup with the effectively
unbounded `memory.limit_in_bytes`. Therefore a user-systemd scope must not be
represented as a hard memory lock.

For the trusted, single-threaded, CPU-only formal certificate process, a
different no-root kernel envelope is available as a production-lock candidate:

1. require real and effective non-root identities;
2. require exactly one task before the target exec;
3. require that effective `CAP_SYS_RESOURCE` is absent;
4. set both the soft and hard `RLIMIT_NPROC` values to one and verify readback;
5. force the standard OpenBLAS/OpenMP/MKL/NumExpr/Accelerate/BLIS thread-count
   environment variables to one before the target exec;
6. set both the soft and hard `RLIMIT_AS` values to the frozen address-space
   budget;
7. set the core-file limit to zero;
8. exec the hash-bound trusted certificate worker without creating another
   process; and
9. retain the external live `timerfd` deadline, `pidfd` exit observation,
   process-group cleanup, and selector-safe resource report.

This mode is default-off and is incompatible with an allow-descendants policy.
It does not modify system configuration or require root access.

## Kernel argument

Let the worker start with one Linux task and lack the root identity and
`CAP_SYS_RESOURCE`. After the hard `RLIMIT_NPROC` value is set to one, any
operation that would create another task for the same real UID would exceed the
limit. The process cannot raise the hard limit. Thus, for the closed trusted
worker, the task count remains one. A server probe verified both relevant
effects: `fork()` failed with `EAGAIN`, and Python thread creation failed.

With exactly one process, the inherited hard `RLIMIT_AS=M` bounds that process's
user virtual address space. Consequently the aggregate user-space address space
of the complete worker tree is also bounded by `M`; no sampling argument is
needed to establish this upper bound. RSS and swapped mapped pages remain
sampled observations, but cannot require mapped user address space beyond the
hard virtual-address-space ceiling.

The external monotonic deadline remains independent of worker cooperation. The
worker's initial process group is killed and verified empty after every terminal
state.

## Explicit limits

This is not cgroup-v2 enforcement and the report must continue to say so. The
candidate does not bound:

- kernel memory such as page tables and other per-process kernel bookkeeping;
- GPU device memory;
- memory allocated by an external service that the worker contacts;
- a malicious or unreviewed executable that regains privilege or deliberately
  escapes the hash-bound trusted code contract; or
- a workload that legitimately requires child processes or additional threads.

Accordingly, this candidate can replace the missing cgroup hard-memory gate only
for the hash-bound, CPU-only, single-threaded formal certificate worker and only
if the frozen production budget is stated as a user-space address-space bound.
It cannot be generalized to the model-serving or GPU experiment pipeline.

## Evidence

- transient user-systemd memory probe: ineffective; observed memory cgroup
  remained `/user.slice/user-1002.slice/user@1002.service` with limit
  `9223372036854771712`
- isolated kernel probe after `RLIMIT_NPROC=(1,1)`: `fork()` rejected with
  errno 11 (`EAGAIN`); new thread rejected
- base implementation commit: `61614d3`
- numerical single-thread hardening commit: `2a08f4d`
- isolated server tests: 11/11 targeted and 635/635 full-suite passed
- cgroup-v2 enforcement claimed by the new report: false
- permitted scope label:
  `trusted_hard_single_process_resource_lock_candidate`

## Remaining gate

Before this candidate can authorize a real certificate, it still requires:

1. a source/hash closure update that binds the strict exec shim and policy; and
2. a protocol freeze explicitly selecting the single-process address-space
   definition rather than claiming cgroup physical-memory accounting.

Until both items close, `production_authorized` remains false and no real
certificate, development, or confirmation outcome may run.

The first actual-shape probe failed closed before native computation because
NumPy's OpenBLAS initialization inherited a 64-thread default and the kernel
lock rejected every thread creation. This is expected enforcement, not a
numerical failure. The strict shim now overwrites the standard numerical-library
thread-count variables with one before exec. A fresh probe was therefore
required; its result follows.

The fresh `v2` probe completed the actual-shape closed-synthetic native path
under the corrected lock. Its externally supervised run took
`155.29491757502547` seconds, observed exactly one process and zero descendants,
peaked at `185319424` bytes RSS with zero swap, and exited zero. The strict
worker made six 384-bit native dispatches and returned the expected
`RESOURCE_INCONCLUSIVE / MAX_DEPTH_REACHED` without retaining a response Jet or
applying a scientific threshold. Before and after native execution it read back
the same hard limits: `RLIMIT_NPROC=(1,1)`, `RLIMIT_AS=(4294967296,4294967296)`,
`RLIMIT_CORE=(0,0)`, one thread, UID 1002, and no effective capabilities.

Independent canonical-hash readback passed for both artifacts:

- external wrapper report:
  `a7cd8a6de22111b82c343911dc44caaa4f4927220c89cf5c73a237583075d14a`
- selector-inaccessible numerics report:
  `0c8aa54a87468ee6247b48e9f44185d8198072aa1b947a41c9297127ec8ba640`

The historical `CertificateResourceLock v1` remains unchanged because it binds
the superseded constructive same-partition 512 count and cgroup-only resource
definition. A separate prepare-only v2 schema now freezes the corrected
full-history phase order and charges `2L+1` passes at each precision. For 17
radii and `L=14`, this is 493 official plus 493 audit passes, 986 total. The v2
schema rejects `production_authorized=true`, constructive audit history,
concurrency above one, center reuse, sampled descendant enforcement, cgroup
claims, and any token budget that does not cover the corrected full-history
cost. Its 11 focused tests pass.
