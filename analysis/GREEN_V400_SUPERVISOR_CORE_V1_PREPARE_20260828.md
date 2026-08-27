# GREEN v4.0 external supervisor core v1 — prepare-only

Date: 2026-08-28

## Current decision

The prepare-only accounting/publication/deadline primitives are implemented. Formal production execution remains `BLOCK`.

Implemented and unit-tested:

- a thread-safe durable admission ledger that fsyncs each charge before returning admission; binds every token to attempt, precision, and exact-domain SHA-256; makes token replay accounting-idempotent while permitting dispatch start/finish exactly once; never refunds failure; and prohibits 512-bit admission until every admitted official pass is finished, the minimal all-radius pass count is present, and a partition-manifest identity is frozen;
- strict worker-candidate validation against the resource-lock semantic hash, allowed interval/resource statuses, threshold firewall, flat artifact identities with SHA-256 and byte length, path/symlink/special/unexpected-file rejection, and file/directory fsync;
- same-filesystem no-replace publication using Linux `renameat2(RENAME_NOREPLACE)` rather than overwrite;
- cgroup-v2 host probing and fail-closed `memory.max`/`memory.swap.max=0` write/readback helpers;
- an absolute `CLOCK_MONOTONIC` timerfd retained after worker exit through the publication window, plus pidfd monitoring with a direct `pidfd_open` syscall fallback when Python does not expose it;
- explicit refusal to launch production while `CertificateResourceLock.production_authorized` remains false.

This is not yet the complete `external_monotonic_supervisor_v1`. Independent review correctly identifies remaining P0 work: FD-bound validation and publication must be one transaction; the supervisor must create its own commit attestation rather than trust a worker-named manifest; admission must bind exact domain/token/attempt identities one-to-one to physical dispatch; cgroup creation/attach must be race-free and followed by memory-event monitoring, `cgroup.kill`, and `populated=0`; runtime executable/backend/resource identities must be rehashed; and timeout/OOM priority and administrative-record semantics must be frozen.

## Host audit

The current server is a hybrid hierarchy. Its cgroup-v2 mount is `/sys/fs/cgroup/unified`, but the active delegated path exposes no controllers, is not writable, and has none of `memory.max`, `memory.swap.max`, or `memory.events`. The memory controller remains on cgroup v1, which is not an authorized substitute. `timerfd_create`, `timerfd_settime`, and `renameat2` are present; direct kernel syscall 434 successfully opens a pidfd even though the selected Python build lacks `os.pidfd_open`.

The frozen machine report is `GREEN_V400_SUPERVISOR_ENVIRONMENT_V1_AUDIT_20260828.json`, semantic hash `ac1d75f5959b90ae1f371c7870dac8f8e23b798f07995e6d0697fe33f67f52be`.

Formal OOM/timeout fault injection and 17-radius dry orchestration require a pure or properly delegated cgroup-v2 host with the memory controller enabled. Changing to such a host is resource engineering only; it does not change the Joint Witness estimand or certificate mathematics, but it does require new hardware/resource identity hashes before authorization.
