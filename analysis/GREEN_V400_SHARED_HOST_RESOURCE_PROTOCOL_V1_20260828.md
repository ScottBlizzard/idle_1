# GREEN v4.0 shared-host resource protocol v1

Date: 2026-08-28

## Decision

The unavailable delegated cgroup-v2 memory controller on the university server
is an infrastructure detail, not a scientific gate. It does not alter the Joint
Witness estimand, theorem, numerical evaluator, radius schedule, confirmation
split, or any success threshold. Routine experiments may proceed on the shared
host without administrator intervention under the controls below.

The earlier cgroup-v2 requirement remains binding for the separate official
GREEN formal-certificate artifact. Its current host BLOCK must not be propagated
to non-certificate experiments or presented as evidence against the method.

## No-root experiment controls

Each routine experiment is launched by an external parent process with:

- a child-inherited per-process `RLIMIT_AS` virtual-address-space ceiling and
  disabled core dumps;
- a post-spawn, kernel-timerfd-backed monotonic wall deadline enforced against
  the leader and initial process group while the supervisor remains alive;
- a new initial worker process group, killed and checked for emptiness on
  deadline or policy violation;
- sampled aggregate process-tree RSS and swap telemetry;
- rejection, initial-group cleanup, and identity-checked cleanup of descendants
  observed by the sampler; and
- a machine-readable outcome-free resource report containing the applied
  policy, peak observations, exit status, and explicit guarantee scope.

No system setting, boot parameter, systemd unit, cgroup hierarchy, driver, or
shared-server service may be changed for this protocol.

## Claim boundary

The protocol may support the following paper statement:

> Experiments were run under an external monotonic timeout and a per-process
> address-space limit; peak process-tree resident memory was recorded by an
> independent Linux `/proc` sampler. The initial worker process group was
> cleaned after each job; observed unexpected child processes caused fail-closed
> termination.

It must not be described as cgroup-v2 enforcement, a hard aggregate process-tree
memory cap, or complete containment of adversarial `setsid`/double-fork workers.
The sampled tree-memory and descendant checks are operational controls that may
miss events between samples. The inherited `RLIMIT_AS` is a kernel limit. The
post-spawn timerfd deadline is externally enforced while the supervisor remains
alive, but no zero-overshoot claim is made; detection and kill latency are
recorded. This distinction belongs in implementation and reproducibility
documentation, not in the paper's scientific result narrative.

## Job scope

This protocol is limited to trusted non-certificate jobs: implementation and
regression tests, performance/resource benchmarks, calibration, baselines, and
scientific experiments governed by their own frozen non-certificate protocol.
It is not authorized to launch or publish the official GREEN formal certificate, cannot set
`CertificateResourceLock.production_authorized=true`, and cannot substitute its
report for the cgroup-v2 fields of that certificate lock.

## Required validation before server experiments

`analysis/green_v400_shared_host_resource_audit.py` performs outcome-blind Linux
fault injection for clean exit, kernel-timerfd timeout, unexpected descendants,
an observed tree-memory violation, and an oversized address-space allocation. A
pass validates these control mechanics on the host; it does not itself authorize
a scientific job or the separate cgroup-v2 formal certificate claim.
