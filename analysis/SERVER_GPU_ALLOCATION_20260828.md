# Server GPU allocation note — 2026-08-28

- Host: `ccj@10.10.217.244`
- GREEN/ICLR work may use physical GPU indices `4,5,6,7` only.
- Physical GPU indices `0,1,2,3` are reserved for the user's collaborator and
  must not be selected, inspected for job control, or have their processes
  interrupted by this project.
- CPU-only MPFR, sanitizer, and resource-calibration jobs must continue to set
  `CUDA_VISIBLE_DEVICES=""` and `NVIDIA_VISIBLE_DEVICES=none`.
- This allocation does not authorize driver, cgroup, daemon, reboot, or other
  shared-system configuration changes.
