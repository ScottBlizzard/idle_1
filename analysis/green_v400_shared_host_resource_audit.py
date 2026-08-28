"""Linux fault-injection audit for the no-root shared-host resource envelope."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_schemas import sha256_canonical
import green_bridge_v400_shared_host as shared_host
from green_bridge_v400_shared_host import (
    SharedHostResourcePolicy, run_shared_host_command,
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8",
    ).strip()


def _read_stdout(result) -> str:
    return (Path(result.report_path).parent / "worker.stdout.log").read_text(
        encoding="utf-8"
    )


def _run_report(result) -> dict:
    return json.loads(Path(result.report_path).read_text(encoding="utf-8"))


def _cleanup_verified(result) -> bool:
    return bool(_run_report(result)["observations"]["cleanup"]["cleanup_verified"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    output = Path(args.output_directory).resolve()
    try:
        relative = output.relative_to(Path("/mnt/sdb").resolve())
    except ValueError as error:
        raise RuntimeError("audit output must be below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("audit output directory must be new below /mnt/sdb")
    output.mkdir(parents=True)

    base = SharedHostResourcePolicy(
        wall_deadline_seconds=5.0,
        per_process_address_space_bytes=512 << 20,
        observed_tree_memory_bytes=384 << 20,
        sample_interval_seconds=0.01,
    )
    clean = run_shared_host_command(
        [sys.executable, "-c", "import time; print('CLEAN_OK'); time.sleep(.05)"],
        cwd=ROOT, attempt_directory=output / "clean", policy=base,
    )
    timeout = run_shared_host_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=ROOT, attempt_directory=output / "timeout",
        policy=SharedHostResourcePolicy(
            0.5, 512 << 20, 384 << 20, 0.01,
        ),
    )
    descendant = run_shared_host_command(
        [sys.executable, "-c", (
            "import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
            "time.sleep(30)"
        )],
        cwd=ROOT, attempt_directory=output / "descendant", policy=base,
    )
    address_limit = run_shared_host_command(
        [sys.executable, "-c", (
            "import sys,time; "
            "exec(\"try:\\n bytearray(768 << 20)\\n print('LIMIT_FAILED')\\n"
            "except MemoryError:\\n print('ADDRESS_SPACE_DENIED')\"); "
            "time.sleep(.05)"
        )],
        cwd=ROOT, attempt_directory=output / "address_limit", policy=base,
    )
    observed_memory = run_shared_host_command(
        [sys.executable, "-c", "import time; x=bytearray(96<<20); time.sleep(30)"],
        cwd=ROOT, attempt_directory=output / "observed_memory",
        policy=SharedHostResourcePolicy(5, 512 << 20, 48 << 20, 0.01),
    )
    fast_descendant = run_shared_host_command(
        [sys.executable, "-c", (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])"
        )],
        cwd=ROOT, attempt_directory=output / "fast_descendant",
        policy=SharedHostResourcePolicy(5, 512 << 20, 384 << 20, 0.01),
    )
    escaped_descendant = run_shared_host_command(
        [sys.executable, "-c", (
            "import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
            "start_new_session=True); time.sleep(30)"
        )],
        cwd=ROOT, attempt_directory=output / "escaped_descendant", policy=base,
    )
    original_sampler = shared_host.sample_linux_process_tree
    try:
        def _injected_sampler_failure(_pid):
            raise OSError("injected sampler failure")

        shared_host.sample_linux_process_tree = _injected_sampler_failure
        sampler_failure = run_shared_host_command(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            cwd=ROOT, attempt_directory=output / "sampler_failure", policy=base,
        )
    finally:
        shared_host.sample_linux_process_tree = original_sampler
    exec_failure = run_shared_host_command(
        ["/definitely/not/a/real/green-command"],
        cwd=ROOT, attempt_directory=output / "exec_failure", policy=base,
    )

    checks = {
        "clean_command_completed": (
            clean.status == "COMPLETED" and "CLEAN_OK" in _read_stdout(clean)
            and _cleanup_verified(clean)
        ),
        "absolute_deadline_killed_worker": (
            timeout.status == "WALL_DEADLINE_REACHED"
            and timeout.termination_signal == 9
            and _cleanup_verified(timeout)
        ),
        "descendant_policy_cleaned_initial_group": (
            descendant.status == "DESCENDANT_PROCESS_POLICY_REACHED"
            and descendant.termination_signal == 9
            and _cleanup_verified(descendant)
        ),
        "rlimit_as_denied_oversized_allocation": (
            address_limit.status == "COMPLETED"
            and "ADDRESS_SPACE_DENIED" in _read_stdout(address_limit)
            and "LIMIT_FAILED" not in _read_stdout(address_limit)
            and _cleanup_verified(address_limit)
        ),
        "sampled_tree_memory_violation_cleaned_group": (
            observed_memory.status == "OBSERVED_TREE_MEMORY_REACHED"
            and observed_memory.termination_signal == 9
            and _cleanup_verified(observed_memory)
        ),
        "fast_root_exit_did_not_leave_same_group_child": (
            fast_descendant.status == "DESCENDANT_PROCESS_POLICY_REACHED"
            and _cleanup_verified(fast_descendant)
        ),
        "observed_setsid_descendant_was_identity_cleaned": (
            escaped_descendant.status == "DESCENDANT_PROCESS_POLICY_REACHED"
            and _cleanup_verified(escaped_descendant)
        ),
        "sampler_exception_failed_closed_and_cleaned": (
            sampler_failure.status == "SUPERVISOR_INFRASTRUCTURE_FAILED"
            and sampler_failure.termination_signal == 9
            and _cleanup_verified(sampler_failure)
        ),
        "exec_failure_was_reported_and_cleaned": (
            exec_failure.status == "WORKER_FAILED"
            and exec_failure.exit_code not in {None, 0}
            and _cleanup_verified(exec_failure)
        ),
    }
    passed = all(checks.values())
    report = {
        "schema_version": "green-v400-shared-host-resource-audit-v1",
        "report_contains_scientific_outcome": False,
        "supervisor_applied_scientific_threshold": False,
        "status": (
            "PASS_SHARED_HOST_RESOURCE_CONTROLS" if passed
            else "FAIL_SHARED_HOST_RESOURCE_CONTROLS"
        ),
        "checks": checks,
        "scope": {
            "shared_host_control_mechanics_validated": passed,
            "job_authorization_granted_by_this_audit": False,
            "eligible_job_scope": "trusted_non_certificate_jobs_only",
            "formal_cgroup_v2_certificate_claimed": False,
            "host_configuration_change_required": False,
            "aggregate_process_tree_memory_is_observational": True,
        },
        "runs": {
            name: {
                "status": result.status,
                "exit_code": result.exit_code,
                "termination_signal": result.termination_signal,
                "elapsed_seconds": result.elapsed_seconds,
                "peak_tree_rss_bytes": result.peak_tree_rss_bytes,
                "peak_tree_swap_bytes": result.peak_tree_swap_bytes,
                "peak_process_count": result.peak_process_count,
                "report_semantic_hash": result.report_semantic_hash,
            }
            for name, result in {
                "clean": clean,
                "timeout": timeout,
                "descendant": descendant,
                "address_limit": address_limit,
                "observed_memory": observed_memory,
                "fast_descendant": fast_descendant,
                "escaped_descendant": escaped_descendant,
                "sampler_failure": sampler_failure,
                "exec_failure": exec_failure,
            }.items()
        },
        "platform": platform.platform(),
        "provenance": {
            "repository_commit": _git("rev-parse", "HEAD"),
            "repository_clean_before_audit": not bool(_git(
                "status", "--porcelain=v1", "--untracked-files=all"
            )),
        },
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    report_path = output / "shared_host_resource_audit.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "checks": checks,
        "report": str(report_path),
        "report_semantic_hash": report["report_semantic_hash"],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
