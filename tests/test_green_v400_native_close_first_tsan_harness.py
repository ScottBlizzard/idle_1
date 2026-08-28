from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "native" / "green_v400_native_close_first_tsan_harness.cpp"
BUILD = ROOT / "scripts" / "build_green_v400_native_close_first_tsan_harness.sh"
RUN = ROOT / "scripts" / "run_green_v400_native_close_first_tsan_harness.sh"


def test_harness_has_frozen_payload_identity_and_384_bit_context() -> None:
    text = HARNESS.read_text(encoding="utf-8")
    for value in (
        "bc673467ac237e59e542634d38d02b8eaa12053cbb0abfc39e4dcaa6659ba3ee",
        "38f40999524d465b8ee58fcc8d2d1822caf9af6c36897a72bd404a8fff34fe62",
        "eb4c907ab4a86f3aac2fda445deed67099f2831c41e9712463688cccf1b6f008",
        "34bcd45371c08720c23f66d8f723dfc0249779e9e47eee5499c04d6064dc3560",
        "bd734f457bd3baee252af47f1c048dbd606ec15bf6a1b6533751c7bb943319c1",
        "28517632ULL",
        "kPrecisionBits = 384U",
    ):
        assert value in text


def test_close_precedes_release_and_worker_is_joined() -> None:
    text = HARNESS.read_text(encoding="utf-8")
    close_at = text.index("green_v400_native_precision_context_close_v1(context_handle);", text.index("while (true)"))
    done_check_at = text.index("dispatch_done.load", close_at)
    release_at = text.index("green_v400_native_audit_after_find_hook_release_v1();", close_at)
    join_at = text.index("dispatch_thread.join();", release_at)
    assert close_at < done_check_at < release_at < join_at
    assert "result.dispatch_status == 2" in text
    assert "result.metric_dispatch_entries == 0" in text
    assert "result.metric_active_dispatches == 0" in text
    assert "result.metric_peak_dispatches == 0" in text
    assert "result.post_close_info_status == 2" in text


def test_harness_report_is_outcome_free_and_strictly_statused() -> None:
    text = HARNESS.read_text(encoding="utf-8")
    assert '\"contains_scientific_outcome\\\":false' in text
    assert '\"scientific_threshold_applied\\\":false' in text
    assert "PASS_CLOSE_FIRST_PRELOCK_WAITER" in text
    assert "return pass ? 0 : 2" in text
    banned = re.compile(r"(ctypes|numpy|Py_Initialize|python)", re.IGNORECASE)
    assert banned.search(text) is None


def test_build_links_both_executable_and_dso_with_tsan() -> None:
    text = BUILD.read_text(encoding="utf-8")
    assert "-fsanitize=thread" in text
    assert "readelf -d" in text and "libtsan" in text
    assert "nm -D --defined-only" in text
    assert "green_v400_native_audit_after_find_hook_release_v1" in text
    assert '"-l:$dso_basename"' in text
    assert '-Wl,--no-as-needed' in text
    assert '-ltsan' in text
    assert 'object_file="${output_executable}.tsan.o"' in text
    assert text.index('-fsanitize=thread') < text.index('-c \\\n')


def test_runner_has_external_liveness_bound_and_immutable_outputs() -> None:
    text = RUN.read_text(encoding="utf-8")
    assert "timeout --signal=TERM --kill-after=30s 30m" in text
    assert "halt_on_error=1:exitcode=66" in text
    assert "refusing to overwrite existing evidence file" in text
    assert "green-v400-sanitizer-process-exit-v1" in text
    process_exit_at = text.index("process_status=$?")
    sealer_at = text.index("green_v400_native_close_first_tsan_seal.py")
    assert process_exit_at < sealer_at
    banned_in_tsan_process = re.compile(r"(ctypes|numpy|python)", re.IGNORECASE)
    assert banned_in_tsan_process.search(text[:process_exit_at]) is None
    assert '--raw-audit "$raw_audit_json"' in text
    assert '--output "$sealed_audit_json"' in text


def test_harness_is_valid_cxx17_syntax_when_compiler_is_available() -> None:
    completed = subprocess.run(
        ["g++", "-std=c++17", "-fsyntax-only", str(HARNESS)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
