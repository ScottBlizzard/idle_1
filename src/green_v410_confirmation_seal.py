"""Dual-implementation confirmation seal scanner for GREEN v4.1.

The scanner is read-only.  It does not authorize confirmation; it only emits
canonical evidence consumed by the later transition gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from green_v410_protocol import PROTOCOL_ID, canonical_json_bytes, sha256_bytes, sha256_canonical
from green_v410_schemas import with_artifact_self_hash


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_PATH = ROOT / "configs" / "green_v410_confirmation_seal_policy.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "protocol_id", "attempt_index", "target_protocol_ids",
        "phase_tokens", "output_path_tokens", "model_output_keys",
        "forbidden_true_flags", "execution_markers", "text_extensions",
        "binary_output_extensions", "ignored_directory_names",
        "max_text_file_bytes", "allowed_file_sha256s",
    }
    if set(payload) != required:
        raise ValueError("confirmation seal policy is not closed")
    if (
        payload["schema_version"] != "green-v410-confirmation-seal-policy-v1"
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["attempt_index"] != 1
    ):
        raise ValueError("confirmation seal policy identity mismatch")
    if payload["max_text_file_bytes"] <= 0:
        raise ValueError("confirmation seal text limit is invalid")
    if len(payload["target_protocol_ids"]) != len(set(payload["target_protocol_ids"])):
        raise ValueError("confirmation seal protocol IDs are duplicated")
    return payload


def seal_policy_sha256(policy: Mapping[str, Any]) -> str:
    return sha256_canonical(policy)


def _relative(path: Path, root: Path, root_label: str) -> str:
    return f"{root_label}/{path.relative_to(root).as_posix()}"


def _enumerate_pathlib(root: Path, ignored: set[str]) -> Iterator[Path]:
    candidates = sorted(root.rglob("*"), key=lambda path: path.as_posix())
    for path in candidates:
        relative_parts = path.relative_to(root).parts
        if any(part in ignored for part in relative_parts):
            continue
        if path.is_symlink():
            continue
        if path.is_file():
            yield path


def _enumerate_scandir(root: Path, ignored: set[str]) -> Iterator[Path]:
    pending = [root]
    files: list[Path] = []
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name in ignored:
                    continue
                if entry.is_symlink():
                    continue
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(path)
    yield from sorted(files, key=lambda path: path.as_posix())


def _walk_json(value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _parse_json_documents(text: str, suffix: str) -> list[Any]:
    if suffix == ".jsonl":
        documents = []
        for line in text.splitlines():
            if line.strip():
                documents.append(json.loads(line))
        return documents
    return [json.loads(text)]


def _path_is_suspicious(relative_path: str, policy: Mapping[str, Any]) -> bool:
    lowered = relative_path.lower()
    has_phase = any(token in lowered for token in policy["phase_tokens"])
    has_output = any(token in lowered for token in policy["output_path_tokens"])
    return has_phase and has_output


def _finding(relative_path: str, path: Path, reason: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "size_bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "reason": reason,
    }


def _detect_a(path: Path, relative_path: str, policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    digest = None
    allowed = set(policy["allowed_file_sha256s"])
    if allowed:
        digest = file_sha256(path)
        if digest in allowed:
            return []
    suspicious_path = _path_is_suspicious(relative_path, policy)
    if suffix in set(policy["binary_output_extensions"]):
        return [_finding(relative_path, path, "CONFIRMATION_BINARY_OUTPUT_PATH")] if suspicious_path else []
    if suffix not in set(policy["text_extensions"]):
        return []
    if path.stat().st_size > policy["max_text_file_bytes"]:
        return [_finding(relative_path, path, "SUSPICIOUS_TEXT_EXCEEDS_SCAN_LIMIT")] if suspicious_path else []
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeError:
        return [_finding(relative_path, path, "SUSPICIOUS_NON_UTF8_TEXT")] if suspicious_path else []
    lowered = text.lower()
    target = suspicious_path or any(protocol_id.lower() in lowered for protocol_id in policy["target_protocol_ids"])
    if not target:
        return []
    findings: set[str] = set()
    try:
        for document in _parse_json_documents(text, suffix):
            pairs = list(_walk_json(document))
            keys = {key for key, _ in pairs}
            scoped_output = False
            pending = [document]
            while pending:
                value = pending.pop()
                if isinstance(value, Mapping):
                    context = any(
                        str(value.get(field, "")).lower() == "confirmation"
                        for field in ("phase", "split", "role")
                    )
                    if context and set(value) & set(policy["model_output_keys"]):
                        scoped_output = True
                    pending.extend(value.values())
                elif isinstance(value, list):
                    pending.extend(value)
            if (suspicious_path and keys & set(policy["model_output_keys"])) or scoped_output:
                findings.add("CONFIRMATION_MODEL_DERIVED_JSON")
            if any(key in policy["forbidden_true_flags"] and value is True for key, value in pairs):
                findings.add("CONFIRMATION_EXECUTION_FLAG_TRUE")
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    if suffix in {".log", ".txt", ".out", ".err"} and any(
        marker in lowered for marker in policy["execution_markers"]
    ) and any(
        token in lowered or token in relative_path.lower() for token in policy["phase_tokens"]
    ):
        findings.add("CONFIRMATION_EXECUTION_LOG_MARKER")
    return [_finding(relative_path, path, reason) for reason in sorted(findings)]


def _detect_b(path: Path, relative_path: str, policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    # Independent formulation: classify by raw bytes first, then recursively
    # inspect parsed JSON values without calling scanner A's detector.
    suffix = path.suffix.casefold()
    if policy["allowed_file_sha256s"] and file_sha256(path) in policy["allowed_file_sha256s"]:
        return []
    path_suspicious = (
        any(token.casefold() in relative_path.casefold() for token in policy["phase_tokens"])
        and any(token.casefold() in relative_path.casefold() for token in policy["output_path_tokens"])
    )
    if suffix in policy["binary_output_extensions"]:
        return [_finding(relative_path, path, "CONFIRMATION_BINARY_OUTPUT_PATH")] if path_suspicious else []
    if suffix not in policy["text_extensions"]:
        return []
    size = os.path.getsize(path)
    if size > policy["max_text_file_bytes"]:
        return [_finding(relative_path, path, "SUSPICIOUS_TEXT_EXCEEDS_SCAN_LIMIT")] if path_suspicious else []
    raw = path.read_bytes()
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [_finding(relative_path, path, "SUSPICIOUS_NON_UTF8_TEXT")] if path_suspicious else []
    folded = decoded.casefold()
    in_scope = path_suspicious or any(item.casefold() in folded for item in policy["target_protocol_ids"])
    if not in_scope:
        return []
    reasons: list[str] = []
    try:
        documents = _parse_json_documents(decoded, suffix)
        flattened = [pair for document in documents for pair in _walk_json(document)]
        scoped_output = False
        stack = list(documents)
        while stack:
            value = stack.pop()
            if isinstance(value, Mapping):
                context = any(
                    str(value.get(field, "")).casefold() == "confirmation"
                    for field in ("phase", "split", "role")
                )
                if context and any(key in policy["model_output_keys"] for key in value):
                    scoped_output = True
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        any_output_key = any(key in policy["model_output_keys"] for key, _ in flattened)
        if (path_suspicious and any_output_key) or scoped_output:
            reasons.append("CONFIRMATION_MODEL_DERIVED_JSON")
        if any(key in policy["forbidden_true_flags"] and value is True for key, value in flattened):
            reasons.append("CONFIRMATION_EXECUTION_FLAG_TRUE")
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    phase_seen = any(token.casefold() in folded or token.casefold() in relative_path.casefold()
                     for token in policy["phase_tokens"])
    marker_seen = suffix in {".log", ".txt", ".out", ".err"} and any(
        marker.casefold() in folded for marker in policy["execution_markers"]
    )
    if phase_seen and marker_seen:
        reasons.append("CONFIRMATION_EXECUTION_LOG_MARKER")
    return [_finding(relative_path, path, reason) for reason in sorted(set(reasons))]


def _scan(
    roots: Mapping[str, Path],
    policy: Mapping[str, Any],
    *,
    scanner_id: str,
) -> dict[str, Any]:
    ignored = set(policy["ignored_directory_names"])
    if scanner_id == "PATHLIB_Rglob_V1":
        enumerate_files = _enumerate_pathlib
        detect = _detect_a
    elif scanner_id == "OS_SCANDIR_V1":
        enumerate_files = _enumerate_scandir
        detect = _detect_b
    else:
        raise ValueError("unknown confirmation seal scanner")
    findings: list[dict[str, Any]] = []
    file_count = 0
    byte_count = 0
    root_records = []
    for label, root in sorted(roots.items()):
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"seal root is not a directory: {resolved}")
        root_records.append({"label": label, "resolved_path": str(resolved)})
        for path in enumerate_files(resolved, ignored):
            file_count += 1
            byte_count += path.stat().st_size
            relative_path = _relative(path, resolved, label)
            findings.extend(detect(path, relative_path, policy))
    findings.sort(key=lambda row: (row["path"], row["reason"], row["file_sha256"]))
    summary = {
        "schema_version": "green-v410-confirmation-seal-scanner-summary-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "roots": root_records,
        "scanned_file_count": file_count,
        "scanned_byte_count": byte_count,
        "findings": findings,
        "seal_policy_sha256": seal_policy_sha256(policy),
    }
    return {
        "scanner_id": scanner_id,
        "summary": summary,
        "summary_sha256": sha256_canonical(summary),
    }


def dual_scan(roots: Mapping[str, Path], policy: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    first = _scan(roots, policy, scanner_id="PATHLIB_Rglob_V1")
    second = _scan(roots, policy, scanner_id="OS_SCANDIR_V1")
    return first, second


def build_seal_receipt(
    roots: Mapping[str, Path], policy: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[dict[str, Any], dict[str, Any]]]:
    first, second = dual_scan(roots, policy)
    identical = first["summary"] == second["summary"]
    findings = first["summary"]["findings"] if identical else [{
        "path": "<scanner-comparison>",
        "size_bytes": 0,
        "file_sha256": "0" * 64,
        "reason": "SCANNER_SUMMARIES_DIFFER",
    }]
    terminal = "PASS" if identical and not findings else "STOP_CONFIRMATION_SEAL_FAILED"
    receipt = with_artifact_self_hash({
        "schema_version": "green-v410-sfc-jwtec-confirmation-seal-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "scanner_reports": [first, second],
        "scanner_report_sha256s": [
            sha256_canonical(first), sha256_canonical(second),
        ],
        "summaries_identical": identical,
        "forbidden_findings": findings,
        "terminal_state": terminal,
        "scanned_roots": [str(path.resolve()) for _, path in sorted(roots.items())],
        "seal_policy_sha256": seal_policy_sha256(policy),
    })
    return receipt, (first, second)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--root", action="append", required=True, help="LABEL=PATH")
    args = parser.parse_args()
    roots: dict[str, Path] = {}
    for item in args.root:
        if "=" not in item:
            raise ValueError("--root must be LABEL=PATH")
        label, raw_path = item.split("=", 1)
        if not label or label in roots:
            raise ValueError("seal root label is empty or duplicated")
        roots[label] = Path(raw_path)
    receipt, _ = build_seal_receipt(roots, load_policy(args.policy))
    print(canonical_json_bytes(receipt).decode("utf-8"))


if __name__ == "__main__":
    main()
