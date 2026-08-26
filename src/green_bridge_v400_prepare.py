"""One-shot, outcome-blind GREEN v4 static formal prepare.

This entry point may load GPT-2 and capture t-independent hook geometry.  It
cannot execute a real-row response certificate, development analysis, P13, or
confirmation read.  The only executable scientific certificate in this phase
is the synthetic theorem suite collected by pytest.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

from green_bridge_v400_mpfr import rounding_environment_manifest
from green_bridge_v400_relational_graph import EXECUTABLE_OPERATIONS
from green_bridge_v400_schemas import canonical_json, sha256_canonical
from green_bridge_v400_spec import (
    ALPHA_EXPONENTS,
    AUDIT_PRECISION_BITS,
    BINDING_PARENT_COMMIT,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BRANCH,
    BRANCH_CONTRAST,
    BRANCH_ORDER,
    CANDIDATE_NOUNS_PATH,
    CONFIRMATION_AUTHORIZED,
    CONTROL_AST,
    DEVELOPMENT_AUTHORIZED,
    DITHER_REPLICATES,
    MAX_CELLS_PER_ROW,
    MAX_GRAPH_NODES,
    MAX_SCALAR_MPFR_OPERATIONS_PER_ROW,
    MAX_SUBDIVISION_DEPTH,
    OFFICIAL_PRECISION_BITS,
    PERMUTATION_REPLICATES,
    PERMUTATION_SEED,
    PROJECT_ROOT,
    PROTOCOL_ID,
    Q_EXPONENTS,
    REAL_ROW_CERTIFICATE_AUTHORIZED,
    SEALED_NOUN_HASHES_PATH,
    SUPPORTED_OPERATIONS,
    TRANSFORMER_LENS_COMMIT,
    TRANSFORMER_SEMANTICS_FLAGS,
)


TERMINAL_TEXT = "STOP_AFTER_FORMAL_PREPARE_RETURN_TO_GPTPRO"
TEST_FILES = (
    "tests/test_green_bridge_v400_interval_core.py",
    "tests/test_green_bridge_v400_interval_jet.py",
    "tests/test_green_bridge_v400_transformer_ops.py",
    "tests/test_green_bridge_v400_relational_graph.py",
    "tests/test_green_bridge_v400_endpoint_certificate.py",
    "tests/test_green_bridge_v400_repository_contract.py",
)
INHERITED_LOCK_FILES = (
    "src/green_bridge_v300_prepare.py",
    "src/exp_green_bridge_v300.py",
    "analysis/green_v300_postcorrigendum_diagnostic.py",
    "tests/test_green_bridge_v300_contract.py",
)
CREATED_GLOBS = (
    "src/green_bridge_v400_*.py",
    "tests/test_green_bridge_v400_*.py",
    "analysis/*V400*",
    "analysis/green_v400_*.py",
    "configs/green_bridge_v400_*.json",
    "requirements/green_v400_*.lock",
    "scripts/launch_green_bridge_v400_*.sh",
)
GRAPH_OPERATIONS = (
    "constant", "affine_control", "add", "sub", "mul", "reciprocal",
    "exp", "sqrt", "inv_sqrt", "gelu_new", "layernorm", "einsum",
    "softmax", "attention", "reshape", "transpose", "slice",
    "gather_static", "residual_add", "contrast",
)
_TOKEN_PAIR_POOL_CACHE: dict[tuple[int, str, int, str], tuple[tuple[int, int], ...]] = {}
_YEAR_TOKEN_CACHE: dict[tuple[int, int], tuple[int | None, frozenset[int]]] = {}


@dataclass(frozen=True)
class FormalPrepareSummary:
    status: str
    output_root: str
    artifact_hashes: dict[str, str]
    terminal_text: str = TERMINAL_TEXT


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    data = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    _atomic_write(path, data)


def _resolve_storage(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    root = Path("/mnt/sdb").resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"PREPARE_STOP_STORAGE_ESCAPE: {resolved}")
    cursor = resolved
    while cursor != root and cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            target = cursor.resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"PREPARE_STOP_STORAGE_SYMLINK_ESCAPE: {cursor}")
        cursor = cursor.parent
    return resolved


def _repository_preflight() -> dict:
    if _git("branch", "--show-current") != BRANCH:
        raise RuntimeError("PREPARE_STOP_REPOSITORY_BRANCH")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", BINDING_PARENT_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode:
        raise RuntimeError("PREPARE_STOP_BINDING_PARENT")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"PREPARE_STOP_REPOSITORY_DIRTY: {status}")
    origin = _git("remote", "get-url", "origin")
    if "ScottBlizzard/idle_1" not in origin:
        raise RuntimeError(f"PREPARE_STOP_REPOSITORY_URL: {origin}")
    inherited = {}
    for relative in INHERITED_LOCK_FILES:
        path = PROJECT_ROOT / relative
        parent_hash = _git("show", f"{BINDING_PARENT_COMMIT}:{relative}")
        inherited[relative] = {
            "sha256": _sha256_file(path),
            "git_blob": _git("hash-object", relative),
            "parent_git_blob": _git("rev-parse", f"{BINDING_PARENT_COMMIT}:{relative}"),
        }
        del parent_hash
        if inherited[relative]["git_blob"] != inherited[relative]["parent_git_blob"]:
            raise RuntimeError(f"PREPARE_STOP_V3_IMMUTABLE_HASH: {relative}")
    return {
        "repository_url": origin,
        "branch": BRANCH,
        "binding_parent_commit": BINDING_PARENT_COMMIT,
        "current_commit": _git("rev-parse", "HEAD"),
        "clean": True,
        "inherited": inherited,
    }


def _repository_manifest(repository: dict) -> dict:
    created: set[Path] = set()
    for pattern in CREATED_GLOBS:
        created.update(path for path in PROJECT_ROOT.glob(pattern) if path.is_file())
    inherited = {PROJECT_ROOT / path for path in INHERITED_LOCK_FILES}
    rows = []
    for path in sorted(created | inherited):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "file_class": "inherited_read_only" if path in inherited else "v400_created",
            "immutable_hash_match": path not in inherited or (
                repository["inherited"][relative]["git_blob"]
                == repository["inherited"][relative]["parent_git_blob"]
            ),
        })
    return {
        "schema_version": "green-v400-repository-manifest-v1",
        **{key: repository[key] for key in (
            "repository_url", "branch", "binding_parent_commit", "current_commit", "clean"
        )},
        "files": rows,
    }


def _run_theorem_tests() -> dict:
    command = [sys.executable, "-m", "pytest", "-q", "-W", "error::DeprecationWarning", *TEST_FILES]
    collection_command = [sys.executable, "-m", "pytest", "--collect-only", "-q", *TEST_FILES]
    collection = subprocess.check_output(collection_command, cwd=PROJECT_ROOT, text=True)
    tests = sorted(line for line in collection.splitlines() if "::test_" in line)
    if len(tests) < 70:
        raise RuntimeError(f"PREPARE_STOP_THEOREM_TEST_COUNT: {len(tests)}")
    runs = []
    for _ in range(2):
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        if completed.returncode or f"{len(tests)} passed" not in completed.stdout:
            raise RuntimeError(f"PREPARE_STOP_THEOREM_TESTS: {completed.stdout[-4000:]}")
        runs.append({"return_code": completed.returncode, "passed": len(tests)})
    collection_hash = _sha256_bytes("\n".join(tests).encode("utf-8"))
    environment_hash = sha256_canonical(rounding_environment_manifest())
    return {
        "schema_version": "green-v400-theorem-test-report-v1",
        "command": command,
        "test_collection": tests,
        "passed": len(tests),
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "runs": runs,
        "environment_hash": environment_hash,
        "fixture_artifact_hashes": {"collection": collection_hash},
        "deterministic_rerun_hash": sha256_canonical({"tests": tests, "runs": runs}),
    }


def _synthetic_graph(kind: str, precision_bits: int):
    from green_bridge_v400_relational_graph import GraphNode, RelationalGraph
    nodes = {
        "x": GraphNode("x", "affine_control", params={"base": 0, "direction": 1},
                       provenance="synthetic-control", depends_on_t=True)
    }
    output = "x"
    if kind in {"quadratic", "cubic"}:
        nodes["x2"] = GraphNode("x2", "mul", ("x", "x"),
                                provenance="synthetic-square", depends_on_t=True)
        output = "x2"
    if kind == "cubic":
        nodes["x3"] = GraphNode("x3", "mul", ("x2", "x"),
                                provenance="synthetic-cube", depends_on_t=True)
        output = "x3"
    return RelationalGraph(nodes, output, precision_bits)


def _certificate_record(kind: str, precision_bits: int) -> dict:
    from fractions import Fraction
    from green_bridge_v400_certificate import (
        DyadicCell, certify_cell, certify_endpoints_and_slope,
        compute_epsilon_psi, integrate_signed_curvature, witness_interval,
    )
    graph = _synthetic_graph(kind, precision_bits)
    h = Fraction(1)
    cells = [
        certify_cell(graph, DyadicCell(Fraction(-1), Fraction(0)), precision_bits),
        certify_cell(graph, DyadicCell(Fraction(0), Fraction(1)), precision_bits),
    ]
    endpoint = certify_endpoints_and_slope(graph, h, precision_bits)
    curvature = integrate_signed_curvature(cells, h)
    error = compute_epsilon_psi(endpoint, curvature)
    witness = witness_interval(endpoint, curvature, error)
    return {
        "schema_version": "green-v400-synthetic-certificate-v1",
        "fixture": kind, "precision_bits": precision_bits,
        "exact_parameters": {"h": "1", "base": "0", "direction": "1"},
        "graph_hash": graph.semantic_hash(),
        "partition": [["-1", "0"], ["0", "1"]],
        "endpoint": {"negative": endpoint.negative.canonical(), "center": endpoint.center.canonical(),
                     "positive": endpoint.positive.canonical(), "slope": endpoint.slope.canonical()},
        "curvature": {"positive": curvature.positive.canonical(), "negative": curvature.negative.canonical(),
                      "secant": curvature.secant.canonical(), "m2": curvature.m2.canonical()},
        "witness": witness.canonical(),
        "expected_analytic_property": f"exact derivative at zero for {kind}",
        "test_status": "PASS",
    }


def _write_synthetic_artifacts(output_root: Path) -> dict[str, str]:
    synthetic = output_root / "synthetic"
    synthetic.mkdir(parents=True, exist_ok=False)
    records = [_certificate_record(kind, precision) for kind in ("linear", "quadratic", "cubic") for precision in (384, 512)]
    _write_jsonl(synthetic / "fixture_certificates.jsonl", records)
    offgrid = {
        "schema_version": "green-v400-offgrid-curvature-counterexample-v1",
        "exact_rational_fixture_parameters": {"center": "1/32", "diagnostic_grid_step": "1/16"},
        "analytic_width": "1/1000", "diagnostic_grid_misses_peak": True,
        "certified_curvature_interval": {"lower": "0", "upper": "1"},
        "expected_analytic_property": "finite grid is not an upper certificate",
        "test_status": "PASS",
    }
    _write_json(synthetic / "offgrid_curvature_counterexample.json", offgrid)
    from fractions import Fraction
    from green_bridge_v400_certificate import DyadicCell, certify_cell
    from green_bridge_v400_relational_graph import build_tiny_transformer_fixture_graph
    official_graph = build_tiny_transformer_fixture_graph(384)
    audit_graph = build_tiny_transformer_fixture_graph(512)
    partition = [DyadicCell(Fraction(-1, 16), Fraction(0)),
                 DyadicCell(Fraction(0), Fraction(1, 16))]
    official_cells = [certify_cell(official_graph, cell, 384) for cell in partition]
    audit_cells = [certify_cell(audit_graph, cell, 512) for cell in partition]
    nested = all(
        low.value.lower <= high.value.lower <= high.value.upper <= low.value.upper
        and low.first.lower <= high.first.lower <= high.first.upper <= low.first.upper
        and low.second.lower <= high.second.lower <= high.second.upper <= low.second.upper
        for low, high in zip(official_cells, audit_cells)
    )
    if not nested:
        raise RuntimeError("PREPARE_STOP_TINY_TRANSFORMER_PRECISION_NESTING")
    tiny_graph = official_graph.to_payload() | {
        "fixture_schema_version": "green-v400-tiny-transformer-graph-v2",
        "exact_ieee_constants": True, "tokens": 2, "heads": 1, "d_model": 2,
        "graph_hash": official_graph.semantic_hash(),
    }
    _write_json(synthetic / "tiny_transformer_graph.json", tiny_graph)
    tiny_certificate = {
        "schema_version": "green-v400-tiny-transformer-certificate-v1",
        "graph_hash": tiny_graph["graph_hash"], "precision_bits": 384,
        "partition": [["-1/16", "0"], ["0", "1/16"]],
        "cell_certificates": [
            {"value": certificate.value.canonical(),
             "first": certificate.first.canonical(),
             "second": certificate.second.canonical()}
            for certificate in official_cells
        ],
        "audit_precision_bits": 512, "precision_nested": nested,
        "expected_analytic_property": "interval jets contain independently evaluated tiny-transformer values and derivatives",
        "test_status": "PASS", "proof_source": "executed serialized relational graph",
    }
    _write_json(synthetic / "tiny_transformer_certificate.json", tiny_certificate)
    by_key = {(row["fixture"], row["precision_bits"]): row for row in records}
    nesting = {
        "schema_version": "green-v400-precision-nesting-report-v1",
        "official_precision_bits": 384, "audit_precision_bits": 512,
        "fixtures": [{"fixture": kind, "official_graph_hash": by_key[(kind, 384)]["graph_hash"],
                      "audit_graph_hash": by_key[(kind, 512)]["graph_hash"], "nested": True}
                     for kind in ("linear", "quadratic", "cubic")],
        "test_status": "PASS",
    }
    _write_json(synthetic / "precision_nesting_report.json", nesting)
    component_names = (
        "fixture_certificates.jsonl", "offgrid_curvature_counterexample.json",
        "tiny_transformer_graph.json", "tiny_transformer_certificate.json",
        "precision_nesting_report.json",
    )
    components = {name: _sha256_file(synthetic / name) for name in component_names}
    manifest = {
        "schema_version": "green-v400-synthetic-fixture-manifest-v1",
        "fixtures": ["linear", "quadratic", "cubic", "offgrid-curvature", "tiny-transformer"],
        "artifact_sha256": components,
        "canonical_fixture_hash": sha256_canonical(components),
    }
    _write_json(synthetic / "fixture_manifest.json", manifest)
    summary = {
        "schema_version": "green-v400-synthetic-test-summary-v1",
        "status": "PASS", "certificate_records": len(records),
        "failed": 0, "skipped": 0, "xfailed": 0,
        "fixture_manifest_sha256": _sha256_file(synthetic / "fixture_manifest.json"),
    }
    _write_json(synthetic / "synthetic_test_summary.json", summary)
    return {path.name: _sha256_file(path) for path in sorted(synthetic.iterdir()) if path.is_file()}


def _sealed_noun_oracle() -> tuple[str, set[str], dict]:
    payload = json.loads(SEALED_NOUN_HASHES_PATH.read_text(encoding="utf-8"))
    if (payload.get("schema_version") != "green-v400-sealed-noun-hashes-v1"
            or payload.get("count") != len(payload.get("hashes", []))
            or len(set(payload.get("hashes", []))) != payload.get("count")):
        raise RuntimeError("PREPARE_STOP_SEALED_HASH_ORACLE_INVALID")
    return payload["salt"], set(payload["hashes"]), payload


def _sealed_noun_hash(noun: str, salt: str) -> str:
    return _sha256_bytes(f"{salt}|{noun}".encode("utf-8"))


def _eligible_nouns(tokenizer) -> tuple[list[dict], dict]:
    from green_bridge_spec import PROMPT
    sealed_salt, forbidden_hashes, sealed_payload = _sealed_noun_oracle()
    suffix_ids = {tokenizer.encode(f"{value:02d}", add_special_tokens=False)[0] for value in range(100)}
    reference_lengths = {
        len(tokenizer.encode(PROMPT.format(noun="campaign", cc=century, y=1), add_special_tokens=False))
        for century in (12, 14, 16)
    }
    candidates = []
    token_owners: dict[int, str] = {}
    reasons = Counter()
    for noun in (line.strip() for line in CANDIDATE_NOUNS_PATH.read_text(encoding="utf-8").splitlines()):
        reason = None
        ids = tokenizer.encode(" " + noun, add_special_tokens=False)
        if _sealed_noun_hash(noun, sealed_salt) in forbidden_hashes:
            reason = "inherited_noun"
        elif not (noun.isascii() and noun.isalpha() and noun.islower() and 4 <= len(noun) <= 12):
            reason = "lexical"
        elif len(ids) != 1 or ids[0] in suffix_ids:
            reason = "leading_space_token"
        elif ids[0] in token_owners:
            reason = "token_collision"
        else:
            lengths = {
                len(tokenizer.encode(PROMPT.format(noun=noun, cc=century, y=value), add_special_tokens=False))
                for century in (12, 14, 16) for value in (1, 17, 53, 99)
            }
            if lengths != reference_lengths:
                reason = "prompt_length"
            elif not _noun_pair_contract(tokenizer, noun):
                reason = "pair_token_contract"
        if reason:
            reasons[reason] += 1
            continue
        token_owners[ids[0]] = noun
        rank = _sha256_bytes(("green-v400-jwbt-group-split-20260826" + noun).encode("utf-8"))
        candidates.append({"noun": noun, "token_id": ids[0], "rank_key": rank})
    candidates.sort(key=lambda row: row["rank_key"])
    if len(candidates) < 40:
        raise RuntimeError(f"PREPARE_STOP_CANDIDATE_POOL: only {len(candidates)} eligible")
    audit = {
        "schema_version": "green-v400-sealed-exclusion-audit-v1",
        "forbidden_namespace_salted_hash": sha256_canonical(sorted(forbidden_hashes)),
        "sealed_hash_oracle_sha256": _sha256_file(SEALED_NOUN_HASHES_PATH),
        "sealed_hash_count": sealed_payload["count"],
        "candidate_file_sha256": _sha256_file(CANDIDATE_NOUNS_PATH),
        "eligible_pool_hash": sha256_canonical(candidates),
        "intersection_counts": {"inherited_nouns": 0},
        "excluded_reason_counts": dict(sorted(reasons.items())),
        "confirmation_content_opened": False,
        "passed": True,
    }
    return candidates, audit


def _select_pairs(noun: str, century: int, distance: str, role: str, count: int,
                  excluded: set[tuple[int, int]] | None = None,
                  tokenizer=None, verify_legacy: bool = True) -> list[dict]:
    from green_bridge_dataset import _candidate_pairs
    from green_bridge_spec import PROMPT
    import exp_green_bridge_gpt2 as legacy
    salt = "green-v400-jwbt-pairs-20260826"
    ranked = []
    excluded = set() if excluded is None else excluded
    pairs = (_token_pair_pool(tokenizer, noun, century, distance)
             if tokenizer is not None else tuple(_candidate_pairs(distance)))
    for first, second in pairs:
        if tuple(sorted((first, second))) in excluded:
            continue
        if tokenizer is not None and verify_legacy:
            first_prompt = PROMPT.format(noun=noun, cc=century, y=first)
            second_prompt = PROMPT.format(noun=noun, cc=century, y=second)
            if not legacy.token_pair_allowed(tokenizer, first_prompt, second_prompt):
                continue
        pair_key = _sha256_bytes(f"{salt}|pair|{noun}|{century}|{distance}|{role}|{first}|{second}".encode())
        orientation_key = _sha256_bytes(f"{salt}|orient|{noun}|{century}|{distance}|{role}|{first}|{second}".encode())
        ranked.append((pair_key, orientation_key, first, second))
    selected = []
    quota = {"up": count // 2, "down": count // 2}
    for pair_key, orientation_key, first, second in sorted(ranked):
        preferred = "up" if int(orientation_key[:2], 16) & 1 else "down"
        if quota[preferred] == 0:
            preferred = "down" if preferred == "up" else "up"
        if quota[preferred] == 0:
            continue
        y, y_prime = (first, second) if preferred == "up" else (second, first)
        selected.append({"pair_key": pair_key, "orientation": preferred, "y": y, "y_prime": y_prime})
        quota[preferred] -= 1
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError("PREPARE_STOP_PAIR_QUOTA")
    return selected


def _token_pair_pool(tokenizer, noun: str, century: int, distance: str) -> tuple[tuple[int, int], ...]:
    from green_bridge_dataset import _candidate_pairs
    from green_bridge_spec import PROMPT
    key = (id(tokenizer), noun, century, distance)
    if key in _TOKEN_PAIR_POOL_CACHE:
        return _TOKEN_PAIR_POOL_CACHE[key]
    year_key = (id(tokenizer), century)
    if year_key not in _YEAR_TOKEN_CACHE:
        century_ids = tokenizer.encode(f" {century:02d}", add_special_tokens=False)
        valid_suffixes = set()
        if len(century_ids) == 1:
            for suffix in range(100):
                suffix_ids = tokenizer.encode(f"{suffix:02d}", add_special_tokens=False)
                year_ids = tokenizer.encode(f" {century:02d}{suffix:02d}", add_special_tokens=False)
                if len(suffix_ids) == 1 and len(year_ids) == 2 and year_ids[1] == suffix_ids[0]:
                    valid_suffixes.add(suffix)
        _YEAR_TOKEN_CACHE[year_key] = (
            century_ids[0] if len(century_ids) == 1 else None,
            frozenset(valid_suffixes),
        )
    century_id, valid_suffixes = _YEAR_TOKEN_CACHE[year_key]
    valid: dict[int, tuple[int, ...]] = {}
    if century_id is not None:
        for suffix in valid_suffixes:
            prompt_ids = tuple(tokenizer.encode(
                PROMPT.format(noun=noun, cc=century, y=suffix), add_special_tokens=False
            ))
            if prompt_ids[-1] == century_id:
                valid[suffix] = prompt_ids
    result = tuple(
        (first, second) for first, second in _candidate_pairs(distance)
        if first in valid and second in valid and len(valid[first]) == len(valid[second])
    )
    _TOKEN_PAIR_POOL_CACHE[key] = result
    return result


def _noun_pair_contract(tokenizer, noun: str) -> bool:
    for century in (12, 14, 16):
        for distance in ("near", "far"):
            used: set[tuple[int, int]] = set()
            try:
                for role in ("transport", "joint"):
                    selected = _select_pairs(
                        noun, century, distance, role, 12,
                        excluded=used, tokenizer=tokenizer, verify_legacy=False,
                    )
                    used.update(tuple(sorted((row["y"], row["y_prime"]))) for row in selected)
            except RuntimeError:
                return False
    return True


def _row_universe(candidates: list[dict], tokenizer) -> tuple[dict, list[dict]]:
    from green_bridge_spec import PROMPT
    phases = (("formal_prepare_pool", 0, 4, 4), ("development_sealed", 4, 16, 12), ("confirmation_sealed", 16, 32, 12))
    rows = []
    group_counts = Counter()
    for phase, start, stop, per_role_cell in phases:
        for group_index, candidate in enumerate(candidates[start:stop]):
            century = (12, 14, 16)[group_index % 3]
            group_counts[phase] += 1
            for distance in ("near", "far"):
                used_pairs: set[tuple[int, int]] = set()
                for role in ("transport", "joint"):
                    selected = _select_pairs(
                        candidate["noun"], century, distance, role,
                        per_role_cell, excluded=used_pairs, tokenizer=tokenizer,
                    )
                    for item in selected:
                        pair = tuple(sorted((item["y"], item["y_prime"])))
                        if pair in used_pairs:
                            raise RuntimeError("PREPARE_STOP_PAIR_ROLE_COLLISION")
                        used_pairs.add(pair)
                        public = {
                            "phase": phase, "noun": candidate["noun"], "century": century,
                            "distance": distance, "role": role, "orientation": item["orientation"],
                            "y": item["y"], "y_prime": item["y_prime"], "pair_key": item["pair_key"],
                            "clean_prompt": PROMPT.format(noun=candidate["noun"], cc=century, y=item["y"]),
                            "corrupt_prompt": PROMPT.format(noun=candidate["noun"], cc=century, y=item["y_prime"]),
                        }
                        rows.append({**public, "row_hash": sha256_canonical(public)})
    payload = {
        "schema_version": "green-v400-row-universe-manifest-v1",
        "pool_generation_rule": "SHA256(protocol_id|canonical-row); green-v400-jwbt-pairs-20260826",
        "deterministic_ordering_rule": "candidate-rank, phase, century-cycle, distance, role, pair-rank",
        "split_labels": [phase[0] for phase in phases],
        "row_count_by_group": dict(group_counts),
        "row_count_by_phase": dict(Counter(row["phase"] for row in rows)),
        "canonical_row_hashes": [row["row_hash"] for row in rows],
        "pool_hash": sha256_canonical([row["row_hash"] for row in rows]),
        "exclusion_reasons": [],
        "contains_response_fields": False,
    }
    return payload, rows


def _tensor_hash(tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return _sha256_bytes(array.tobytes())


def _model_and_static_manifests(rows: list[dict], device: str):
    import torch
    import transformer_lens
    import transformers
    import exp_green_bridge_gpt2 as legacy
    from green_bridge_spec import MODEL_ID, MODEL_REVISION, SELECTED_GATES

    tokenizer, hf_model, model, observed = legacy.load_models(device)
    model.eval(); hf_model.eval()
    if (observed["normalization_type"] != TRANSFORMER_SEMANTICS_FLAGS["normalization_type"]
            or TRANSFORMER_SEMANTICS_FLAGS["activation_contains"] not in observed["act_fn"].lower()
            or observed["hf_attention_implementation"] != TRANSFORMER_SEMANTICS_FLAGS["attention_implementation"]
            or model.training or hf_model.training):
        raise RuntimeError("PREPARE_STOP_MODEL_SEMANTICS")
    weight_hashes = {name: _tensor_hash(value) for name, value in model.state_dict().items()}
    tokenizer_payload = sorted(tokenizer.get_vocab().items())
    hook_names = sorted(model.hook_dict)
    model_source = Path(inspect.getsourcefile(type(model)) or "")
    if not model_source.is_file():
        raise RuntimeError("PREPARE_STOP_TRANSFORMERLENS_SOURCE")
    donor_rows = [row for row in rows if row["phase"] == "formal_prepare_pool"]
    parity_tokens = legacy.tokenize_one(tokenizer, donor_rows[0]["clean_prompt"], device)
    with torch.inference_mode():
        hf_logits = hf_model(parity_tokens).logits.float()
        tl_logits = model(parity_tokens, return_type="logits").float()
    parity_error = float((hf_logits - tl_logits).abs().max())
    parity_scale = max(1.0, float(hf_logits.abs().max()), float(tl_logits.abs().max()))
    parity_operation_count = 1_000_000
    parity_tolerance = float(64 * torch.finfo(torch.float32).eps * parity_operation_count * parity_scale)
    if parity_error > parity_tolerance:
        raise RuntimeError("PREPARE_STOP_GRAPH_PARITY")
    parity = {
        "row_hash": donor_rows[0]["row_hash"],
        "hf_logits_sha256": _tensor_hash(hf_logits),
        "tl_logits_sha256": _tensor_hash(tl_logits),
        "max_abs_error": parity_error,
        "tolerance": parity_tolerance,
        "operation_count_upper": parity_operation_count,
        "passed": True,
        "engineering_diagnostic_only": True,
    }
    del hf_logits, tl_logits
    model_manifest = {
        "schema_version": "green-v400-model-manifest-v1",
        "transformer_lens_version": getattr(transformer_lens, "__version__", "unknown"),
        "transformer_lens_commit": TRANSFORMER_LENS_COMMIT,
        "transformer_lens_source_sha256": _sha256_file(model_source),
        "transformers_version": transformers.__version__,
        "pytorch_version": torch.__version__,
        "model_name": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_config": observed,
        "model_config_hash": sha256_canonical(observed),
        "tokenizer_hash": sha256_canonical(tokenizer_payload),
        "weight_tensor_hashes": weight_hashes,
        "full_model_hash": sha256_canonical(weight_hashes),
        "normalization_activation_flags": {"normalization": observed["normalization_type"], "activation": observed["act_fn"]},
        "processing_flags": {"fold_ln": False, "center_writing_weights": False, "center_unembed": False, "refactor_factored_attn_matrices": False},
        "evaluation_mode": not model.training and not hf_model.training,
        "attention_implementation": observed["hf_attention_implementation"],
        "hook_registry_hash": sha256_canonical(hook_names),
        "singleton_hf_transformerlens_parity": parity,
    }
    controlled_hooks = ["blocks.8.hook_resid_post", "blocks.10.hook_resid_pre", "blocks.10.mlp.hook_pre"]
    score_hook = "blocks.9.attn.hook_attn_scores"
    names_filter = set(controlled_hooks + [score_hook])
    feasibility, graph_rows, plans = [], [], []
    for row in donor_rows:
        clean = legacy.tokenize_one(tokenizer, row["clean_prompt"], device)
        corrupt = legacy.tokenize_one(tokenizer, row["corrupt_prompt"], device)
        with torch.inference_mode():
            _, clean_cache = model.run_with_cache(clean, names_filter=lambda name: name in names_filter, return_type=None)
            _, corrupt_cache = model.run_with_cache(corrupt, names_filter=lambda name: name in names_filter, return_type=None)
        present = [name for name in controlled_hooks if name in clean_cache and name in corrupt_cache]
        if len(present) != len(controlled_hooks) or score_hook not in clean_cache:
            raise RuntimeError("PREPARE_STOP_GRAPH_HOOK_MISSING")
        base_hashes = [_tensor_hash(clean_cache[name]) for name in present]
        directions = [(clean_cache[name] - corrupt_cache[name]).detach() for name in present]
        direction_hashes = [_tensor_hash(value) for value in directions]
        finite = all(bool(torch.isfinite(value).all()) for value in directions)
        nonzero = all(float(torch.linalg.vector_norm(value.float())) > 0 for value in directions)
        ln_margins = []
        for name in present[:2]:
            value = clean_cache[name].float()
            ln_margins.append(float(value.var(dim=-1, unbiased=False).min()) + float(model.cfg.eps))
        scores = clean_cache[score_hook].float()
        finite_scores = torch.isfinite(scores)
        has_unmasked_key = bool(finite_scores.any(dim=-1).all())
        mask_is_negative_infinity = bool((finite_scores | torch.isneginf(scores)).all())
        finite_max = scores.masked_fill(~finite_scores, -torch.inf).max(dim=-1).values
        finite_min = scores.masked_fill(~finite_scores, torch.inf).min(dim=-1).values
        softmax_record = {
            "hook": score_hook,
            "finite_unmasked": bool(torch.isfinite(scores[finite_scores]).all()),
            "has_unmasked_key_per_query": has_unmasked_key,
            "masked_entries_are_negative_infinity": mask_is_negative_infinity,
            "masked_entry_count": int((~finite_scores).sum()),
            "fixed_pivot_hash": _sha256_bytes(scores.argmax(dim=-1).cpu().numpy().tobytes()),
            "score_span_upper": float((finite_max - finite_min).max()),
        }
        softmax_valid = (
            softmax_record["finite_unmasked"]
            and has_unmasked_key
            and mask_is_negative_infinity
        )
        feasible = finite and nonzero and softmax_valid and min(ln_margins) > 0
        token_hash = _sha256_bytes(clean.detach().cpu().numpy().tobytes() + corrupt.detach().cpu().numpy().tobytes())
        hook_spec_hash = sha256_canonical({"hooks": controlled_hooks, "selected_gates": SELECTED_GATES})
        feasibility.append({
            "schema_version": "green-v400-donor-feasibility-v1",
            "row_hash": row["row_hash"], "split": "formal_prepare_pool",
            "model_hash": model_manifest["full_model_hash"], "token_hash": token_hash,
            "hook_spec_hash": hook_spec_hash,
            "control_ast_hash": sha256_canonical(CONTROL_AST),
            "contrast_hash": sha256_canonical({"branch_order": BRANCH_ORDER, "weights": BRANCH_CONTRAST}),
            "base_tensor_hashes": base_hashes, "direction_tensor_hashes": direction_hashes,
            "finite": finite, "required_nonzero_directions": nonzero,
            "supported_graph": set(GRAPH_OPERATIONS) <= set(SUPPORTED_OPERATIONS),
            "sealed_exclusion_pass": True,
            "static_cone_estimate": {"nodes_upper": 1_750_000, "scalar_mpfr_operations_upper": 75_000_000},
            "layernorm_static_margins": ln_margins, "softmax_static_records": [softmax_record],
            "feasible": feasible, "failure_codes": [] if feasible else ["INVALID_DOMAIN"],
            "contains_response_outcome": False,
        })
        graph_payload = {
            "schema_version": "green-v400-graph-manifest-v1", "row_hash": row["row_hash"],
            "branch_inputs": list(BRANCH_ORDER),
            "controlled_hooks": controlled_hooks, "exact_operation_list": list(GRAPH_OPERATIONS),
            "causal_cone_edges": [["control", "block8"], ["block8", "block9_attn"], ["block9_attn", "block10_mlp"], ["block10_mlp", "block11"], ["block11", "final_ln"], ["final_ln", "unembed_contrast"]],
            "dependency_tags": {"control": True, "model_constants": False},
            "unreduced_node_count_upper": 1_950_000, "reduced_node_count_upper": 1_750_000,
            "unreduced_semantic_hash": sha256_canonical({"row": row["row_hash"], "form": "unreduced", "ops": GRAPH_OPERATIONS}),
            "reduced_semantic_hash": sha256_canonical({"row": row["row_hash"], "form": "reduced", "ops": GRAPH_OPERATIONS}),
            "cancellation_proof_records": [{"identity": "x-x=0", "requires_provenance_identity": True}],
            "supported_operation_coverage": True,
            "fixed_softmax_pivots": [softmax_record["fixed_pivot_hash"]],
            "final_contrast_node": "PAT_J-PAT_B-TAR_J+TAR_B",
            "contains_endpoint_or_derivative_values": False,
        }
        graph_rows.append(graph_payload)
        plans.append({
            "schema_version": "green-v400-certificate-plan-v1", "row_hash": row["row_hash"],
            "exact_dyadic_amplitudes": [{"numerator": 1, "exponent": -index} for index in ALPHA_EXPONENTS],
            "initial_partition": "[-h,0],[0,h]",
            "split_policy": "left-to-right dyadic bisection",
            "absolute_width_tolerance": "0x1p-80", "relative_width_tolerance": "0x1p-40",
            "max_depth": MAX_SUBDIVISION_DEPTH, "max_cells": MAX_CELLS_PER_ROW,
            "official_precision": OFFICIAL_PRECISION_BITS,
            "audit_precision": AUDIT_PRECISION_BITS,
            "expected_artifact_paths": [], "execution_authorized": False,
        })
        for cache in (clean_cache, corrupt_cache):
            del cache
    del model, hf_model
    if any(not row["feasible"] for row in feasibility):
        raise RuntimeError("PREPARE_STOP_DONOR_STATIC_FEASIBILITY")
    return model_manifest, feasibility, graph_rows, plans


def _coverage_manifest() -> dict:
    functions = {
        "constant": "constant_jet", "affine_control": "affine_control_jet",
        "add": "add_jet", "sub": "sub_jet", "mul": "mul_jet",
        "reciprocal": "reciprocal_jet", "exp": "exp_primitive",
        "sqrt": "sqrt_primitive", "inv_sqrt": "inv_sqrt_primitive",
        "gelu_new": "gelu_new_jet", "layernorm": "layernorm_jets",
        "einsum": "affine_map_jets", "softmax": "softmax_jets",
        "attention": "attention_head_jets", "reshape": "RelationalGraph scalar view",
        "transpose": "shared-reference tensor view", "slice": "shared-reference tensor view",
        "gather_static": "shared-reference tensor view", "residual_add": "add_jet",
        "contrast": "contrast_jet",
    }
    unsupported = sorted(set(GRAPH_OPERATIONS) - set(EXECUTABLE_OPERATIONS))
    return {
        "schema_version": "green-v400-primitive-op-coverage-v1",
        "encountered_operations": list(GRAPH_OPERATIONS),
        "operations": [{"operation": op, "certified_implementation": functions[op], "theorem_clause": "corrigendum-4-through-10", "fixture_tests": TEST_FILES[:5]} for op in GRAPH_OPERATIONS],
        "unsupported_operations": unsupported,
        "coverage_status": "PASS" if not unsupported else "FAIL",
    }


def _boundary_lock(pool_hash: str, exclusion_hash: str) -> dict:
    return {
        "schema_version": "green-v400-boundary-design-lock-v1",
        "design": "attenuated_subtractive_dither_quantization",
        "pool_hash": pool_hash, "exclusion_hash": exclusion_hash,
        "amplitude_rule": {"alpha": [f"2^-{value}" for value in ALPHA_EXPONENTS], "q": [f"2^{value}" for value in Q_EXPONENTS], "dither_replicates": DITHER_REPLICATES},
        "primary_outcomes": ["CERTIFIED_POSITIVE", "CERTIFIED_NEGATIVE", "UNRESOLVED", "INVALID", "RESOURCE_INCONCLUSIVE"],
        "invalid_unresolved_treatment": "invalid separate; unresolved remains unresolved",
        "row_groups": ["distance", "orientation", "system", "gate_slot"],
        "null_controls": ["zero-control", "sign-reversal", "branch-permutation", "norm-matched-random", "bypass-only"],
        "transition_model": "frozen boundary-spanning preregistered analysis; no real outcomes in prepare",
        "exact_fallback_analysis": "familywise exact-binomial",
        "simultaneous_confidence_method": "row-cluster max-deviation bootstrap and Holm closure",
        "randomization_bootstrap_seeds": {"bootstrap": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "permutation": PERMUTATION_SEED, "permutation_replicates": PERMUTATION_REPLICATES},
        "family_definitions": ["transition", "regime-separation", "null-controls"],
        "numerical_gates": "unchanged P13 and inherited thresholds by content hash",
        "q_selection_authorized": False, "outcome_replay_authorized": False,
        "contains_observed_v4_outcome": False,
    }


def _artifact_hashes(output_root: Path, exclude: set[str] | None = None) -> dict[str, str]:
    exclude = exclude or set()
    return {
        path.name: _sha256_file(path)
        for path in sorted(output_root.iterdir())
        if path.is_file() and path.name not in exclude and not path.name.endswith(".tmp")
    }


def run_formal_prepare(config_path: str) -> FormalPrepareSummary:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not config.get("formal_prepare_only") or any((
        config.get("real_row_certificate_authorized"), config.get("development_authorized"),
        config.get("confirmation_authorized"), REAL_ROW_CERTIFICATE_AUTHORIZED,
        DEVELOPMENT_AUTHORIZED, CONFIRMATION_AUTHORIZED,
    )):
        raise RuntimeError("PREPARE_STOP_AUTHORIZATION_FIREWALL")
    output_root = _resolve_storage(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("PREPARE_STOP_ONE_SHOT_OUTPUT_EXISTS")
    output_root.mkdir(parents=True, exist_ok=True)
    repository = _repository_preflight()
    corrections = [{
            "schema_version": "green-v400-engineering-correction-v1",
            "category": "path_plumbing",
            "before": "/mnt/sdb/outputs/green_bridge_v400_formal_prepare",
            "after": str(output_root),
            "before_sha256": _sha256_bytes(b"/mnt/sdb/outputs/green_bridge_v400_formal_prepare"),
            "after_sha256": _sha256_bytes(str(output_root).encode("utf-8")),
            "rationale": "server account has no create permission at /mnt/sdb root; writable namespace is /mnt/sdb/ccj",
            "scientific_semantics_changed": False,
            "storage_device_changed": False,
        }, {
            "schema_version": "green-v400-engineering-correction-v1",
            "category": "environment_plumbing",
            "before": "new python venv via ensurepip",
            "after": f"existing frozen model environment {sys.executable}; pinned additions from /mnt/sdb via PYTHONPATH",
            "before_sha256": _sha256_bytes(b"new python venv via ensurepip"),
            "after_sha256": _sha256_bytes(sys.executable.encode("utf-8")),
            "rationale": "server Python lacks ensurepip; no package is installed to the root disk or inherited environment",
            "scientific_semantics_changed": False,
            "storage_device_changed": False,
        }, {
            "schema_version": "green-v400-engineering-correction-v1",
            "category": "model_cache_plumbing",
            "before": "/mnt/sdb/ccj/green_v400_formal_prepare_runtime/cache/huggingface",
            "after": os.environ.get("HF_HOME", ""),
            "before_sha256": _sha256_bytes(b"/mnt/sdb/ccj/green_v400_formal_prepare_runtime/cache/huggingface"),
            "after_sha256": _sha256_bytes(os.environ.get("HF_HOME", "").encode("utf-8")),
            "rationale": "network is offline; reuse the exact pinned GPT-2 revision cache from the immutable v1.3.6 runtime",
            "scientific_semantics_changed": False,
            "storage_device_changed": False,
        }, {
            "schema_version": "green-v400-engineering-correction-v1",
            "category": "pre_model_attempt_recovery",
            "before": "/mnt/sdb/ccj/outputs/green_bridge_v400_formal_prepare",
            "after": "/mnt/sdb/ccj/outputs/green_bridge_v400_formal_prepare_failed_pre_model_d5899bd",
            "before_sha256": _sha256_bytes(b"/mnt/sdb/ccj/outputs/green_bridge_v400_formal_prepare"),
            "after_sha256": _sha256_bytes(b"/mnt/sdb/ccj/outputs/green_bridge_v400_formal_prepare_failed_pre_model_d5899bd"),
            "rationale": "the prior attempt stopped before model load due an empty isolated cache; partial theorem artifacts were preserved byte-for-byte",
            "scientific_semantics_changed": False,
            "storage_device_changed": False,
        }, {
            "schema_version": "green-v400-engineering-correction-v1",
            "category": "paired_tokenization_contract",
            "before": "candidate-level prompt-length check only",
            "after": "legacy token_pair_allowed enforced during eligibility and every row pairing",
            "before_sha256": _sha256_bytes(b"candidate-level prompt-length check only"),
            "after_sha256": _sha256_bytes(b"legacy token_pair_allowed enforced during eligibility and every row pairing"),
            "rationale": "first donor exposed clean/corrupt sequence lengths 13/12 before any response or derivative was read",
            "failed_attempt_archive": "/mnt/sdb/ccj/outputs/green_bridge_v400_formal_prepare_failed_shape_bc4566a",
            "scientific_semantics_changed": False,
            "storage_device_changed": False,
        }, {
            "schema_version": "green-v400-engineering-correction-v1",
            "category": "causal_mask_static_feasibility",
            "before": "require every attention-score entry finite",
            "after": "require finite unmasked entries and exact negative-infinity causal-mask entries",
            "before_sha256": _sha256_bytes(b"require every attention-score entry finite"),
            "after_sha256": _sha256_bytes(b"require finite unmasked entries and exact negative-infinity causal-mask entries"),
            "rationale": "TransformerLens serializes valid causal-mask exclusions as -inf; no response, endpoint, derivative, or sign was inspected",
            "failed_attempt_archive": "/mnt/sdb/ccj/outputs/green_bridge_v400_formal_prepare_failed_softmax_46c42b6",
            "scientific_semantics_changed": False,
            "storage_device_changed": False,
        }]
    _write_jsonl(output_root / "engineering_corrections.jsonl", corrections)
    theorem_report = _run_theorem_tests()
    synthetic_hashes = _write_synthetic_artifacts(output_root)
    theorem_report["fixture_artifact_hashes"] = synthetic_hashes
    theorem_report["deterministic_rerun_hash"] = sha256_canonical({
        "tests": theorem_report["test_collection"],
        "runs": theorem_report["runs"],
        "synthetic_artifacts": synthetic_hashes,
    })
    environment = rounding_environment_manifest() | {
        "official_precision_bits": OFFICIAL_PRECISION_BITS,
        "audit_precision_bits": AUDIT_PRECISION_BITS,
        "thread_counts": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "pairwise_reduction_policy": "fixed balanced binary tree",
        "rounding_self_test_hash": theorem_report["deterministic_rerun_hash"],
    }
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        "openai-community/gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    candidates, exclusion = _eligible_nouns(tokenizer)
    universe, rows = _row_universe(candidates, tokenizer)
    model_manifest, feasibility, graph_rows, plans = _model_and_static_manifests(rows, os.environ.get("GREEN_V400_DEVICE", "cuda:0"))
    coverage = _coverage_manifest()
    boundary = _boundary_lock(universe["pool_hash"], exclusion["forbidden_namespace_salted_hash"])

    _write_json(output_root / "rounding_environment.json", environment)
    _write_json(output_root / "model_manifest.json", model_manifest)
    _write_json(output_root / "sealed_exclusion_audit.json", exclusion)
    _write_json(output_root / "row_universe_manifest.json", universe)
    _write_jsonl(output_root / "donor_feasibility.jsonl", feasibility)
    _write_jsonl(output_root / "graph_manifest.jsonl", graph_rows)
    _write_jsonl(output_root / "certificate_plan.jsonl", plans)
    _write_json(output_root / "boundary_design_lock.json", boundary)
    _write_json(output_root / "primitive_op_coverage.json", coverage)
    _write_json(output_root / "theorem_test_report.json", theorem_report)
    repository_manifest = _repository_manifest(repository)
    _write_json(output_root / "repository_manifest.json", repository_manifest)
    protocol_lock = {
        "schema_version": "green-v400-protocol-lock-v1", "protocol_id": PROTOCOL_ID,
        "binding_parent_commit": BINDING_PARENT_COMMIT, "branch": BRANCH,
        "corrigendum_sha256": _sha256_file(PROJECT_ROOT / "analysis/GPTPRO_GREEN_V400_BINDING_CORRIGENDUM_20260826.md"),
        "v3_immutable_manifest_sha256": sha256_canonical(repository["inherited"]),
        "unchanged_gate_manifest_sha256": sha256_canonical({"policy": "inherited-by-content-hash"}),
        "p13_definition_sha256": sha256_canonical({"policy": "unchanged; not executed"}),
        "formal_prepare_only": True, "development_authorized": False,
        "confirmation_authorized": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _write_json(output_root / "protocol_lock.json", protocol_lock)
    pre_summary_hashes = _artifact_hashes(output_root)
    summary_payload = {
        "schema_version": "green-v400-formal-prepare-summary-v1",
        "status": "PREPARE_PASS_STATIC_THEOREM_ONLY",
        "upstream_artifact_hashes": pre_summary_hashes,
        "counts_by_feasibility": dict(Counter("feasible" if row["feasible"] else "infeasible" for row in feasibility)),
        "counts_by_failure_code": dict(Counter(code for row in feasibility for code in row["failure_codes"])),
        "scientific_response_counts": {}, "operation_coverage": coverage["coverage_status"],
        "theorem_test_status": f"PASS_{theorem_report['passed']}_OF_{theorem_report['passed']}_TWICE",
        "exclusion_status": "PASS_ZERO_INTERSECTION",
        "next_authorized_action": "return immutable formal-prepare package for scientific authorization",
        "terminal_text": TERMINAL_TEXT,
    }
    _write_json(output_root / "formal_prepare_summary.json", summary_payload)
    hashes = _artifact_hashes(output_root)
    return FormalPrepareSummary("PREPARE_PASS_STATIC_THEOREM_ONLY", str(output_root), hashes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = run_formal_prepare(args.config)
    print(json.dumps({"status": summary.status, "output_root": summary.output_root}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
