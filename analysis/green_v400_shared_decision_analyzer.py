"""Frozen outcome analyzer for the GREEN silent-failure challenge.

This module consumes already merged, firewalled row records.  It never loads a
model, constructs directions, or changes thresholds.  The primary comparison
matches every baseline to GREEN's accepted-row count without reading endpoint
values during selection and bootstraps whole prompt clusters.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ACCEPT_STATUS = "CERTIFIED_POSITIVE"
KNOWN_GREEN_STATUSES = {
    ACCEPT_STATUS,
    "CERTIFIED_NEGATIVE",
    "UNRESOLVED",
    "RESOURCE_INCONCLUSIVE",
    "INVALID",
}
PRIMARY_ENDPOINT_FIELD = "heldout_transport_symmetric_normalized_error"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_rows(rows: list[dict[str, Any]], methods: list[str]) -> None:
    if not rows:
        raise ValueError("analysis rows must be nonempty")
    row_ids = [row.get("row_id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
        raise ValueError("every analysis row requires row_id")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("analysis row identifiers must be unique")
    for row in rows:
        if not isinstance(row.get("prompt_row_id"), str) or not row["prompt_row_id"]:
            raise ValueError("every row requires prompt_row_id")
        if row.get("endpoint_status") != "VALID":
            raise ValueError("every method shares the same VALID endpoint population")
        endpoint = row.get(PRIMARY_ENDPOINT_FIELD)
        if not isinstance(endpoint, (int, float)) or not math.isfinite(endpoint) or endpoint < 0:
            raise ValueError("primary continuous endpoint must be finite and nonnegative")
        if row.get("green_status") not in KNOWN_GREEN_STATUSES:
            raise ValueError("unknown GREEN status")
        scores = row.get("baseline_risk_scores", {})
        for method in methods:
            score = scores.get(method)
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f"missing finite baseline risk score for {method}")


def primary_population(rows: list[dict[str, Any]], *, task: str) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        restoration = row.get("ordinary_restoration")
        if not isinstance(restoration, (int, float)) or not math.isfinite(restoration):
            raise ValueError("ordinary restoration must be finite")
        if restoration < 0.8:
            continue
        if task == "greater_than" and row.get("clean_task_valid") is not True:
            continue
        selected.append(row)
    if not selected:
        raise ValueError("primary high-restoration population is empty")
    return selected


def matched_acceptance_sets(
    rows: list[dict[str, Any]], methods: list[str]
) -> dict[str, set[str]]:
    green = {
        row["row_id"] for row in rows if row["green_status"] == ACCEPT_STATUS
    }
    accepted_count = len(green)
    result = {"GREEN": green}
    for method in methods:
        ordered = sorted(
            rows,
            key=lambda row: (float(row["baseline_risk_scores"][method]), row["row_id"]),
        )
        result[method] = {row["row_id"] for row in ordered[:accepted_count]}
    return result


def _accepted_risk(rows: list[dict[str, Any]], accepted: set[str]) -> float:
    if not accepted:
        return math.nan
    return float(np.mean([
        float(row[PRIMARY_ENDPOINT_FIELD]) for row in rows if row["row_id"] in accepted
    ]))


def _resample_prompt_clusters(
    rows: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_row_id"], []).append(row)
    prompts = sorted(by_prompt)
    sampled = rng.choice(prompts, size=len(prompts), replace=True)
    result = []
    for replicate, prompt in enumerate(sampled):
        for row in by_prompt[str(prompt)]:
            copied = dict(row)
            copied["row_id"] = f"{replicate}:{row['row_id']}"
            result.append(copied)
    return result


def analyze_primary_selective_risk(
    rows: list[dict[str, Any]],
    *,
    task: str,
    methods: list[str],
    bootstrap_replicates: int = 20000,
    bootstrap_seed: int = 40029001,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie between zero and one")
    validate_rows(rows, methods)
    population = primary_population(rows, task=task)
    accepted = matched_acceptance_sets(population, methods)
    coverage = len(accepted["GREEN"]) / len(population)
    green_risk = _accepted_risk(population, accepted["GREEN"])
    point = {
        method: green_risk - _accepted_risk(population, accepted[method])
        for method in methods
    }

    rng = np.random.Generator(np.random.PCG64DXSM(bootstrap_seed))
    samples = {method: [] for method in methods}
    for _ in range(bootstrap_replicates):
        resampled = _resample_prompt_clusters(population, rng)
        replicate_sets = matched_acceptance_sets(resampled, methods)
        replicate_green = _accepted_risk(resampled, replicate_sets["GREEN"])
        for method in methods:
            contrast = replicate_green - _accepted_risk(
                resampled, replicate_sets[method]
            )
            if math.isfinite(contrast):
                samples[method].append(contrast)

    alpha = 1.0 - confidence_level
    intervals = {}
    for method in methods:
        values = np.asarray(samples[method], dtype=np.float64)
        if values.size < max(100, bootstrap_replicates // 2):
            raise ValueError("too few finite cluster-bootstrap replicates")
        intervals[method] = {
            "point_contrast_green_minus_baseline": point[method],
            "marginal_lower": float(np.quantile(values, alpha / 2)),
            "marginal_upper": float(np.quantile(values, 1 - alpha / 2)),
        }

    # Single-step max deviation gives simultaneous percentile-style bounds.
    centered_max = []
    for index in range(min(len(samples[method]) for method in methods)):
        centered_max.append(max(
            abs(samples[method][index] - point[method]) for method in methods
        ))
    critical = float(np.quantile(np.asarray(centered_max), confidence_level))
    for method in methods:
        intervals[method]["simultaneous_lower"] = point[method] - critical
        intervals[method]["simultaneous_upper"] = point[method] + critical

    invalid_rate = sum(row["green_status"] == "INVALID" for row in population) / len(population)
    return {
        "schema_version": "green-v400-primary-selective-risk-result-v1",
        "contains_scientific_outcome": True,
        "task": task,
        "row_count": len(population),
        "prompt_cluster_count": len({row["prompt_row_id"] for row in population}),
        "green_accepted_count": len(accepted["GREEN"]),
        "green_coverage": coverage,
        "green_method_invalid_rate": invalid_rate,
        "green_accepted_risk": green_risk,
        "baseline_accepted_risks": {
            method: _accepted_risk(population, accepted[method]) for method in methods
        },
        "contrasts": intervals,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "cluster_unit": "prompt_row_id",
        "selection_used_endpoint_values": False,
        "analyzer_source_sha256": source_sha256(),
        "input_rows_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
    }
