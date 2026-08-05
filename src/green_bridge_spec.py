"""Frozen constants and hashing utilities for the GPT-2 green-bridge run.

This file is the executable counterpart of
``analysis/GPTPRO_GREEN_BRIDGE_20260805.md``.  Scientific constants must not be
overridden from the command line.  Hardware-only batch sizes may be lowered on
out-of-memory, but sites, gates, splits, radii, thresholds, and analysis rules
are immutable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "green_bridge"
SCHEMA_VERSION = "green-bridge-v1"
THEORY_BASE_COMMIT = "126556f"
SALT = "idle1-gt-bridge-20260805"
MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TRANSFORMER_LENS_COMMIT = "4a4dc26c750475b29e6f54b362c2aab988702c9c"

PROMPT = "<|endoftext|> The {noun} lasted from the year {cc:02d}{y:02d} to the year {cc:02d}"
EVALUATION_NOUNS = (
    "campaign", "dynasty", "reign", "siege", "treaty", "warfare",
    "expedition", "kingdom",
)
EVALUATION_CENTURIES = (12, 14, 16)
DONOR_NOUNS = (
    "invasion", "insurgency", "rivalry", "hostility", "raids", "sanctions",
    "domination", "confrontation", "pilgrimage", "journey", "voyage",
    "operation", "outbreak", "reforms", "relationship", "modernization",
)
DONOR_CENTURIES = (11, 13, 15, 17)
DISTANCE_BINS = {"near": (8, 16), "far": (40, 56)}
SUFFIX_MIN = 5
SUFFIX_MAX = 94
SELECTED_GATES = (2326, 1138, 2287, 606, 2848, 2305, 46, 2659, 946, 1616)
OUTPUT_SUFFIXES = tuple(range(100))


@dataclass(frozen=True)
class Dimensions:
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_mlp: int = 3072
    residual_rank: int = 4
    selected_gates: int = 10
    output_dimension: int = 100


@dataclass(frozen=True)
class Thresholds:
    hook_untouched_max: float = 1e-7
    hf_tl_max_abs: float = 2e-5
    no_op_max_abs: float = 2e-5
    tail_max_abs: float = 2e-5
    tail_derivative_relative: float = 1e-4
    center_rms: float = 2e-6
    center_max_abs: float = 2e-5
    curvature_rms_min: float = 5e-4
    curvature_snr_min: float = 20.0
    gate_response_rms_min: float = 5e-4
    gate_response_snr_min: float = 20.0
    factorization_residual_max: float = 0.15
    whitebox_a_relative_max: float = 0.05
    whitebox_a_small_absolute_max: float = 1e-4
    tensor_cosine_min: float = 0.95
    tensor_symmetric_change_max: float = 0.25
    richardson_change_max: float = 0.25
    tensor_snr_min: float = 20.0
    bypass_disagreement_max: float = 0.15
    active_gates_min: int = 3
    valid_items_per_cell_min: int = 6
    projected_chord_sigma_min: float = 0.10
    development_cells_min: int = 15
    confirmation_technical_min: int = 28
    confirmation_oral_min: int = 29
    total_cells_technical_min: int = 40
    cells_per_bin_min: int = 14
    conditioning_absolute: float = 0.10
    conditioning_dev_sd: float = 0.25
    development_snr_cells_min: int = 10
    development_snr_min: float = 3.0
    development_stop_below: float = 0.05
    confirmation_open_gain_min: float = 0.10
    confirmation_relative_gain_min: float = 0.20
    confirmation_relative_lcb_min: float = 0.10
    confirmation_absolute_gain_min: float = 0.01
    per_bin_relative_gain_min: float = 0.10
    per_bin_absolute_gain_min: float = 0.005
    cancellation_size_min: int = 8
    cancellation_class_min: int = 3
    cancellation_bin_min: int = 3
    cancellation_main_effect_min: float = 0.05
    cancellation_target_threshold: float = 0.10
    cancellation_auroc_min: float = 0.80
    cancellation_auroc_lcb_min: float = 0.70
    half_radius_spearman_min: float = 0.90
    half_radius_change_max: float = 0.20


DIMENSIONS = Dimensions()
THRESHOLDS = Thresholds()


FROZEN_SPEC: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "theory_base_commit": THEORY_BASE_COMMIT,
    "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
    "transformer_lens_commit": TRANSFORMER_LENS_COMMIT,
    "prompt": PROMPT,
    "evaluation_nouns": EVALUATION_NOUNS,
    "evaluation_centuries": EVALUATION_CENTURIES,
    "donor_nouns": DONOR_NOUNS,
    "donor_centuries": DONOR_CENTURIES,
    "distance_bins": DISTANCE_BINS,
    "suffix_range": (SUFFIX_MIN, SUFFIX_MAX),
    "selected_gates": SELECTED_GATES,
    "dimensions": asdict(DIMENSIONS),
    "thresholds": asdict(THRESHOLDS),
    "sites": {
        "patch": "blocks.8.hook_mlp_out",
        "x": "blocks.10.hook_resid_mid",
        "z": "blocks.10.mlp.hook_pre",
        "gate": "blocks.10.mlp.hook_post",
        "target_bypass_subtraction": "blocks.10.hook_resid_post",
    },
    "radii": {"full": 1.0, "half": 0.5, "multiplier": 0.20},
    "bootstrap": {"replicates": 100_000, "seed": 20260805},
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_spec_hash() -> str:
    return sha256_text(canonical_json(FROZEN_SPEC))


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON through a same-directory temporary file then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def assert_scientific_override_free(arguments: dict[str, Any]) -> None:
    forbidden = {
        "model", "revision", "layer", "gates", "rank", "radius", "threshold",
        "corruption", "bootstrap_seed", "confirmation_retries",
    }
    present = forbidden.intersection(arguments)
    if present:
        raise ValueError(f"scientific command-line overrides are prohibited: {sorted(present)}")
