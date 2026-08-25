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
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "green_bridge_v200"
SCHEMA_VERSION = "green-bridge-v2.0.0"
PROTOCOL_ID = "structural-envelope-matched-bypass-setid-v2.0.0"
PARENT_PROTOCOL_ID = "structural-envelope-matched-bypass-v1.3.6"
DECISION_ID = "GPTPRO-GREEN-V136-TERMINAL-SETID-v1-20260825"
CORRIGENDUM_ID = "GPTPRO-GREEN-V200-CORRIGENDUM-v1-20260825"
AMENDMENT_ID = CORRIGENDUM_ID
NUMERICAL_ERROR_CONTRACT = "dual-route-ad-certified-fine-richardson-v1"
THEORY_BASE_COMMIT = "126556f"
GATE04_AMENDMENT_ID = "GPTPRO-GREEN-GATE04-v2-20260805"
GATE08_AMENDMENT_ID = "GPTPRO-GREEN-GATE08-v2-20260805"
HF_ATTN_IMPLEMENTATION = "eager"
GATE04_LEGACY_PAIR_SLICE = (0, 16)
GATE04_HOLDOUT_PAIR_SLICE = (16, 32)
LEGACY_SALT = "idle1-gt-bridge-20260805"
SALT = LEGACY_SALT
MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TRANSFORMER_LENS_COMMIT = "4a4dc26c750475b29e6f54b362c2aab988702c9c"
PROTOCOL_RUN_ID = "green-bridge-v2.0.0-one-shot"
ATTEMPT_INDEX = 1
RETRY_ALLOWED = False
TAIL_DERIVATIVE_REFERENCE_NORM_FLOOR = 1.0e-5
TAIL_EQUIVALENCE_OUTPUT_DIM = 100
TAIL_FIXED_BATCH_SIZE = 1
PREDECESSOR_RUN = {
    "schema_version": "green-bridge-v1.3.6",
    "protocol_id": "structural-envelope-matched-bypass-v1.3.6",
    "protocol_run_id": "green-bridge-v1.3.6-one-shot",
    "attempt_index": 1,
    "retry_allowed": False,
    "verdict": "STOP_ORAL",
    "first_failed_gate": "12_DEVELOPMENT_SURVIVAL",
    "scientific_spec_sha256": "60ca5e9e221064f288a1993ee3cbf42e99330bbf6f9008946a25556438cbc3d3",
    "frozen_spec_sha256": "cb771c59e91b4fc553ef73a1c7a116ec0ee55f499ce46a2f91e4c600cd8bd41d",
    "confirmation_started": False,
    "artifact_sha256": {
        "dev_tensor_scores.parquet": "660788dde8bc5df1d057db31b4dc1065b222ac7777efc0e4c6220e09f1ed81ff",
        "dev_energy_targets.parquet": "23a99b6998ec2c51184ae26b8f86a7656247ff2091e251752c1fccd06295e593",
        "dev_cells.json": "1294a76d6d79c81f240c20c4257aa6b0fe76457d46b30cfc5d5699e27759ae1f",
        "dev_result.json": "2e15531d62bd5cc1162980fdaa2643a7300b362eb6b11ff5b94bb3d623c37277",
        "development_multigpu_merge.json": "31dbc71fbeaa40f313be6078a627082050aa5a338e132d3a6ed7343869eaad7a",
    },
}

PROMPT = "<|endoftext|> The {noun} lasted from the year {cc:02d}{y:02d} to the year {cc:02d}"
EVALUATION_NOUNS = (
    "campaign", "dynasty", "reign", "siege", "treaty", "warfare",
    "expedition", "kingdom",
)
EVALUATION_CENTURIES = (12, 14, 16)
LEGACY_DONOR_NOUNS = (
    "invasion", "insurgency", "rivalry", "hostility", "raids", "sanctions",
    "domination", "confrontation", "pilgrimage", "journey", "voyage",
    "operation", "outbreak", "reforms", "relationship", "modernization",
)
LEGACY_DONOR_CENTURIES = (11, 13, 15, 17)
DONOR_NOUNS = LEGACY_DONOR_NOUNS
DONOR_CENTURIES = LEGACY_DONOR_CENTURIES
PROBE_FRAME_DIM = 5
COMMON_FRAME_DIM = 4
ALL_GATE_FRAME_DIM = 14
FIRST_ORDER_RESIDUAL_DIRECTIONS = 250
RESIDUAL_RADIUS_MULTIPLIER = 0.20
GATE_RADIUS = 0.20
HALF_RADIUS_MULTIPLIER = 0.50
QUARTER_RADIUS_MULTIPLIER = 0.25
FACTORIZATION_COMPATIBILITY_RATIO_MAX = 1.0
WHITEBOX_COMPATIBILITY_RATIO_MAX = 1.0
WHITEBOX_FACTORIZATION_RATIO_MAX = 1.0
HISTORICAL_DYADIC_BALL_OVERLAP_RATIO_MAX = 1.0
WHITEBOX_COORDINATE_ABS_ERROR_MAX = 1.0e-10
FLOAT64_UNIT_ROUNDOFF = 2.0 ** -53
AD_ROUTE_OPERATION_BUDGET = 65_536
AD_ROUTE_GAMMA = (
    AD_ROUTE_OPERATION_BUDGET * FLOAT64_UNIT_ROUNDOFF
    / (1.0 - AD_ROUTE_OPERATION_BUDGET * FLOAT64_UNIT_ROUNDOFF)
)
AD_AUDIT_STRATA = 40
AD_AUDIT_PERMITTED_MISSES = 0
STRUCTURAL_FRAME_ORTHOGONAL_MAX = 5e-13
STRUCTURAL_ATOM_RESIDUAL_MAX = 1e-12
STRUCTURAL_GRADIENT_RESIDUAL_MAX = 1e-10
STRUCTURAL_GRADIENT_AUTOGRAD_MAX_ABS = 1e-10
STRUCTURAL_GRADIENT_AUTOGRAD_RELATIVE = 1e-9
SHIFT_GRADIENT_NORMALIZED_MAX = 1e-12
FIRST_ORDER_COEFFICIENT_SEED = 8998478401382166109
FIRST_ORDER_COEFFICIENT_SHA256 = (
    "b39a9a0bdda54bf63d1496f690bd4c89"
    "c6fa618ba7beb152364cb9f2b3f18a1a"
)

# Frozen solely to reproduce the archived v1.2 STOP.  Nothing below this
# namespace is an active v1.3 scientific choice.
HISTORICAL_V12_BASIS_SPEC: dict[str, Any] = {
    "salt": "idle1-gt-bridge-basis-v2-20260805",
    "donor_nouns": (
        "rebellion", "revolution", "occupation", "blockade", "crusade",
        "migration", "settlement", "construction", "administration", "regime",
        "competition", "partnership", "transition", "expansion", "uprising",
        "conflict",
    ),
    "donor_centuries": (11, 13, 15, 17),
    "donor_selection_order": (
        ("near", "basis_fit", 2, 2),
        ("far", "basis_fit", 2, 2),
        ("near", "basis_holdout", 1, 1),
        ("far", "basis_holdout", 1, 1),
        ("near", "radius_v2", 2, 2),
        ("far", "radius_v2", 2, 2),
    ),
    "residual_rank": 5,
    "fit_pairs": 512,
    "holdout_pairs": 256,
    "radius_pairs": 512,
    "bootstrap_replicates": 256,
    "bootstrap_quantile": 0.95,
    "rank6_fallback": False,
}
DISTANCE_BINS = {"near": (8, 16), "far": (40, 56)}
SUFFIX_MIN = 5
SUFFIX_MAX = 94
SELECTED_GATES = (2326, 1138, 2287, 606, 2848, 2305, 46, 2659, 946, 1616)
OUTPUT_SUFFIXES = tuple(range(100))

V200_RESPLIT_SALT = "green-v200-resplit-20260825"
V200_DEVELOPMENT_GROUPS = (
    ("dynasty", 16),
    ("dynasty", 12),
    ("reign", 14),
    ("warfare", 14),
)
V200_CONFIRMATION_GROUPS = (
    ("treaty", 12),
    ("warfare", 12),
    ("expedition", 14),
    ("kingdom", 12),
    ("treaty", 16),
    ("kingdom", 16),
    ("campaign", 14),
    ("siege", 16),
    ("reign", 12),
    ("siege", 14),
    ("campaign", 16),
    ("expedition", 16),
)
V200_SPLIT_SHA256 = (
    "0873915c966bef8f54b83d4151a9d7c75"
    "b577da5dfc17ee093b9f5c58a9590f7"
)

HISTORICAL_V136_THRESHOLDS = {
    "factorization_residual_max": 0.15,
    "whitebox_a_relative_max": 0.05,
    "whitebox_a_small_absolute_max": 1e-4,
    "tensor_cosine_min": 0.95,
    "tensor_symmetric_change_max": 0.25,
    "richardson_change_max": 0.25,
}


@dataclass(frozen=True)
class Dimensions:
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_mlp: int = 3072
    probe_frame_dim: int = PROBE_FRAME_DIM
    selected_gates: int = 10
    output_dimension: int = 100


@dataclass(frozen=True)
class Thresholds:
    hook_untouched_max: float = 1e-7
    hf_tl_raw_year_max_abs: float = 3.0e-4
    hf_tl_raw_year_pooled_rms: float = 7.5e-5
    hf_tl_centered_year_max_abs: float = 2.5e-4
    hf_tl_centered_year_pooled_rms: float = 6.0e-5
    hf_tl_margin_max_abs: float = 2.0e-4
    hf_tl_margin_rms: float = 5.0e-5
    hf_tl_resid_mid_max_abs: float = 1.0e-4
    hf_tl_resid_mid_pooled_rms: float = 2.0e-5
    hf_tl_selected_pre_max_abs: float = 5.0e-4
    hf_tl_selected_pre_pooled_rms: float = 1.0e-4
    hf_tl_selected_post_max_abs: float = 5.0e-4
    hf_tl_selected_post_pooled_rms: float = 1.0e-4
    no_op_max_abs: float = 2e-5
    tail_max_abs: float = 2e-5
    tail_derivative_relative: float = 1e-4
    center_rms: float = 2e-6
    center_max_abs: float = 2e-5
    curvature_rms_min: float = 5e-4
    curvature_snr_min: float = 20.0
    gate_response_rms_min: float = 5e-4
    gate_response_snr_min: float = 20.0
    factorization_compatibility_ratio_max: float = FACTORIZATION_COMPATIBILITY_RATIO_MAX
    whitebox_compatibility_ratio_max: float = WHITEBOX_COMPATIBILITY_RATIO_MAX
    whitebox_factorization_ratio_max: float = WHITEBOX_FACTORIZATION_RATIO_MAX
    dyadic_ball_overlap_ratio_max: float = HISTORICAL_DYADIC_BALL_OVERLAP_RATIO_MAX
    tensor_snr_min: float = 20.0
    bypass_disagreement_max: float = 0.15
    active_gates_min: int = 3
    certified_null_contribution_max: float = 0.005
    valid_items_per_cell_min: int = 6
    projected_chord_sigma_min: float = 0.10
    development_cells_min: int = 8
    confirmation_technical_min: int = 21
    confirmation_oral_min: int = 22
    total_cells_technical_min: int = 27
    cells_per_bin_min: int = 11
    conditioning_absolute: float = 0.10
    conditioning_dev_sd: float = 0.25
    development_snr_cells_min: int = 5
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
    "protocol_id": PROTOCOL_ID,
    "decision_id": DECISION_ID,
    "corrigendum_id": CORRIGENDUM_ID,
    "amendment_id": AMENDMENT_ID,
    "v200_split_sha256": V200_SPLIT_SHA256,
    "probe_frames": {
        "probe_frame_dim": PROBE_FRAME_DIM,
        "common_frame_dim": COMMON_FRAME_DIM,
        "all_gate_frame_dim": ALL_GATE_FRAME_DIM,
        "construction": "exact-layernorm-structural-envelope",
    },
    "radii": {
        "full": 1.0,
        "half": HALF_RADIUS_MULTIPLIER,
        "quarter": QUARTER_RADIUS_MULTIPLIER,
        "residual_multiplier": RESIDUAL_RADIUS_MULTIPLIER,
        "gate": GATE_RADIUS,
    },
    "bootstrap": {"replicates": 100_000, "seed": 20260805},
    "gate04_amendment": {
        "id": GATE04_AMENDMENT_ID,
        "hf_attention_implementation": HF_ATTN_IMPLEMENTATION,
        "legacy_pair_slice": GATE04_LEGACY_PAIR_SLICE,
        "holdout_pair_slice": GATE04_HOLDOUT_PAIR_SLICE,
        "prompts_per_pair": ("clean", "corrupt"),
        "prompt_count": 32,
        "batch_size": 1,
        "parameter_mapping_exact": True,
        "hf_tl_error_enters_epsilon_y": False,
    },
    "numerical_error_contract": NUMERICAL_ERROR_CONTRACT,
    "ad_route": {
        "unit_roundoff": FLOAT64_UNIT_ROUNDOFF,
        "operation_budget": AD_ROUTE_OPERATION_BUDGET,
        "gamma": AD_ROUTE_GAMMA,
    },
    "ad_operation_counts": {
        "prepare_certificates": 40,
        "development_certificates": 1280,
        "confirmation_certificates": 3840,
        "total_certificates": 5160,
        "routes_per_certificate": 2,
        "prepare_routes": 80,
        "development_routes": 2560,
        "confirmation_routes": 7680,
        "total_routes": 10320,
        "derivative_objects_per_route": 5,
        "prepare_top_level_calls": 400,
        "development_top_level_calls": 12800,
        "confirmation_top_level_calls": 38400,
        "total_top_level_calls": 51600,
    },
    "structural_envelope_amendment": {
        "id": AMENDMENT_ID,
        "basis_object": "ambient-rank-one-operator",
        "probe_completeness": "exact-layernorm-envelope",
        "first_order_residual_directions": FIRST_ORDER_RESIDUAL_DIRECTIONS,
        "first_order_coefficient_seed": FIRST_ORDER_COEFFICIENT_SEED,
        "first_order_coefficient_sha256": FIRST_ORDER_COEFFICIENT_SHA256,
        "attempt_index": 1,
        "retry_allowed": False,
    },
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
    """Durably write JSON through a same-filesystem temporary and rename."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    # POSIX directory fsync makes the rename durable. Windows does not permit
    # opening directories through ``os.open``; the file flush above remains
    # the strongest portable guarantee there.
    if os.name != "nt":
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def assert_scientific_override_free(arguments: dict[str, Any]) -> None:
    forbidden = {
        "model", "revision", "layer", "gates", "rank", "radius", "threshold",
        "corruption", "bootstrap_seed", "confirmation_retries",
    }
    present = forbidden.intersection(arguments)
    if present:
        raise ValueError(f"scientific command-line overrides are prohibited: {sorted(present)}")
