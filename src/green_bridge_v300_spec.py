"""Immutable scientific constants for GREEN v3.0.0 prepare-only execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from green_bridge_spec import (
    ALL_GATE_FRAME_DIM,
    COMMON_FRAME_DIM,
    DIMENSIONS,
    DISTANCE_BINS,
    MODEL_ID,
    MODEL_REVISION,
    OUTPUT_SUFFIXES,
    PROBE_FRAME_DIM,
    PROMPT,
    SELECTED_GATES,
    SUFFIX_MAX,
    SUFFIX_MIN,
    TAIL_FIXED_BATCH_SIZE,
    TRANSFORMER_LENS_COMMIT,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "green-bridge-v3.0.0"
PROTOCOL_ID = "structural-envelope-matched-bypass-transport-v3.0.0"
PARENT_PROTOCOL_ID = "structural-envelope-matched-bypass-setid-v2.0.0"
DECISION_ID = "GPTPRO-GREEN-V21-POSTMORTEM-TRANSPORT-v1-20260825"
PROTOCOL_RUN_ID = "green-bridge-v3.0.0-one-shot"
ATTEMPT_INDEX = 1
RETRY_ALLOWED = False
RESUME_ALLOWED = False
PHASE_ALL_ALLOWED = False
AUTHORIZED_PHASES = ("prepare",)

POSTMORTEM_COMMIT = "ef09fce529553d5a3d236852a288cde02b88418a"
V200_EXECUTION_COMMIT = "e52e082296c33a10557636706e572147136fce34"
V200_VERDICT = "STOP_ORAL"
V200_FIRST_FAILED_GATE = "12_DEVELOPMENT_SURVIVAL"

V300_SPLIT_SALT = "green-v300-transport-noun-split-20260825"
V300_PAIR_SALT = "green-v300-transport-pairs-20260825"
V300_SPLIT_SHA256 = "509f791b614db58e0e7b47c1106364ef549c156e2c42a48a51e705a196da0bc7"
V300_TECHNICAL_CORRIGENDUM_ID = "CODEX-GREEN-V300-CANONICAL-PAYLOAD-v1-20260826"
# The external decision supplied these two identifiers without the bytes that
# produced them.  They remain immutable provenance, but cannot be used as
# reproducibility checks because SHA-256 does not encode its input.
V300_DECLARED_COEFFICIENT_HASH_ID = "1b5cc44b98b74ae7793957d68087a17d8ce9684ebee822b46a67492e4a7892e5"
V300_DECLARED_RADIUS_CANDIDATE_HASH_ID = "50251164fb42f9ecd97c7725a093ff15084b9f6662b364d3d52be1210c98feb9"
# Reproducible semantic hashes use UTF-8 canonical JSON, no trailing newline,
# and exact symbolic numbers.  The hash itself is deliberately excluded from
# the payload, avoiding a circular definition.
V300_COEFFICIENT_SHA256 = "71d1f91b7a7da68e1d73079e42b116e09cf3544b890f53aac1d58afae4bf4cfa"
V300_RADIUS_CANDIDATE_SHA256 = "370173c38e04bf741145faf09d5cffc826810d206c684b97de65c07d13303d6c"

DEVELOPMENT_NOUNS = ("kingdom", "reign", "siege")
CONFIRMATION_NOUNS = ("warfare", "campaign", "expedition", "treaty")
DEVELOPMENT_GROUPS = (
    ("kingdom", 12), ("kingdom", 16), ("reign", 12),
    ("siege", 14), ("siege", 16),
)
CONFIRMATION_GROUPS = (
    ("warfare", 12), ("campaign", 14), ("campaign", 16),
    ("expedition", 14), ("expedition", 16), ("treaty", 12),
    ("treaty", 16),
)
ROLES = ("transport", "joint")
RECORDS_PER_ROLE_PER_CELL = 8
ORIENTATIONS_PER_ROLE = {"down": 4, "up": 4}

HELMERT_COEFFICIENT_HASH_ID = V300_COEFFICIENT_SHA256
RADIUS_CANDIDATES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
RADIUS_CANDIDATE_SYMBOLS = ("1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64")
RADIUS_RELATIVE_FIDELITY_MAX = 0.10
RECOVERABLE_RELATIVE_WIDTH_MAX = 0.25
DIRECT_NONNULL_SNR_MIN = 4.0
JOINT_SET_SNR_MIN = 4.0
UNRESOLVED_MASS_RATIO_MAX = 0.25
HELDOUT_FRAME_DIRECTIONS = 4
HELDOUT_MIXED_DIRECTIONS = 4
HELDOUT_NULL_DIRECTIONS = 2
HELDOUT_DIRECTIONS_TOTAL = 10

PREPARE_GPU = 4
MAX_PEAK_GIB = 20.0
MAX_EIGHT_GPU_SECONDS = 24 * 60 * 60
EXPECTED_EXISTING_TESTS = 220
EXPECTED_NEW_TESTS = 52
EXPECTED_TOTAL_TESTS = 272


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def noun_rank(noun: str) -> str:
    return sha256_text(f"{V300_SPLIT_SALT}|{noun}")


NOUN_RANKS = {
    "kingdom": "2cb603de8d771d6e31da85fcac4a8c92710acaf3f2247cf4a78536ab71bdb421",
    "reign": "444eba1462ed40d15dc1c16c7a6c8546577790ce8dcd4982c2d246f2c29fbe7e",
    "siege": "47348e3ba5e30dca846c17bc4f49a279cf1e4008e9ed23c8bdca07f07802a312",
    "warfare": "9ecb254c9b50e23347e2ff87b974b0276cc0d68ef66309af56d188f25f09211b",
    "campaign": "a61597275b4c915e9c7817db66a23f6b606bc7dbcd90a4ce887f71c817bfc899",
    "expedition": "c321a47e8b081ed358bc3563699b7468de0d7b1e643ee885fa67eb7e9ea514b9",
    "treaty": "fbb0f0ca767e11356284380417c68e28eee578a4e8307418803ba458e4098edb",
}


@dataclass(frozen=True)
class V300Thresholds:
    recoverable_relative_width_max: float = RECOVERABLE_RELATIVE_WIDTH_MAX
    direct_nonnull_snr_min: float = DIRECT_NONNULL_SNR_MIN
    joint_set_snr_min: float = JOINT_SET_SNR_MIN
    unresolved_mass_ratio_max: float = UNRESOLVED_MASS_RATIO_MAX
    transport_records_per_cell_min: int = 6
    joint_records_per_cell_min: int = 6
    resolved_gate_system_fraction_min: float = 0.80
    recoverable_nonnull_fraction_min: float = 0.25
    development_surviving_cells_min: int = 8
    development_set_snr_cells_min: int = 6
    development_direct_median_max: float = 0.10
    development_direct_p90_max: float = 0.25
    development_joint_median_max: float = 0.15
    development_joint_p90_max: float = 0.30
    development_detectability_spearman_min: float = 0.50
    development_null_leakage_median_max: float = 0.05
    development_null_leakage_p95_max: float = 0.10
    development_best_baseline_gain_min: float = 0.20
    development_every_baseline_gain_min: float = 0.10
    coarse_fine_spearman_min: float = 0.90
    coarse_fine_symmetric_change_max: float = 0.20


THRESHOLDS = V300Thresholds()


FROZEN_SPEC = {
    "schema_version": SCHEMA_VERSION,
    "protocol_id": PROTOCOL_ID,
    "parent_protocol_id": PARENT_PROTOCOL_ID,
    "decision_id": DECISION_ID,
    "protocol_run_id": PROTOCOL_RUN_ID,
    "attempt_index": ATTEMPT_INDEX,
    "retry_allowed": RETRY_ALLOWED,
    "resume_allowed": RESUME_ALLOWED,
    "phase_all_allowed": PHASE_ALL_ALLOWED,
    "authorized_phases": AUTHORIZED_PHASES,
    "postmortem_commit": POSTMORTEM_COMMIT,
    "v200_execution_commit": V200_EXECUTION_COMMIT,
    "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
    "transformer_lens_commit": TRANSFORMER_LENS_COMMIT,
    "prompt": PROMPT,
    "suffix_range": (SUFFIX_MIN, SUFFIX_MAX),
    "output_suffixes": OUTPUT_SUFFIXES,
    "distance_bins": DISTANCE_BINS,
    "selected_gates": SELECTED_GATES,
    "dimensions": asdict(DIMENSIONS),
    "probe_frames": {
        "probe_frame_dim": PROBE_FRAME_DIM,
        "common_frame_dim": COMMON_FRAME_DIM,
        "all_gate_frame_dim": ALL_GATE_FRAME_DIM,
    },
    "split_sha256": V300_SPLIT_SHA256,
    "technical_corrigendum_id": V300_TECHNICAL_CORRIGENDUM_ID,
    "declared_coefficient_hash_id": V300_DECLARED_COEFFICIENT_HASH_ID,
    "declared_radius_candidate_hash_id": V300_DECLARED_RADIUS_CANDIDATE_HASH_ID,
    "coefficient_payload_sha256": V300_COEFFICIENT_SHA256,
    "radius_candidate_payload_sha256": V300_RADIUS_CANDIDATE_SHA256,
    "radius_candidates": RADIUS_CANDIDATES,
    "thresholds": asdict(THRESHOLDS),
    "tail_fixed_batch_size": TAIL_FIXED_BATCH_SIZE,
}


def frozen_spec_sha256() -> str:
    return sha256_text(canonical_json(FROZEN_SPEC))


def radius_candidate_payload_v300() -> dict:
    return {
        "schema": "green-bridge-v3.0.0-global-radius-candidates-v1",
        "candidates": list(RADIUS_CANDIDATE_SYMBOLS),
    }


def computed_radius_candidate_payload_sha256_v300() -> str:
    return sha256_text(canonical_json(radius_candidate_payload_v300()))


def radius_candidate_payload_sha256_v300() -> str:
    """Return the reproducible hash of the exact semantic payload."""
    computed = computed_radius_candidate_payload_sha256_v300()
    if computed != V300_RADIUS_CANDIDATE_SHA256:
        raise AssertionError("radius candidate canonical payload hash changed")
    return computed
