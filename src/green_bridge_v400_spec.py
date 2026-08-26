"""Immutable GREEN v4 static-formal-prepare protocol constants."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


PROTOCOL_ID = "GREEN_V400_JOINT_WITNESS_BOUNDARY_TRANSITION_FORMAL_PREPARE_V1"
BINDING_PARENT_COMMIT = "48182844a43d391439704f27aa26d513d33adaa0"
BRANCH = "codex/green-v400-joint-witness-formal-prepare"
MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TRANSFORMER_LENS_COMMIT = "4a4dc26c750475b29e6f54b362c2aab988702c9c"
TRANSFORMER_LENS_VERSION = "3.6.0"
TRANSFORMER_LENS_RELEASE_TAG = "v3.6.0"
BRANCH_ORDER = ("PAT_J", "PAT_B", "TAR_J", "TAR_B")
BRANCH_CONTRAST = (1, -1, -1, 1)
from green_bridge_v400_branch_semantics import binding_control_ast


CONTROL_AST = binding_control_ast()
TRANSFORMER_SEMANTICS_FLAGS = {
    "normalization_type": "LN",
    "activation_contains": "gelu",
    "fold_ln": False,
    "center_writing_weights": False,
    "center_unembed": False,
    "refactor_factored_attn_matrices": False,
    "attention_implementation": "eager",
    "evaluation_mode": True,
}

LOCAL_PRECISION_BITS = 256
OFFICIAL_PRECISION_BITS = 384
AUDIT_PRECISION_BITS = 512
ABSOLUTE_WIDTH_TOLERANCE_EXPONENT = -80
RELATIVE_WIDTH_TOLERANCE_EXPONENT = -40
MAX_SUBDIVISION_DEPTH = 24
MAX_CELLS_PER_ROW = 262_144
MAX_GRAPH_NODES = 2_000_000
MAX_SCALAR_MPFR_OPERATIONS_PER_ROW = 100_000_000
MAX_MEMORY_GIB_PER_WORKER = 64
MAX_FORMAL_PREPARE_WALL_HOURS = 24
HF_TL_PARITY_MAX_ABS = 3.0e-4  # inherited frozen raw-logit Gate-04 limit
GRAPH_TL_PARITY_MAX_ABS = 3.0e-4

ALPHA_EXPONENTS = tuple(range(17))
Q_EXPONENTS = tuple(range(-52, -11, 2))
DITHER_REPLICATES = 4
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20260805
PERMUTATION_REPLICATES = 100_000
PERMUTATION_SEED = 20260805

FORMAL_PREPARE_ONLY = True
REAL_ROW_CERTIFICATE_AUTHORIZED = False
DEVELOPMENT_AUTHORIZED = False
CONFIRMATION_AUTHORIZED = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "green_bridge_v400_formal_prepare.json"
CANDIDATE_NOUNS_PATH = (
    PROJECT_ROOT / "analysis" / "GREEN_V400_CANDIDATE_NOUNS_20260826.txt"
)
SEALED_NOUN_HASHES_PATH = (
    PROJECT_ROOT / "analysis" / "GREEN_V400_SEALED_NOUN_HASHES_20260826.json"
)


class FailureCode(str, Enum):
    INVALID_DOMAIN = "INVALID_DOMAIN"
    UNSUPPORTED_PRIMITIVE = "UNSUPPORTED_PRIMITIVE"
    INCOMPLETE_CAUSAL_CONE = "INCOMPLETE_CAUSAL_CONE"
    ROUNDING_IMPLEMENTATION_INVALID = "ROUNDING_IMPLEMENTATION_INVALID"
    GRAPH_PARITY_INVALID = "GRAPH_PARITY_INVALID"
    CERTIFICATE_IMPLEMENTATION_INVALID = "CERTIFICATE_IMPLEMENTATION_INVALID"
    RESOURCE_INCONCLUSIVE = "RESOURCE_INCONCLUSIVE"
    SEALED_SET_CONTAMINATION = "SEALED_SET_CONTAMINATION"
    PROVENANCE_MISMATCH = "PROVENANCE_MISMATCH"


SUPPORTED_OPERATIONS = (
    "constant", "affine_control", "add", "sub", "mul", "reciprocal",
    "exp", "log", "sqrt", "inv_sqrt", "tanh", "erf", "sigmoid",
    "gelu_new", "gelu_erf", "layernorm", "einsum", "softmax",
    "attention", "reshape", "transpose", "slice", "concat", "gather_static",
    "residual_add", "contrast",
)


@dataclass(frozen=True)
class PrecisionPolicy:
    local: int = LOCAL_PRECISION_BITS
    official: int = OFFICIAL_PRECISION_BITS
    audit: int = AUDIT_PRECISION_BITS


@dataclass(frozen=True)
class PartitionPolicy:
    absolute_tolerance_exponent: int = ABSOLUTE_WIDTH_TOLERANCE_EXPONENT
    relative_tolerance_exponent: int = RELATIVE_WIDTH_TOLERANCE_EXPONENT
    max_depth: int = MAX_SUBDIVISION_DEPTH
    max_cells: int = MAX_CELLS_PER_ROW
