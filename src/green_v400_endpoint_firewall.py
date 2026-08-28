"""Packet-level firewall for prospective GREEN endpoint adjudication.

The prediction and endpoint routes are intentionally symmetric in distrust:
prediction workers cannot receive held-out directions or outcomes, and endpoint
workers cannot receive GREEN/baseline predictions or certificate state.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


PREDICTION_FORBIDDEN_KEYS = frozenset(
    {
        "endpoint_direction",
        "endpoint_directions",
        "endpoint_panel",
        "endpoint_response",
        "endpoint_responses",
        "heldout_transport_error",
        "heldout_transport_failure",
        "nmh_recovery",
        "endpoint_failure_label",
        "endpoint_calibration_distribution",
    }
)

ENDPOINT_FORBIDDEN_KEYS = frozenset(
    {
        "green_certificate_status",
        "green_certificate_width",
        "p13_status",
        "p13_result",
        "baseline_prediction",
        "baseline_predictions",
        "green_prediction",
        "green_direction",
        "green_directions",
        "green_panel",
        "adaptive_budget_history",
        "selection_score",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def packet_sha256(packet: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(packet)).hexdigest()


def _nested_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _nested_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_keys(item)


def _forbidden_present(packet: dict[str, Any], forbidden: frozenset[str]) -> list[str]:
    present = set()
    for key in _nested_keys(packet):
        if any(key == blocked or key.startswith(blocked + "_") for blocked in forbidden):
            present.add(key)
    return sorted(present)


def _required_identity(packet: dict[str, Any]) -> tuple[str, str]:
    protocol_id = packet.get("protocol_id")
    row_id = packet.get("row_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("packet requires a nonempty protocol_id")
    if not isinstance(row_id, str) or len(row_id) != 64:
        raise ValueError("packet requires a 64-character row_id")
    try:
        int(row_id, 16)
    except ValueError as exc:
        raise ValueError("row_id must be hexadecimal") from exc
    return protocol_id, row_id


def seal_prediction_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Validate and commit a prediction-route packet before endpoint access."""

    _required_identity(packet)
    forbidden = _forbidden_present(packet, PREDICTION_FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError("prediction packet contains endpoint fields: " + ", ".join(forbidden))
    if packet.get("contains_endpoint_outcome") is not False:
        raise ValueError("prediction packet must declare contains_endpoint_outcome=false")
    if packet.get("route") != "prediction":
        raise ValueError("prediction packet route must equal prediction")
    if packet.get("committed_before_endpoint") is not True:
        raise ValueError("prediction must be committed before endpoint adjudication")
    digest = packet_sha256(packet)
    return {
        "schema_version": "green-v400-prediction-commitment-v1",
        "protocol_id": packet["protocol_id"],
        "row_id": packet["row_id"],
        "prediction_packet_sha256": digest,
        "committed_before_endpoint": True,
    }


def seal_endpoint_packet(
    packet: dict[str, Any], prediction_commitment: dict[str, Any]
) -> dict[str, Any]:
    """Validate endpoint isolation and bind it to an earlier prediction hash."""

    protocol_id, row_id = _required_identity(packet)
    forbidden = _forbidden_present(packet, ENDPOINT_FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError("endpoint packet contains prediction fields: " + ", ".join(forbidden))
    if packet.get("route") != "endpoint":
        raise ValueError("endpoint packet route must equal endpoint")
    if packet.get("contains_prediction") is not False:
        raise ValueError("endpoint packet must declare contains_prediction=false")
    if packet.get("adaptive_query_allocation") is not False:
        raise ValueError("endpoint packet must forbid adaptive query allocation")
    if prediction_commitment.get("committed_before_endpoint") is not True:
        raise ValueError("missing prior prediction commitment")
    if prediction_commitment.get("protocol_id") != protocol_id:
        raise ValueError("protocol mismatch between endpoint and prediction commitment")
    if prediction_commitment.get("row_id") != row_id:
        raise ValueError("row mismatch between endpoint and prediction commitment")
    prediction_hash = prediction_commitment.get("prediction_packet_sha256")
    if not isinstance(prediction_hash, str) or len(prediction_hash) != 64:
        raise ValueError("invalid prediction packet commitment hash")
    return {
        "schema_version": "green-v400-endpoint-commitment-v1",
        "protocol_id": protocol_id,
        "row_id": row_id,
        "prediction_packet_sha256": prediction_hash,
        "endpoint_packet_sha256": packet_sha256(packet),
        "prediction_committed_before_endpoint": True,
        "route_fields_disjoint": True,
    }


def audit_commitment_pair(
    prediction_packet: dict[str, Any],
    prediction_commitment: dict[str, Any],
    endpoint_packet: dict[str, Any],
    endpoint_commitment: dict[str, Any],
) -> None:
    """Fail closed if either packet changed after its commitment."""

    expected_prediction = seal_prediction_packet(prediction_packet)
    if expected_prediction != prediction_commitment:
        raise ValueError("prediction packet or commitment changed")
    expected_endpoint = seal_endpoint_packet(endpoint_packet, prediction_commitment)
    if expected_endpoint != endpoint_commitment:
        raise ValueError("endpoint packet or commitment changed")
