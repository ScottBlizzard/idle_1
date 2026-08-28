"""Build an outcome-free Greater-Than replication universe.

Only the tokenizer is used. The module never loads model weights and never
computes logits, activations, attention, certificates, or endpoint values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from analysis.green_v400_silent_failure_prepare import canonical_bytes, sha256_value


REQUIRED_ROLES = (
    "endpoint_calibration",
    "development",
    "confirmation",
    "unused_reserve",
)


def validate_universe_config(config: dict[str, Any]) -> None:
    if config.get("contains_scientific_outcome") is not False:
        raise ValueError("contains_scientific_outcome must be false")
    if config.get("model_weights_may_be_loaded") is not False:
        raise ValueError("model_weights_may_be_loaded must be false")
    if config.get("execution_authorized") is not False:
        raise ValueError("execution_authorized must be false")
    if config.get("task") != "Greater-Than":
        raise ValueError("task must be Greater-Than")
    if config.get("site_family") != "resid_post_at_start_year_suffix":
        raise ValueError("unexpected intervention site family")
    layers = tuple(int(value) for value in config.get("layers", []))
    if layers != tuple(range(9)):
        raise ValueError("replication must preserve the frozen layers 0 through 8")
    template = str(config.get("prompt_template", ""))
    for field in ("{noun}", "{cc:02d}", "{y:02d}"):
        if field not in template:
            raise ValueError(f"prompt template lacks {field}")
    roles = config.get("role_nouns", {})
    if tuple(roles) != REQUIRED_ROLES:
        raise ValueError("role_nouns must use the frozen role order")
    nouns = [str(noun) for role in REQUIRED_ROLES for noun in roles[role]]
    if not nouns or len(nouns) != len(set(nouns)):
        raise ValueError("role nouns must be nonempty and disjoint")
    if any(not noun.isascii() or not noun.isalpha() or not noun.islower() for noun in nouns):
        raise ValueError("nouns must be lowercase ASCII alphabetic strings")
    centuries = tuple(int(value) for value in config.get("centuries", []))
    if not centuries or len(centuries) != len(set(centuries)):
        raise ValueError("centuries must be nonempty and unique")
    low = int(config.get("suffix_min", -1))
    high = int(config.get("suffix_max", -1))
    if not (0 <= low < high <= 99):
        raise ValueError("invalid suffix range")
    bins = config.get("distance_bins", {})
    if tuple(bins) != ("near", "far"):
        raise ValueError("distance bins must be near then far")
    for name, bounds in bins.items():
        if len(bounds) != 2 or not (0 < int(bounds[0]) <= int(bounds[1])):
            raise ValueError(f"invalid {name} distance bin")
    count = int(config.get("records_per_cell", 0))
    quotas = config.get("orientations_per_cell", {})
    if count <= 0 or count != int(quotas.get("up", -1)) + int(quotas.get("down", -1)):
        raise ValueError("orientation quotas must sum to records_per_cell")
    endpoint = config.get("endpoint_contract", {})
    if endpoint.get("secondary_is_prediction_input") is not False:
        raise ValueError("secondary endpoint must remain hidden from prediction")
    if endpoint.get("secondary_hook") != "blocks.10.hook_mlp_out":
        raise ValueError("secondary endpoint hook changed")
    if any(layer >= 10 for layer in layers):
        raise ValueError("secondary endpoint must be strictly downstream")


def _suffix_token_ids(tokenizer: Any) -> list[int]:
    result: list[int] = []
    for suffix in range(100):
        text = f"{suffix:02d}"
        ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        if len(ids) != 1 or tokenizer.decode(ids) != text:
            raise ValueError(f"suffix {text} is not one exact token")
        result.append(ids[0])
    if len(set(result)) != 100:
        raise ValueError("suffix token identifiers must be unique")
    return result


def _eligible_pair(
    tokenizer: Any,
    template: str,
    noun: str,
    century: int,
    first: int,
    second: int,
    suffix_ids: list[int],
) -> dict[str, Any] | None:
    clean_text = template.format(noun=noun, cc=century, y=first)
    corrupt_text = template.format(noun=noun, cc=century, y=second)
    clean = [int(value) for value in tokenizer.encode(clean_text, add_special_tokens=False)]
    corrupt = [int(value) for value in tokenizer.encode(corrupt_text, add_special_tokens=False)]
    if len(clean) != len(corrupt):
        return None
    differing = [index for index, values in enumerate(zip(clean, corrupt)) if values[0] != values[1]]
    if len(differing) != 1:
        return None
    site_position = differing[0]
    if clean[site_position] != suffix_ids[first] or corrupt[site_position] != suffix_ids[second]:
        return None
    century_ids = [int(value) for value in tokenizer.encode(f" {century:02d}", add_special_tokens=False)]
    clean_year = [int(value) for value in tokenizer.encode(f" {century:02d}{first:02d}", add_special_tokens=False)]
    corrupt_year = [int(value) for value in tokenizer.encode(f" {century:02d}{second:02d}", add_special_tokens=False)]
    if len(century_ids) != 1:
        return None
    if clean_year != [century_ids[0], suffix_ids[first]]:
        return None
    if corrupt_year != [century_ids[0], suffix_ids[second]]:
        return None
    if clean[-1] != century_ids[0] or corrupt[-1] != century_ids[0]:
        return None
    return {
        "clean_text": clean_text,
        "corrupt_text": corrupt_text,
        "clean_token_ids": clean,
        "corrupt_token_ids": corrupt,
        "site_position": site_position,
        "final_position": len(clean) - 1,
        "clean_suffix_token_id": suffix_ids[first],
        "corrupt_suffix_token_id": suffix_ids[second],
        "output_century_token_id": century_ids[0],
        "sequence_length": len(clean),
    }


def _candidate_pairs(config: dict[str, Any], distance: str) -> list[tuple[int, int]]:
    lower, upper = (int(value) for value in config["distance_bins"][distance])
    suffix_min = int(config["suffix_min"])
    suffix_max = int(config["suffix_max"])
    return [
        (first, second)
        for first in range(suffix_min, suffix_max + 1)
        for second in range(first + 1, suffix_max + 1)
        if lower <= second - first <= upper
    ]


def _rank(
    domain: str,
    kind: str,
    role: str,
    noun: str,
    century: int,
    distance: str,
    first: int,
    second: int,
) -> str:
    payload = (
        f"{domain}\0{kind}\0{role}\0{noun}\0{century:02d}\0"
        f"{distance}\0{first:02d}\0{second:02d}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_untouched_universe(tokenizer: Any, config: dict[str, Any]) -> dict[str, Any]:
    validate_universe_config(config)
    suffix_ids = _suffix_token_ids(tokenizer)
    domain = str(config["candidate_hash_domain"])
    count = int(config["records_per_cell"])
    requested_quota = {
        "up": int(config["orientations_per_cell"]["up"]),
        "down": int(config["orientations_per_cell"]["down"]),
    }
    rows: list[dict[str, Any]] = []
    for role in REQUIRED_ROLES:
        for noun in config["role_nouns"][role]:
            for century in config["centuries"]:
                for distance in config["distance_bins"]:
                    ranked: list[tuple[str, str, int, int, dict[str, Any]]] = []
                    for first, second in _candidate_pairs(config, distance):
                        encoded = _eligible_pair(
                            tokenizer,
                            config["prompt_template"],
                            noun,
                            int(century),
                            first,
                            second,
                            suffix_ids,
                        )
                        if encoded is None:
                            continue
                        ranked.append((
                            _rank(domain, "pair", role, noun, int(century), distance, first, second),
                            _rank(domain, "orient", role, noun, int(century), distance, first, second),
                            first,
                            second,
                            encoded,
                        ))
                    quota = dict(requested_quota)
                    selected = 0
                    for pair_digest, orientation_digest, first, second, encoded in sorted(ranked):
                        preferred = "up" if int(orientation_digest[:2], 16) & 1 else "down"
                        orientation = preferred
                        if quota[orientation] == 0:
                            orientation = "down" if orientation == "up" else "up"
                        if quota[orientation] == 0:
                            continue
                        y, y_prime = (first, second) if orientation == "up" else (second, first)
                        if orientation == "down":
                            encoded = _eligible_pair(
                                tokenizer,
                                config["prompt_template"],
                                noun,
                                int(century),
                                y,
                                y_prime,
                                suffix_ids,
                            )
                            if encoded is None:
                                raise AssertionError("eligible pair lost under orientation reversal")
                        public = {
                            "role": role,
                            "noun": noun,
                            "century": int(century),
                            "distance_bin": distance,
                            "orientation": orientation,
                            "y": y,
                            "y_prime": y_prime,
                            "pair_digest": pair_digest,
                            "orientation_digest": orientation_digest,
                            **encoded,
                        }
                        row_id = hashlib.sha256(canonical_bytes(public)).hexdigest()
                        rows.append({"row_id": row_id, **public})
                        quota[orientation] -= 1
                        selected += 1
                        if selected == count:
                            break
                    if selected != count or any(quota.values()):
                        raise RuntimeError(
                            f"pair quota failure for {role}/{noun}/{century}/{distance}: {quota}"
                        )

    role_counts = {
        role: sum(row["role"] == role for row in rows)
        for role in REQUIRED_ROLES
    }
    site_positions = sorted({int(row["site_position"]) for row in rows})
    return {
        "schema_version": "green-v400-greater-than-untouched-universe-artifact-v1",
        "protocol_id": config["protocol_id"],
        "parent_protocol_id": config["parent_protocol_id"],
        "task": "Greater-Than",
        "contains_scientific_outcome": False,
        "model_weights_loaded": False,
        "tokenizer_only": True,
        "execution_authorized": False,
        "config_sha256": sha256_value(config),
        "suffix_token_ids": suffix_ids,
        "row_count": len(rows),
        "role_counts": role_counts,
        "site_positions": site_positions,
        "rows_sha256": sha256_value(rows),
        "secondary_endpoint_contract": dict(config["endpoint_contract"]),
        "rows": rows,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_universe_config(config)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=bool(config["local_files_only"]),
    )
    universe = build_untouched_universe(tokenizer, config)
    _atomic_write(args.output, universe)


if __name__ == "__main__":
    main()
