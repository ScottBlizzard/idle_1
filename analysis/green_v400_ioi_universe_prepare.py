"""Prepare a fresh IOI prompt universe without running model outcomes.

The script uses tokenizer calls only. It never loads model weights and never
computes logits, activations, attention, GREEN certificates, or endpoint values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from analysis.green_v400_silent_failure_prepare import canonical_bytes, sha256_value


def _hash_bytes(domain: str, template_id: str, counter: int) -> bytes:
    return hashlib.sha256(f"{domain}\0{template_id}\0{counter}".encode("utf-8")).digest()


def _single_token_names(tokenizer: Any, candidates: list[str]) -> list[tuple[str, int]]:
    valid: list[tuple[str, int]] = []
    seen: set[int] = set()
    for name in candidates:
        ids = tokenizer.encode(f" {name}", add_special_tokens=False)
        if len(ids) == 1 and int(ids[0]) not in seen:
            valid.append((name, int(ids[0])))
            seen.add(int(ids[0]))
    return valid


def _encode_candidate(
    tokenizer: Any,
    template: str,
    io_name: str,
    s_name: str,
    place: str,
    item: str,
) -> dict[str, Any] | None:
    clean_text = template.format(
        place=place, IO=io_name, S=s_name, item=item
    )
    corrupt_text = template.format(
        place=place, IO=s_name, S=s_name, item=item
    )
    clean = [int(v) for v in tokenizer.encode(clean_text, add_special_tokens=False)]
    corrupt = [int(v) for v in tokenizer.encode(corrupt_text, add_special_tokens=False)]
    if len(clean) != len(corrupt):
        return None
    io_ids = tokenizer.encode(f" {io_name}", add_special_tokens=False)
    s_ids = tokenizer.encode(f" {s_name}", add_special_tokens=False)
    if len(io_ids) != 1 or len(s_ids) != 1:
        return None
    io_id, s_id = int(io_ids[0]), int(s_ids[0])
    io_positions = [index for index, token in enumerate(clean) if token == io_id]
    s_positions = [index for index, token in enumerate(clean) if token == s_id]
    if len(io_positions) != 1 or len(s_positions) != 2:
        return None
    return {
        "clean_text": clean_text,
        "corrupt_text": corrupt_text,
        "clean_token_ids": clean,
        "corrupt_token_ids": corrupt,
        "io_token_id": io_id,
        "s_token_id": s_id,
        "signature": [len(clean), io_positions[0], s_positions[0], s_positions[1]],
    }


def validate_universe_config(config: dict[str, Any]) -> None:
    if config.get("contains_scientific_outcome") is not False:
        raise ValueError("contains_scientific_outcome must be false")
    if config.get("execution_authorized") is not False:
        raise ValueError("execution_authorized must be false")
    if int(config.get("rows_per_template", 0)) <= 0:
        raise ValueError("rows_per_template must be positive")
    templates = config.get("templates", [])
    ids = [entry.get("id") for entry in templates]
    if not templates or len(ids) != len(set(ids)):
        raise ValueError("template ids must be present and unique")
    old = config.get("historical_template_forbidden")
    if not old or any(entry.get("text") == old for entry in templates):
        raise ValueError("the historical prompt grammar must remain excluded")
    for entry in templates:
        text = entry.get("text", "")
        for field in ("{place}", "{IO}", "{S}", "{item}"):
            if field not in text:
                raise ValueError(f"template {entry.get('id')} lacks {field}")
        if text.count("{IO}") != 1 or text.count("{S}") != 2:
            raise ValueError("each template must contain IO once and S twice")


def build_untouched_universe(tokenizer: Any, config: dict[str, Any]) -> dict[str, Any]:
    validate_universe_config(config)
    valid_names = _single_token_names(tokenizer, list(config["names"]))
    if len(valid_names) < 16:
        raise ValueError("fewer than 16 distinct leading-space single-token names")

    domain = config["candidate_hash_domain"]
    target = int(config["rows_per_template"])
    all_rows: list[dict[str, Any]] = []
    seen_semantics: set[tuple[str, str, str, str, str]] = set()
    seen_prompt_hashes: set[str] = set()

    for template in config["templates"]:
        accepted = 0
        counter = 0
        maximum = target * 500
        while accepted < target and counter < maximum:
            digest = _hash_bytes(domain, template["id"], counter)
            counter += 1
            io_index = int.from_bytes(digest[0:4], "big") % len(valid_names)
            s_index = int.from_bytes(digest[4:8], "big") % (len(valid_names) - 1)
            if s_index >= io_index:
                s_index += 1
            place_index = int.from_bytes(digest[8:12], "big") % len(config["places"])
            item_index = int.from_bytes(digest[12:16], "big") % len(config["items"])
            io_name = valid_names[io_index][0]
            s_name = valid_names[s_index][0]
            place = config["places"][place_index]
            item = config["items"][item_index]
            semantic = (template["id"], io_name, s_name, place, item)
            if semantic in seen_semantics:
                continue
            encoded = _encode_candidate(
                tokenizer, template["text"], io_name, s_name, place, item
            )
            if encoded is None:
                continue
            prompt_hash = hashlib.sha256(
                encoded["clean_text"].encode("utf-8")
            ).hexdigest()
            if prompt_hash in seen_prompt_hashes:
                continue
            seen_semantics.add(semantic)
            seen_prompt_hashes.add(prompt_hash)
            row_payload = {
                "template_id": template["id"],
                "role": template["role"],
                "io_name": io_name,
                "s_name": s_name,
                "place": place,
                "item": item,
                "prompt_sha256": prompt_hash,
                **encoded,
            }
            row_id = hashlib.sha256(canonical_bytes(row_payload)).hexdigest()
            all_rows.append({"row_id": row_id, **row_payload})
            accepted += 1
        if accepted != target:
            raise RuntimeError(
                f"could construct only {accepted}/{target} rows for {template['id']}"
            )

    role_counts: dict[str, int] = {}
    signature_counts: dict[str, int] = {}
    for row in all_rows:
        role_counts[row["role"]] = role_counts.get(row["role"], 0) + 1
        key = json.dumps(row["signature"], separators=(",", ":"))
        signature_counts[key] = signature_counts.get(key, 0) + 1

    return {
        "schema_version": "green-v400-ioi-untouched-universe-artifact-v1",
        "protocol_id": config["protocol_id"],
        "contains_scientific_outcome": False,
        "model_weights_loaded": False,
        "tokenizer_only": True,
        "execution_authorized": False,
        "config_sha256": sha256_value(config),
        "valid_single_token_name_count": len(valid_names),
        "row_count": len(all_rows),
        "role_counts": role_counts,
        "signature_counts": signature_counts,
        "historical_prompt_grammar_excluded": True,
        "rows_sha256": sha256_value(all_rows),
        "rows": all_rows,
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
        config["model_id"], local_files_only=bool(config["local_files_only"])
    )
    universe = build_untouched_universe(tokenizer, config)
    _atomic_write(args.output, universe)


if __name__ == "__main__":
    main()

