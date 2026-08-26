# GREEN v3.0.0 Canonical Payload Technical Corrigendum

Date: 2026-08-26  
Authority: Codex implementation-level correction  
Identity: `CODEX-GREEN-V300-CANONICAL-PAYLOAD-v1-20260826`

## Scope

This corrigendum repairs an omitted byte-serialization contract. It does not
change any coefficient, radius candidate, scientific threshold, dataset split,
phase authorization, success rule, or one-shot constraint in the GREEN v3.0.0
scientific decision.

The external decision recorded two SHA-256 identifiers without recording the
payload bytes or a serialization algorithm. A SHA-256 digest is not reversible,
so those identifiers cannot serve as reproducibility checks. They remain in the
frozen specification as immutable provenance under `declared_*_hash_id`.

## Canonical serialization

Both repaired payloads use:

```python
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
).encode("utf-8")
```

There is no byte-order mark and no trailing newline. The digest field is not
included in its own payload.

## Helmert coefficient payload

Exact symbols are used instead of evaluated binary floats:

```json
{"rows":[["1/sqrt(5)","1/sqrt(5)","1/sqrt(5)","1/sqrt(5)","1/sqrt(5)"],["1/sqrt(2)","-1/sqrt(2)","0","0","0"],["1/sqrt(6)","1/sqrt(6)","-2/sqrt(6)","0","0"],["1/sqrt(12)","1/sqrt(12)","1/sqrt(12)","-3/sqrt(12)","0"]],"schema":"green-bridge-v3.0.0-helmert-coefficients-v1"}
```

Reproducible SHA-256:

```text
71d1f91b7a7da68e1d73079e42b116e09cf3544b890f53aac1d58afae4bf4cfa
```

## Radius candidate payload

Exact rational symbols are used instead of evaluated binary floats:

```json
{"candidates":["1","1/2","1/4","1/8","1/16","1/32","1/64"],"schema":"green-bridge-v3.0.0-global-radius-candidates-v1"}
```

Reproducible SHA-256:

```text
370173c38e04bf741145faf09d5cffc826810d206c684b97de65c07d13303d6c
```

## Scientific invariance

At runtime the four coefficient rows still evaluate to the exact Helmert
formulae in the decision, and the radius set still evaluates to
`{1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64}`. No outcome or response data influenced
this correction.
