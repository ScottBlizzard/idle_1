"""Deterministic local-import closure for a frozen GREEN v4.1 runtime bundle."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

from green_v410_artifacts import file_sha256
from green_v410_protocol import PROTOCOL_ID, sha256_canonical


def _module_path(root: Path, module: str) -> Path | None:
    relative = Path(*module.split("."))
    candidates = (
        root / "src" / relative.with_suffix(".py"),
        root / "analysis" / relative.with_suffix(".py"),
        root / relative.with_suffix(".py"),
    )
    return next((path for path in candidates if path.is_file()), None)


def discover_python_closure(root: Path, seeds: Iterable[Path]) -> list[Path]:
    pending = [path.resolve() for path in seeds]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file() or root.resolve() not in path.parents:
            raise ValueError(f"source seed is outside the repository: {path}")
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module)
        for module in modules:
            candidate = _module_path(root, module)
            if candidate is not None and candidate.resolve() not in seen:
                pending.append(candidate.resolve())
    return sorted(seen, key=lambda path: path.relative_to(root.resolve()).as_posix())


def build_source_bundle_manifest(
    *, root: Path, seeds: Iterable[Path], frozen_inputs: Iterable[Path],
) -> dict[str, Any]:
    root = root.resolve()
    paths = discover_python_closure(root, seeds)
    for path in frozen_inputs:
        resolved = path.resolve()
        if not resolved.is_file() or root not in resolved.parents:
            raise ValueError(f"frozen source input is outside the repository: {resolved}")
        paths.append(resolved)
    unique = sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())
    records = [{
        "path": path.relative_to(root).as_posix(),
        "byte_length": path.stat().st_size,
        "file_sha256": file_sha256(path),
    } for path in unique]
    payload = {
        "schema_version": "green-v410-p13-source-bundle-v1",
        "protocol_id": PROTOCOL_ID,
        "file_count": len(records),
        "files": records,
        "contains_endpoint_material": False,
    }
    return payload | {"bundle_sha256": sha256_canonical(payload)}
