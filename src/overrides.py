"""
Manual code overrides.

A technician can correct or add a code from the dashboard; those edits are saved
to a per-dataset overrides file and ALWAYS re-applied on top of the automatic
mapping — so manual fixes survive re-mapping, model changes and new runs.

Overrides file: <OVERRIDES_DIR>/<dataset>_overrides.json
Shape: { "cluster::variable": {"code": "...", "code_system": "...", "display": "..."} }
An empty/blank code resets that variable back to a local fallback code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _dir(cfg: Any) -> Path:
    return Path(getattr(cfg, "OVERRIDES_DIR", cfg.BASE_DIR / "overrides"))


def overrides_path(dataset_name: str, cfg: Any) -> Path:
    return _dir(cfg) / f"{dataset_name or 'default'}_overrides.json"


def load_overrides(dataset_name: str, cfg: Any) -> dict[str, dict]:
    p = overrides_path(dataset_name, cfg)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_overrides(dataset_name: str, cfg: Any, edits: dict[str, dict],
                   merge: bool = True) -> dict[str, dict]:
    """Merge `edits` into the stored overrides and persist. Returns the full set."""
    data = load_overrides(dataset_name, cfg) if merge else {}
    for key, val in (edits or {}).items():
        if not val or not (val.get("code") or "").strip():
            data.pop(key, None)          # blank code → clear the override
        else:
            data[key] = {
                "code": val.get("code", "").strip(),
                "code_system": (val.get("code_system") or "").strip(),
                "display": (val.get("display") or "").strip(),
            }
    p = overrides_path(dataset_name, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def clear_overrides(dataset_name: str, cfg: Any) -> int:
    """Delete all saved overrides for a dataset. Returns how many were removed."""
    p = overrides_path(dataset_name, cfg)
    n = 0
    if p.exists():
        try:
            n = len(json.loads(p.read_text()))
        except Exception:
            n = 0
        p.unlink()
    return n


def apply_overrides(state: dict, cfg: Any) -> dict:
    """Overlay the dataset's overrides onto the pipeline's mapping index."""
    name = (state.get("dataset") or {}).get("name", "")
    ov = load_overrides(name, cfg)
    if not ov:
        return state
    index = dict(state.get("mapping_index", {}))
    for key, o in ov.items():
        if not o:
            continue
        cluster, _, variable = key.partition("::")
        prev = index.get(key, {})
        index[key] = {
            "variable": variable or prev.get("variable", ""),
            "cluster": cluster or prev.get("cluster", ""),
            "code_system": o.get("code_system") or prev.get("code_system", ""),
            "code": o.get("code") or "UNMAPPED",
            "display": o.get("display") or prev.get("display", variable),
            "confidence": "override",
            "rationale": "Manual override (technician)",
            "candidates": prev.get("candidates", []),
            "ucum_unit": o.get("ucum_unit", prev.get("ucum_unit", "")),
        }
    state["mapping_index"] = index
    state["code_mappings"] = list(index.values())
    return state
