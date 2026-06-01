"""Update results_csv/eval_summary.csv from LIBERO JSON result files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


DEFAULT_ROOT = Path("/fs/nexus-scratch/yliang17/Research/VLA/openpi")
DEFAULT_RESULTS_DIR = DEFAULT_ROOT / "results"
DEFAULT_SUMMARY_CSV = DEFAULT_ROOT / "results_csv" / "eval_summary.csv"

SUMMARY_FIELDS = [
    "model",
    "action_horizon",
    "seeds",
    "libero10_avg",
    "libero_pro_object_avg",
    "libero_pro_semantic_avg",
    "libero_pro_task_avg",
]

RESULT_FILE_GLOB = "libero_eval_*.json"
LIBERO10_SUITES = {"libero10"}
LIBERO_PRO_COLUMNS = {
    "object": "libero_pro_object_avg",
    "semantic": "libero_pro_semantic_avg",
    "task": "libero_pro_task_avg",
}

MODEL_ALIASES = {
    "action_expert_jax": "action_expert",
    "action_expert_jax_v2": "action_expert_v2",
    "libero_vlm_lora": "vlm_lora",
    "ki_vlm_lora_action_expert": "ki",
    "skill_router_stage1": "skill_router",
    "skill_router_stage2": "skill_router",
    "skill_router_stage2_adarms_only": "skill_router_adarms_only",
    "skill_router_stage2_film_only": "skill_router_film_only",
    "skill_router_stage2_film_vlm": "skill_router_film_vlm",
    "vlm_lora_skill_effect_gate_router_joint": "skill_effect_gate_router",
    "vlm_lora_gated_film_skill_router_joint": "gated_film_skill_router",
}


def _format_average(value: float | None) -> str:
    if value is None:
        return ""
    rounded = round(float(value), 2)
    text = f"{rounded:.2f}".rstrip("0").rstrip(".")
    return text if "." in text else f"{text}.0"


def _normalize_folder_name(folder_name: str) -> str:
    base = folder_name
    for suffix in ("_libero10", "_libero_10", "_libero_pro", "_libero_pro_object", "_libero_pro_semantic", "_libero_pro_task"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.startswith("pi05_libero_"):
        base = base.removeprefix("pi05_libero_")
    elif base.startswith("pi05_"):
        base = base.removeprefix("pi05_")
    return MODEL_ALIASES.get(base, base)


def _model_from_result(path: Path, result: dict[str, object]) -> str:
    """Model identifier for a result JSON not stored inside a model folder.

    Falls back to the `model_name` field, then to the `model<...>_task` token
    in the filename, then to the bare filename stem.
    """
    name = str(result.get("model_name") or "").strip()
    if name:
        return name
    match = re.search(r"model(.+?)_task", path.stem)
    return match.group(1) if match else path.stem


def _result_column(result: dict[str, object]) -> str | None:
    suite = str(result.get("task_suite_name", "")).replace("_", "").lower()
    perturbation_type = str(result.get("perturbation_type", "")).replace("_", "").lower()

    # Check perturbation_type first: libero_pro files store task_suite_name as "libero_10"
    # but set perturbation_type to distinguish them from vanilla libero10.
    if perturbation_type and perturbation_type != "none":
        if perturbation_type in LIBERO_PRO_COLUMNS:
            return LIBERO_PRO_COLUMNS[perturbation_type]
        return "libero_pro_object_avg"

    if suite in LIBERO10_SUITES:
        return "libero10_avg"
    if suite.startswith("liberopro"):
        return "libero_pro_object_avg"
    return None


def _iter_result_files(results_dir: Path):
    yield from sorted(results_dir.glob(f"**/{RESULT_FILE_GLOB}"))


def _load_result_rows(results_dir: Path) -> dict[tuple[str, str], dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for path in _iter_result_files(results_dir):
        rel_parts = path.relative_to(results_dir).parts

        with path.open() as f:
            result = json.load(f)

        horizon = str(result.get("action_horizon", ""))
        if not horizon:
            continue

        if len(rel_parts) >= 2:
            # Result JSON inside a per-model folder: folder name is the model.
            model = _normalize_folder_name(rel_parts[0])
        else:
            # Result JSON stored directly under results/ (no model folder):
            # derive the model from the file's own metadata / filename.
            model = _normalize_folder_name(_model_from_result(path, result))
        key = (model, horizon)
        row = grouped.setdefault(
            key,
            {
                "model": model,
                "action_horizon": horizon,
                "seeds": set(),
                "values": defaultdict(list),
            },
        )

        seed = result.get("random_seed")
        if seed not in (None, ""):
            row["seeds"].add(int(seed))

        column = _result_column(result)
        if column is None:
            continue

        success_percent = result.get("success_percent")
        if success_percent is None:
            success_rate = result.get("success_rate")
            success_percent = None if success_rate is None else float(success_rate) * 100
        if success_percent is not None:
            row["values"][column].append(float(success_percent))

    return grouped


def _read_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    if not summary_csv.exists():
        return []
    with summary_csv.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _build_row_template(model: str, action_horizon: str) -> dict[str, str]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row["model"] = model
    row["action_horizon"] = action_horizon
    return row


def _apply_discovered_row(base_row: dict[str, str], discovered: dict[str, object]) -> dict[str, str]:
    row = {field: base_row.get(field, "") for field in SUMMARY_FIELDS}
    row["model"] = str(discovered["model"])
    row["action_horizon"] = str(discovered["action_horizon"])
    row["seeds"] = ",".join(str(seed) for seed in sorted(discovered["seeds"]))

    values: dict[str, list[float]] = discovered["values"]  # type: ignore[assignment]
    for column in ("libero10_avg", "libero_pro_object_avg", "libero_pro_semantic_avg", "libero_pro_task_avg"):
        row[column] = _format_average(mean(values[column])) if values.get(column) else ""
    return row


def update_summary(summary_csv: Path, results_dir: Path) -> int:
    rows = _read_summary_rows(summary_csv)
    rows_by_key = {(row["model"], row["action_horizon"]): row for row in rows}

    discovered_rows = _load_result_rows(results_dir)
    updated = 0
    for key in sorted(discovered_rows):
        discovered = discovered_rows[key]
        base_row = rows_by_key.get(key, _build_row_template(discovered["model"], discovered["action_horizon"]))
        new_row = _apply_discovered_row(base_row, discovered)
        rows_by_key[key] = new_row
        updated += 1

    ordered_rows = []
    seen_keys = set()
    for row in rows:
        key = (row["model"], row["action_horizon"])
        if key in rows_by_key:
            ordered_rows.append(rows_by_key[key])
            seen_keys.add(key)
        else:
            ordered_rows.append(row)
            seen_keys.add(key)

    for key in sorted(rows_by_key):
        if key not in seen_keys:
            ordered_rows.append(rows_by_key[key])

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered_rows)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    updated = update_summary(args.summary_csv, args.results_dir)
    print(f"Updated {updated} dynamic rows in {args.summary_csv}")


if __name__ == "__main__":
    main()
