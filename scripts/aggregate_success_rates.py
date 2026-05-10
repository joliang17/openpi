"""Aggregate libero eval success rates from results/ into a TSV."""

import json
import pathlib

ROOT = pathlib.Path("/fs/nexus-scratch/yliang17/Research/VLA/openpi/results")
OUT = pathlib.Path("/fs/nexus-scratch/yliang17/Research/VLA/openpi/eval_success_rates.tsv")

# (subdir, method, suite) — top-level "" maps to action_expert/freeze_vlm on libero_pro
GROUPS = [
    ("",                 "freeze_vlm", "libero_pro"),
    ("ki_libero10",      "ki",         "libero_10"),
    ("ki_libero_pro",    "ki",         "libero_pro"),
    ("vlm_lora_libero10","vlm_lora",   "libero_10"),
    ("vlm_lora_libero_pro","vlm_lora", "libero_pro"),
]

COLS = [
    "method", "suite", "perturbation_type", "seed", "action_horizon",
    "total_successes", "total_episodes", "success_rate", "success_percent",
    "source_file",
]

rows = []
for subdir, method, suite in GROUPS:
    d = ROOT / subdir if subdir else ROOT
    for fp in sorted(d.glob("libero_eval_*.json")):
        with open(fp) as f:
            j = json.load(f)
        rows.append([
            method,
            suite,
            j.get("perturbation_type", ""),
            j.get("random_seed", ""),
            j.get("action_horizon", ""),
            j.get("total_successes", ""),
            j.get("total_episodes", ""),
            j.get("success_rate", ""),
            j.get("success_percent", ""),
            str(fp.relative_to(ROOT.parent)),
        ])

# sort: method, suite, seed, action_horizon
rows.sort(key=lambda r: (r[0], r[1], r[3], r[4]))

with open(OUT, "w") as f:
    f.write("\t".join(COLS) + "\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")

print(f"Wrote {len(rows)} rows to {OUT}")
