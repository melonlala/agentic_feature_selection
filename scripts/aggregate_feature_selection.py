"""Aggregate the feature-selection pipeline outputs into one comparison CSV.

Walks outputs/feature_selection/{env}/{task}/students/{method}/summary.csv and,
per (env, task, method), reports the best student test score over the topk sweep
plus the ranking time, into summary_seed{SEED}.csv.

Usage:
    python scripts/aggregate_feature_selection.py --root outputs/feature_selection --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Per task: the test column to optimise and whether higher is better.
_PRIMARY = {
    "bc":  ("test_r2", True),
    "irl": ("test_r2", True),
    "pc":  ("test_pref_accuracy", True),
}


def _ranking_time(rank_meta: Path) -> float | None:
    if not rank_meta.exists():
        return None
    meta = json.loads(rank_meta.read_text())
    for k in ("ranking_time_sec", "ranking_time", "elapsed_s"):
        if k in meta:
            return float(meta[k])
    return None


def run(root: Path, seed: int) -> None:
    if not root.is_dir():
        print(f"[aggregate] root {root} does not exist; nothing to aggregate.")
        return
    rows = []
    for env_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in (env_dir).glob("*") if p.is_dir()):
            task = task_dir.name
            col, higher = _PRIMARY.get(task, ("test_r2", True))
            students = task_dir / "students"
            if not students.is_dir():
                continue
            for method_dir in sorted(p for p in students.iterdir() if p.is_dir()):
                method = method_dir.name
                summ = method_dir / "summary.csv"
                if not summ.exists():
                    continue
                df = pd.read_csv(summ)
                best = None
                if col in df.columns and len(df):
                    idx = df[col].idxmax() if higher else df[col].idxmin()
                    best = df.loc[idx]
                rank_meta = task_dir / "rankings" / method / "metadata.json"
                rows.append({
                    "env": env_dir.name,
                    "task": task,
                    "method": method,
                    "primary_metric": col,
                    "best_score": (float(best[col]) if best is not None else None),
                    "best_k": (int(best["k"]) if best is not None else None),
                    "ranking_time_sec": _ranking_time(rank_meta),
                })
    out = root / f"summary_seed{seed}.csv"
    df = pd.DataFrame(rows).sort_values(["env", "task", "method"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"[aggregate] wrote {out} ({len(df)} rows)")
    if len(df):
        print(df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate feature-selection results.")
    p.add_argument("--root", default="outputs/feature_selection")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(Path(a.root), a.seed)
