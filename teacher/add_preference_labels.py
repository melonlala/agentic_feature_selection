"""Post-process an offline preference dataset: derive binary preference labels.

``teacher/collect_ant_preferences.py`` labels each comparison with the two
fragments' discounted returns (``return_a`` / ``return_b``). This script adds the
Bradley-Terry-style **binary preference** label expected by reward-learning code:

    preference = 1   if return_a >  return_b   (A is preferred)
    preference = 0   if return_a <  return_b   (B is preferred)
    preference = 0.5 if return_a == return_b   (tie; standard SyntheticGatherer convention)

For each ``returns_{source}.npz`` found under a run dir it rewrites that npz with an
added ``preference`` array, and (if present) injects the same array into the matching
``preferences_{source}.pkl`` dict. ``metadata.json`` gains a per-source label summary.

Usage:
    python teacher/add_preference_labels.py --input_dir outputs/preferences/seals_ant/seed0
    # or several at once:
    python teacher/add_preference_labels.py \\
        --input_dir outputs/preferences/seals_ant/seed{0,1,2}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def binary_preference(return_a: np.ndarray, return_b: np.ndarray) -> np.ndarray:
    """1.0 if A>B, 0.0 if A<B, 0.5 on ties — as float32."""
    pref = np.where(return_a > return_b, 1.0, 0.0)
    pref = np.where(return_a == return_b, 0.5, pref)
    return pref.astype(np.float32)


def process_source(npz_path: str) -> "dict":
    source = Path(npz_path).stem.replace("returns_", "")
    run_dir = Path(npz_path).parent

    data = dict(np.load(npz_path))
    return_a, return_b = data["return_a"], data["return_b"]
    pref = binary_preference(return_a, return_b)
    data["preference"] = pref

    # Rewrite the npz with the added label.
    np.savez(npz_path, **data)

    # Inject the label into the self-describing pickle too, if it exists.
    pkl_path = run_dir / f"preferences_{source}.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as fh:
            payload = pickle.load(fh)
        payload["preference"] = pref
        with open(pkl_path, "wb") as fh:
            pickle.dump(payload, fh)

    summary = {
        "n": int(len(pref)),
        "n_a_preferred": int((pref == 1.0).sum()),
        "n_b_preferred": int((pref == 0.0).sum()),
        "n_ties": int((pref == 0.5).sum()),
        "frac_a_preferred": float((pref == 1.0).mean()),
    }
    print(f"[add_pref_labels] {run_dir.name}/{source:7s} "
          f"A>B={summary['n_a_preferred']} A<B={summary['n_b_preferred']} "
          f"ties={summary['n_ties']} (frac_A={summary['frac_a_preferred']:.3f})")
    return {"source": source, "summary": summary}


def update_metadata(run_dir: Path, results: "list[dict]") -> None:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    meta["binary_preference"] = {
        "rule": "1 if return_a>return_b else 0 (0.5 on tie)",
        "label_key": "preference",
        "by_source": {r["source"]: r["summary"] for r in results},
    }
    # Advertise the new label alongside the existing return labels.
    keys = meta.get("label_keys", [])
    if "preference" not in keys:
        meta["label_keys"] = keys + ["preference"]
    meta_path.write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", nargs="+", required=True,
                   help="One or more preference run dirs containing returns_*.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for d in args.input_dir:
        npz_paths = sorted(glob.glob(os.path.join(d, "returns_*.npz")))
        if not npz_paths:
            print(f"[add_pref_labels] WARNING: no returns_*.npz under {d}")
            continue
        results = [process_source(p) for p in npz_paths]
        update_metadata(Path(d), results)
    print("[add_pref_labels] Done.")


if __name__ == "__main__":
    main()
