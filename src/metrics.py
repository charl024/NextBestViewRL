"""Learning-curve metrics and the run manifest, from SLM-Lab output folders.

    uv run --project SLM-Lab python src/metrics.py --runs "full=SLM-Lab/data/nbv_full_*" \
        --runs "hist1=SLM-Lab/data/nbv_hist1_*" --threshold 3.82 --floor 3.25
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from slm_policy import final_runs, load_spec


def load_run(run_dir):
    ''' (seed, eval DataFrame, spec) for one SLM-Lab output folder. '''
    run_dir = Path(run_dir)
    spec = load_spec(run_dir)
    df = pd.read_csv(next((run_dir / "info").glob("*_s0_session_df_eval.csv")))
    return int(spec["meta"]["random_seed"]), df, spec


def device_of(run_dir):
    for line in open(next((Path(run_dir) / "log").glob("*_s0.log")), errors="ignore"):
        if "device:" in line:
            return line.split("device:")[1].split(")")[0].split()[0]
    return "unknown"


def metrics(df, threshold, floor):
    t, r = df["frame"].to_numpy(), df["total_reward"].to_numpy()
    hit = np.flatnonzero(r >= threshold)
    return dict(auc=float(np.trapezoid(r, t) / (t[-1] - t[0])), final=float(r[-1]),
                to_threshold=int(t[hit[0]]) if len(hit) else None,
                failed=bool(r.max() < floor), wall_min=round(float(df["wall_t"].iloc[-1]) / 60, 1))


def sd(v):
    return float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def write_manifest(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True)   # label=glob of SLM-Lab run folders
    ap.add_argument("--threshold", type=float, required=True)   # farthest-point return on the train meshes
    ap.add_argument("--floor", type=float, required=True)       # random return on the train meshes
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    found, manifest = [], []
    for kv in a.runs:
        label, pattern = kv.split("=", 1)
        runs, rows = {}, {}
        for d in final_runs(pattern):
            seed, df, spec = load_run(d)
            runs[seed] = df
            rows[seed] = m = metrics(df, a.threshold, a.floor)
            manifest.append(dict(condition=label, spec_name=spec["name"], seed=seed,
                                 frame_budget=spec["env"]["max_frame"], status="completed",
                                 output_path=d, wall_minutes=m["wall_min"], device=device_of(d)))
        if not runs:
            print(f"{label:8s} no runs found")
            continue
        found.append(label)
        auc = np.array([m["auc"] for m in rows.values()])
        fin = np.array([m["final"] for m in rows.values()])
        print(f"{label:8s} AUC {auc.mean():.3f} +/- {sd(auc):.3f}  final {fin.mean():.3f} +/- {sd(fin):.3f}  "
              f"to_threshold {[m['to_threshold'] for m in rows.values()]}  "
              f"failed {sum(m['failed'] for m in rows.values())}/{len(rows)}  "
              f"minutes {[m['wall_min'] for m in rows.values()]}")

    if not found:
        raise SystemExit("no runs found for any label")
    write_manifest(manifest, Path(a.out) / "run-manifest.csv")


if __name__ == "__main__":
    main()
