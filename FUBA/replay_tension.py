"""Descriptive replay of a training ledger; no unlearning or fitted threshold.

Historical caches contain CLASS-MEAN log-probability effects, not per-input JVPs.
Rankings below diagnose the proposed boundary score and are not new calibrated
alarms or evidence of exact FU behavior. Checkpoints are not loaded at all.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tension_audit import boundary_profile


def run(folder, out):
    paths = sorted(folder.glob("bd_*_r100w5_k20paper_s*_push.npz"))
    if not paths:
        raise ValueError("no registered paper-setting training caches found")
    report = {"method": "boundary_tension_v3", "model_unlearning_calls": 0, "runs": [],
              "measurement": "historical cumulative class-mean log-probability effects",
              "source_sha256": {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                for n in ("tension_audit.py", "replay_tension.py")},
              "limits": ["descriptive rankings, no calibrated alarm", "not per-input JVP measurements",
                         "same historical FUBA trajectories, no new training", "not original BadFU data-shard evaluation"]}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            k, t = int(data["K"]), int(data["T"])
            ts, js, vs = data["ts"], data["js"], data["vs"]
        if vs.ndim != 2 or len(ts) != len(js) or len(ts) != len(vs):
            raise ValueError("invalid class-mean cache")
        if not np.isin(js, np.arange(k)).all() or not np.isin(ts, np.arange(1,t+1)).all():
            raise ValueError("invalid event identifiers")
        run = {"cache": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "snapshots": []}
        for stop in sorted(set([min(10,t), min(20,t), min(40,t), t])):
            e = np.zeros((k, vs.shape[1]))
            for j, value in zip(js[ts <= stop], vs[ts <= stop]):
                e[int(j)] += value
            scores = []
            for j in range(k):
                r = np.zeros(k); r[j] = 1
                p = boundary_profile(e, r)
                c = int(p["score"].argmax())
                scores.append({"unit": j, "coordinate": c, "score": float(p["score"][c]),
                               "balance": float(p["balance"][c]), "selective_mass": float(p["selective_mass"][c])})
            scores.sort(key=lambda s: (-s["score"], s["unit"]))
            run["snapshots"].append({"round": stop, "ranking": scores})
        report["runs"].append(run)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for r in report["runs"]:
        print(Path(r["cache"]).name, [(s["round"], s["ranking"][0]) for s in r["snapshots"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("FUBA/logs/ledger"))
    parser.add_argument("--out", type=Path, default=Path("FUBA/logs/tension_audit/cache_replay_v3.json"))
    a = parser.parse_args()
    run(a.folder, a.out)
