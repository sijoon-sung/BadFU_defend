# -*- coding: utf-8 -*-
"""여러 실행(det1, det2, ...) 결과 모으기.

FUBA 폴더에서:
    python -m diff_audit.aggregate --names det1 det2 det3 --inv universal
출력: logs/diff_audit_aggregate_{inv}.json
"""
import argparse
import json

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--inv", default="universal")
    a = p.parse_args(argv)
    runs = [json.load(open(f"logs/diff_audit_{n}_{a.inv}.json", encoding="utf-8")) for n in a.names]
    methods = list(runs[0]["summary"]["methods"].keys())
    agg = {"names": a.names, "inv": a.inv, "methods": {}, "localize": {}}
    for m in methods:
        rows = [r["summary"]["methods"][m] for r in runs]
        agg["methods"][m] = {
            "detect_rate": float(np.mean([bool(x.get("detected")) for x in rows])),
            "flag_is_target_rate": float(np.mean([bool(x.get("flag_is_target")) for x in rows])),
            "margin_mean": float(np.mean([x.get("margin_over_max_benign", np.nan) for x in rows])),
            "per_request_inversions": rows[0]["per_request_inversions"],
            "per_request_seconds_mean": float(np.mean([x["per_request_seconds_est"] for x in rows])),
        }
    for s in runs[0]["summary"]["localize"]:
        ranks = [r["summary"]["localize"][s]["requester_target_rank"] for r in runs]
        agg["localize"][s] = {"requester_target_ranks": ranks,
                              "top1_rate": float(np.mean([x == 0 for x in ranks]))}
    out = f"logs/diff_audit_aggregate_{a.inv}.json"
    json.dump(agg, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{'method':28s} {'detect':>7s} {'flag=tgt':>8s} {'margin':>8s} {'inv/req':>7s} {'s/req':>7s}")
    for m, r in agg["methods"].items():
        print(f"{m:28s} {r['detect_rate']:7.2f} {r['flag_is_target_rate']:8.2f} {r['margin_mean']:8.3f} "
              f"{r['per_request_inversions']:7d} {r['per_request_seconds_mean']:7.2f}")
    for s, r in agg["localize"].items():
        print(f"localize[{s}] 타깃 순위={r['requester_target_ranks']} 1위 비율={r['top1_rate']:.2f}")
    print("saved", out)


if __name__ == "__main__":
    main()
