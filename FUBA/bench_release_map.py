# -*- coding: utf-8 -*-
"""빼 보기(release_map.py)의 요청 1건당 비용을 K(클라이언트 수)별로 잰다. 합성 궤적(무작위 가중치)이라 숫자는 시간만 의미.

  python bench_release_map.py --fmt fuba --Ks 8 20 50 100 --rounds 4
  python bench_release_map.py --fmt badfu_synth --Ks 6 20 50 --rounds 3      # ResNet-18 가중치로 badfu 형식 합성 궤적
설정 3종을 비교한다 (채점용 ASR 계산은 모두 끔):
  full      : 기준 = 나머지 전원, 탐색 = 클라이언트마다  (기본값; K², K회 탐색)
  cheap     : --ref_sample 5 --search_once                 (K·m, 탐색 1회)
  clean_only: --delta_mode none --ref_sample 5             (탐색 없음: 1단계 싼 검사)
출력: logs/bench/release_map_{fmt}.json + 콘솔 표 (초)
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
PY = sys.executable


def synth_fuba(ck, name, K, R):
    from model import MNISTAutoencoder, Net
    torch.manual_seed(0)
    base = Net().state_dict()
    for r in range(1, R + 1):
        g = {k: v + 0.01 * torch.randn_like(v) for k, v in base.items()}
        locs = [{k: v + 0.02 * torch.randn_like(v) for k, v in g.items()} for _ in range(K)]
        pickle.dump(g, open(f"{ck}/global_net_model_{name}_round_{r}.pkl", "wb"))
        pickle.dump(locs, open(f"{ck}/local_net_model_{name}_round_{r}.pkl", "wb"))
    # 생성기를 저장하지 않으면 release_map 이 ASR 채점을 건너뛴다 (방어 비용만 측정)


def synth_badfu(traj, K, R):
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "BadFU_src", "BadFU-main")))
    from badfu_common import ResNet18
    torch.manual_seed(0)
    base = {k: v.clone() for k, v in ResNet18(pretrained=False).state_dict().items()}
    for r in range(R):
        g = {k: (v + 0.001 * torch.randn_like(v) if torch.is_floating_point(v) else v) for k, v in base.items()}
        cms = [{k: (v + 0.002 * torch.randn_like(v) if torch.is_floating_point(v) else v) for k, v in g.items()} for _ in range(K)]
        torch.save({"gm": g, "cms": cms}, os.path.join(traj, f"round_{r}.pt"))
        torch.save({"gm_after": g}, os.path.join(traj, f"glob_{r}.pt"))


CONFIGS = {
    "full": [],
    "cheap": ["--ref_sample", "5", "--search_once"],
    "clean_only": ["--delta_mode", "none", "--ref_sample", "5"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fmt", choices=["fuba", "badfu_synth"], default="fuba")
    p.add_argument("--Ks", type=int, nargs="+", default=[8, 20, 50, 100])
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--n_gpu", type=int, default=1)
    a = p.parse_args()
    os.makedirs("logs/bench", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="bench_rm_", dir=os.environ.get("TEMP", "/tmp"))
    rows = {}
    try:
        for K in a.Ks:
            if a.fmt == "fuba":
                ck = os.path.join(tmp, f"ck{K}")
                os.makedirs(ck)
                synth_fuba(ck, "bench", K, a.rounds)
                common = ["--fmt", "fuba", "--name", "bench", "--ckpt_dir", ck, "--K", str(K), "--rounds", str(a.rounds)]
            else:
                traj = os.path.join(tmp, f"traj{K}")
                os.makedirs(traj)
                synth_badfu(traj, K, a.rounds)
                common = ["--fmt", "badfu", "--name", "bench", "--traj_dir", traj, "--rounds", str(a.rounds),
                          "--data_root", os.path.join(HERE, "..", "BadFU_src", "BadFU-main", "data"), "--target", "0", "--attackers", "0"]
            rows[K] = {}
            for cfg in a.configs:
                out = os.path.join(tmp, f"out_{K}_{cfg}.json")
                cmd = [PY, "release_map.py", *common, "--requester", "1", "--probe", str(a.probe), "--n_gpu", str(a.n_gpu),
                       "--out", out, *CONFIGS[cfg]]
                t0 = time.time()
                r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                sec = round(time.time() - t0, 1)
                rows[K][cfg] = sec if r.returncode == 0 else f"ERR({r.returncode})"
                print(f"K={K:4d} {cfg:10s} {rows[K][cfg]}s", flush=True)
                if r.returncode != 0:
                    print(r.stderr[-1500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = f"logs/bench/release_map_{a.fmt}.json"
    json.dump({"fmt": a.fmt, "rounds": a.rounds, "probe": a.probe, "rows": rows}, open(out, "w", encoding="utf-8"), indent=2)
    print("\n| K | " + " | ".join(a.configs) + " |")
    print("|---|" + "---|" * len(a.configs))
    for K in a.Ks:
        print(f"| {K} | " + " | ".join(str(rows[K][c]) for c in a.configs) + " |")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
