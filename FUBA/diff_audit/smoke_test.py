# -*- coding: utf-8 -*-
"""코드 점검용: 무작위 체크포인트와 무작위 이미지로 run 이 끝까지 도는지만 본다 (결과 숫자는 의미 없음).

FUBA 폴더에서:
    python -m diff_audit.smoke_test
"""
import os
import pickle
import tempfile

import torch

from model import Net, MNISTAutoencoder
from diff_audit import run


def make_fake_ckpt(d, name, rounds, K):
    torch.manual_seed(0)
    for r in range(1, rounds + 1):
        pickle.dump(Net().state_dict(), open(os.path.join(d, f"global_net_model_{name}_round_{r}.pkl"), "wb"))
        pickle.dump([Net().state_dict() for _ in range(K)],
                    open(os.path.join(d, f"local_net_model_{name}_round_{r}.pkl"), "wb"))
    pickle.dump(MNISTAutoencoder().state_dict(),
                open(os.path.join(d, f"backdoor_net_model_{name}_round_{rounds}.pkl"), "wb"))


def main():
    with tempfile.TemporaryDirectory() as d:
        make_fake_ckpt(d, "smoke", rounds=3, K=4)
        for inv in ("universal", "per_sample", "patch"):
            out = run.main(["--name", "smoke", "--ckpt_dir", d, "--rounds", "3", "--K", "4",
                            "--atk", "0", "1", "--req", "2", "--target", "8", "--probe", "200",
                            "--inv", inv, "--steps", "3", "--alphas", "2", "--smoke",
                            "--dis_modes", "universal", "per_sample",
                            "--out", os.path.join(d, f"smoke_{inv}.json")])
            assert out["summary"]["methods"], inv
    print("smoke ok")


if __name__ == "__main__":
    main()
