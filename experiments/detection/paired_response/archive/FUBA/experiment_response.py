"""Compare hard/soft deletion-response observations with a fixed protocol.

Uses registered saved trajectories. `matched` reuses previous inputs; `fresh`
uses new public inputs and requester samples excluding earlier probe indices.
Neither protocol adds independent training seeds or authentic BadFU shard FU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from experiment_paired import save
from response_audit import ResponseComparator
from structural_guard import Policy


def run(args):
    import torch
    import torchvision
    from torchvision import transforms
    from run_structural_guard import ContributionBackend
    torch.set_num_threads(4)
    torch.manual_seed(20261014)
    cwd = os.getcwd()
    from release_map import load_fuba, load_badfu
    os.chdir(cwd)
    from existing_guard_components import DoseComparator
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    frozen = json.loads(args.source.read_text(encoding="utf-8"))
    root, cfg = Path(frozen["source_root"]), SimpleNamespace(**frozen["configuration"])
    case = frozen["case"]
    start = time.time()
    factory, state, u, flat, _ = load_badfu(cfg) if case == "badfu" else load_fuba(cfg)
    requests = sorted(set(r["requester"] for r in frozen["rows"]))
    if case == "badfu":
        from badfu_common import BadFUData, TRANSFORM
        data = BadFUData(seed=42)
        if data.counts != cfg.counts:
            raise ValueError("data counts do not match trajectory")
        sets = data.client_sets
        test = torchvision.datasets.CIFAR10(str(root/"BadFU_src/BadFU-main/data"), train=False, download=False, transform=TRANSFORM)
    else:
        tf = transforms.ToTensor()
        train = torchvision.datasets.MNIST(str(root/"FUBA/data"), train=True, download=False, transform=tf)
        test = torchvision.datasets.MNIST(str(root/"FUBA/data"), train=False, download=False, transform=tf)
        sizes = [len(train)//cfg.K+(j < len(train)%cfg.K) for j in range(cfg.K)]
        sets = torch.utils.data.random_split(train, sizes, generator=torch.Generator().manual_seed(522))
    print(f"loaded {case} in {time.time()-start:.1f}s", flush=True)
    policy = Policy()
    n = frozen["probe_n"]
    results = {"case": case, "source": str(args.source.resolve()),
               "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
               "date": "2026-10-14", "policy": vars(policy), "rows": [],
               "split_seed": 20261014, "source_hashes": {},
               "limits": ["exploratory; one training seed", "fixed heuristic thresholds, no calibrated FPR",
                          "BadFU data are a client-level surrogate", "detection only, no model release or mitigation"]}
    for name in ("response_audit.py", "experiment_response.py", "existing_guard_components.py"):
        results["source_hashes"][name] = hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    net = factory().to(args.device).eval()
    for protocol in ("matched", "fresh"):
        offset = 0 if protocol == "matched" else n
        public = torch.stack([test[i][0] for i in range(offset, offset+n)])
        for j in requests:
            original = next(r for r in frozen["rows"] if r["requester"] == j)
            idx = original["request_input_indices"]
            if protocol == "fresh":
                available = sorted(set(range(len(sets[j])))-set(idx))
                idx = np.random.default_rng(20261014+j).choice(available, n, replace=False).tolist()
            req = torch.stack([sets[j][int(i)][0] for i in idx])
            backend = ContributionBackend(factory, flat, flat.flatten(state), u,
                                          {"public": public.numpy(), "request": req.numpy()}, args.device)
            cache = {}
            def predict(removal, view):
                key = (tuple(sorted(removal.items())), view)
                if key not in cache:
                    theta = backend.anchor.clone()
                    for unit, fraction in removal.items():
                        theta -= fraction * u[int(unit)]
                    net.load_state_dict(flat.to_sd(theta))
                    with torch.no_grad():
                        cache[key] = torch.cat([net(xs.to(args.device)).softmax(1).cpu()
                                                for xs in backend.probes[view].split(128)]).numpy()
                return cache[key]
            # This lets the old DOSE initializer inspect our already-computed anchor.
            backend.propose = lambda removal: dict(removal)
            backend.predict = predict
            old = DoseComparator(backend, "badfu" if case == "badfu" else "fuba")
            methods = {"hard_heldout": ResponseComparator("hard"), "soft_heldout": ResponseComparator("soft"),
                       "existing_dose": old}
            for rho in frozen["doses"]:
                removal = {str(j): rho}
                row = {"protocol": protocol, "requester": j, "dose": rho,
                       "role_evaluation_only": original["ground_truth_role_evaluation_only"],
                       "public_indices": [offset, offset+n], "request_indices": idx, "methods": {}}
                for name, comparator in methods.items():
                    row["methods"][name] = {v: comparator(predict({}, v), predict(removal, v),
                        [predict({str(peer): rho}, v) for peer in range(cfg.K) if peer != j], policy)
                        for v in backend.views}
                row["public_tv"] = float(abs(predict(removal, "public")-predict({}, "public")).sum(1).mean()/2)
                row["retention_hold"] = row["public_tv"] > policy.retain_budget
                results["rows"].append(row)
                results["elapsed_seconds"] = time.time()-start
                save(args.out, results)
                alarms = {name: any(m["risk"] for m in view.values()) for name, view in row["methods"].items()}
                print(json.dumps({"case": case, "protocol": protocol, "requester": j,
                                  "dose": rho, "alarms": alarms}), flush=True)
            del backend
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
