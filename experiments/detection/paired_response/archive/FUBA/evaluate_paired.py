"""Independent labeled/trigger evaluation of already frozen audit interventions.

No fitting, threshold tuning, or selection occurs here. Ground-truth triggers are
used only AFTER experiment_paired.py has persisted the selected interventions.
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torchvision
from torchvision import transforms
from experiment_paired import save


def evaluate(a):
    torch.set_num_threads(4)
    frozen = json.loads(a.audit.read_text(encoding="utf-8"))
    root = Path(frozen["source_root"])
    cwd = os.getcwd()
    from release_map import load_fuba, load_badfu
    os.chdir(cwd)
    cfg = SimpleNamespace(**frozen["configuration"])
    is_fuba = frozen["case"] == "fuba"
    factory, state, contributions, flat, bd = (load_fuba(cfg) if is_fuba else load_badfu(cfg))
    if is_fuba:
        test = torchvision.datasets.MNIST(str(root/"FUBA/data"), train=False, download=False, transform=transforms.ToTensor())
        target = 8
    else:
        from badfu_common import TRANSFORM, BackdoorDataset
        br = root/"BadFU_src/BadFU-main"
        test = torchvision.datasets.CIFAR10(str(br/"data"), train=False, download=False, transform=TRANSFORM)
        target = 0
    ids = list(range(frozen["probe_n"], frozen["probe_n"]+a.n))
    xs = torch.stack([test[i][0] for i in ids])
    labels = torch.tensor([test[i][1] for i in ids])
    if is_fuba:
        bd = bd.to(a.device).eval()
        with torch.no_grad():
            triggers = torch.cat([torch.clamp(x.to(a.device) + torch.clamp(bd(x.to(a.device)), -.04, .04), -1, 1).cpu()
                                  for x in xs.split(128)])
        mask = labels != target
        triggers = triggers[mask]
        trigger_ids = [i for i, keep in zip(ids, mask.tolist()) if keep]
    else:
        d = torch.load(br/"record/badnet_dataset/pert_result.pt", weights_only=False, map_location="cpu")
        dd = d["bd_test"]["bd_data_container"]["data_dict"]
        # Keep evaluation source inputs outside public probe IDs, with non-target originals.
        keys = [k for k in dd if int(k) >= frozen["probe_n"] and test.targets[int(k)] != target][:a.n]
        subset = {}
        for k in keys:
            info = dict(dd[k])
            old = Path(info["path"])
            if not old.exists():
                resolved = br/"record/badnet_dataset/bd_test_dataset"/old.parent.name/old.name
                if not resolved.exists():
                    raise FileNotFoundError(resolved)
                info["path"] = str(resolved)
            subset[k] = info
        dataset = BackdoorDataset(subset)
        triggers = torch.stack([dataset[i][0] for i in range(len(dataset))])
        trigger_ids = [int(k) for k in keys]
    net = factory().to(a.device).eval()
    def score(sd):
        net.load_state_dict(sd)
        with torch.no_grad():
            clean = torch.cat([net(x.to(a.device)).argmax(1).cpu() for x in xs.split(128)])
            backdoor = torch.cat([net(x.to(a.device)).argmax(1).cpu() for x in triggers.split(128)])
        return {"acc": float((clean == labels).float().mean()),
                "asr_non_target": float((backdoor == target).float().mean()),
                "n_clean": len(xs), "n_trigger_non_target": len(triggers)}
    anchor = flat.flatten(state)
    cache = {}
    def intervention(removals):
        key = tuple(sorted(removals.items()))
        if key not in cache:
            theta = anchor.clone()
            for unit, fraction in removals.items():
                theta -= fraction * contributions[int(unit)]
            cache[key] = score(flat.to_sd(theta))
        return cache[key]
    result = {"audit_file": str(a.audit.resolve()), "audit_sha256": hashlib.sha256(a.audit.read_bytes()).hexdigest(),
              "case": frozen["case"], "target_evaluation_only": target,
              "clean_test_indices": ids, "trigger_test_indices": trigger_ids,
              "anchor": intervention({}), "rows": [],
              "limitations": "Partial-dose audit models, not full deletion; no forgetting certificate; one training seed."}
    for row in frozen["rows"]:
        removal = row["paired"]["request"]
        joint = {**removal, **row["paired"]["partners"]}
        record = {"requester": row["requester"], "dose": row["dose"],
                  "request": intervention(removal), "joint": intervention(joint)}
        result["rows"].append(record)
        print(json.dumps(record), flush=True)
    if is_fuba:
        p = root/"FUBA/checkpoints/global_net_model_retrain_loc_k20paper_s0__req4_benign_ex0123_round_99.pkl"
        if p.exists():
            with p.open("rb") as handle:
                retrained = pickle.load(handle)
            result["existing_exclusion_retrain_reference"] = {
                "path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "metrics": score(retrained),
                "note": "Existing run with known excluded0..4; oracle exclusion reference, not the new mixed partner set; not retrained here."}
    save(a.out, result)
    print(json.dumps({"anchor": result["anchor"], "saved": str(a.out)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    evaluate(a)
