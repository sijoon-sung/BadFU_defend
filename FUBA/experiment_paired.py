"""Reproducible paired-confirmation experiments; role labels used ONLY in reporting.

Synthetic checks are mechanistic fixtures, never attack-performance evidence.
Real experiments reuse saved training trajectories and perform no new training.
All probe labels, trigger generators and known roles are excluded from the audit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from paired_audit import PairedPolicy, paired_audit, rank_calibrate


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def synthetic(out):
    from tests.test_paired_audit import FunctionalFixture
    modes = ["clean", "cancel", "localized", "nonlinear", "overfit", "honest_opposition"]
    rows = []
    for mode in modes:
        for seed in range(50):
            result = paired_audit(FunctionalFixture(mode, seed), {"11": .2})
            rows.append({"mode": mode, "seed": seed, "status": result["status"],
                         "score": result["score"], "linear_fit": result["discovery_linear_restoration"],
                         "confirmed_restore": result["views"]["request"]["restoration"]})
    # The null generator is fixed before looking at scores. Calibration and test
    # are independent generated EPISODES, not input rows of a shared model.
    bank = [paired_audit(FunctionalFixture("clean", s), {"11": .2})["score"] for s in range(1000, 1199)]
    test = [paired_audit(FunctionalFixture("clean", s), {"11": .2})["score"] for s in range(2000, 2200)]
    calibrated = [rank_calibrate(x, bank) for x in test]
    summary = {m: {"n": 50, "review_priority": sum(r["status"] == "review_priority" for r in rows if r["mode"] == m),
                          "linear_fit_ge_half": sum(r["linear_fit"] >= .5 for r in rows if r["mode"] == m)} for m in modes}
    data = {"kind": "synthetic_mechanism_only", "policy": vars(PairedPolicy()), "summary": summary, "rows": rows,
            "calibration": {"n_bank": len(bank), "n_test": len(test), "alpha": .05,
                            "false_alerts": sum(x["reject"] for x in calibrated),
                            "bank": bank, "test_scores": test, "p_values": [x["p"] for x in calibrated]},
            "limitations": "Constructed linear/nonlinear fixtures; honest opposition is indistinguishable; no ASR claim."}
    save(out, data)
    print(json.dumps({"summary": summary, "calibration_false_alerts": data["calibration"]["false_alerts"]}), flush=True)


def real(args):
    import torch
    import torchvision
    from torchvision import transforms
    from run_structural_guard import ContributionBackend
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    np.random.seed(20261013)
    root = Path(args.artifact_root).resolve()
    output = args.out.resolve()
    original = os.getcwd()
    from release_map import load_fuba, load_badfu
    os.chdir(original)
    start = time.time()
    n = args.probe_n
    if args.case in ("fuba", "clean"):
        k, rounds, name = (20, 100, "loc_k20paper_s0") if args.case == "fuba" else (8, 8, "loc_clean_k8")
        cfg = SimpleNamespace(K=k, rounds=rounds, name=name, dataset="mnist", ckpt_dir=str(root/"FUBA/checkpoints"))
        factory, state, u, flat, bd = load_fuba(cfg)
        tf = transforms.ToTensor()
        train = torchvision.datasets.MNIST(str(root/"FUBA/data"), train=True, download=False, transform=tf)
        test = torchvision.datasets.MNIST(str(root/"FUBA/data"), train=False, download=False, transform=tf)
        # Match the training IID split; use an explicit generator to avoid RNG side effects.
        sizes = [len(train)//k + (i < len(train)%k) for i in range(k)]
        partitions = torch.utils.data.random_split(train, sizes, generator=torch.Generator().manual_seed(522))
        requests = [4, 5, 19] if args.case == "fuba" else list(range(k))
        request_sets = {j: partitions[j] for j in requests}
        final_path = root/f"FUBA/checkpoints/global_net_model_{name}_round_{rounds}.pkl"
        artifact_paths = [root/f"FUBA/checkpoints/global_net_model_{name}_round_{r}.pkl" for r in range(1, rounds+1)]
        artifact_paths += [root/f"FUBA/checkpoints/local_net_model_{name}_round_{r}.pkl" for r in range(2, rounds+1)]
        role = {4: "suppressor", 5: "benign", 19: "benign"} if args.case == "fuba" else {j: "benign" for j in requests}
        data_note = "IID MNIST split seed522; authorized requester training samples; public disjoint from evaluation"
    else:
        br = root/"BadFU_src/BadFU-main"
        cfg = SimpleNamespace(K=6, rounds=15, counts=[10545,9647,9641,9649,9668,10545], traj_dir=str(br/"logs/traj_full"))
        factory, state, u, flat, bd = load_badfu(cfg)
        from badfu_common import BadFUData, TRANSFORM
        data = BadFUData(seed=42)
        if data.counts != cfg.counts:
            raise ValueError("training counts mismatch")
        test = torchvision.datasets.CIFAR10(str(br/"data"), train=False, download=False, transform=TRANSFORM)
        requests, role = [5, 1, 2], {5: "camouflage_requester", 1: "benign", 2: "benign"}
        # Requester5's full client mixes clean+camouflage; evaluate this actual
        # deletion view, plus a separate explicit cv-only authorized shard view.
        request_sets = {j: data.client_sets[j] for j in requests}
        final_path = br/"logs/traj_full/glob_14.pt"
        artifact_paths = [br/f"logs/traj_full/round_{r}.pt" for r in range(15)] + [final_path]
        data_note = "BadFU ul 6-client surrogate, NOT original same-client shard unlearning; seeded uniform client-input sample"
    print(f"loaded {args.case} in {time.time()-start:.1f}s", flush=True)
    def tensor_sample(dataset, count, seed):
        idx = np.random.default_rng(seed).choice(len(dataset), min(count, len(dataset)), replace=False)
        return torch.stack([dataset[int(i)][0] for i in idx]), idx.tolist()
    public = torch.stack([test[i][0] for i in range(n)])
    ex = torch.stack([test[i][0] for i in range(n, n+args.eval_n)])
    ey = np.array([test[i][1] for i in range(n, n+args.eval_n)])
    rows = []
    manifest = {"kind": "saved_trajectory_audit_no_training", "case": args.case,
                "source_root": str(root), "configuration": vars(cfg), "probe_n": n,
                "eval_n": args.eval_n, "device": args.device, "torch": torch.__version__,
                "input_protocol": data_note, "policy": vars(PairedPolicy()),
                "doses": args.doses, "training_seeds": 1, "checkpoint_manifest": [],
                "limitations": ["heuristic score; no independent benign-episode calibration",
                                "contribution subtraction is approximate, not a forgetting certificate",
                                "requests/doses/views of one trajectory are correlated"], "rows": rows}
    for p in artifact_paths:
        manifest["checkpoint_manifest"].append({"path": str(p), "bytes": p.stat().st_size,
                                                 "mtime_ns": p.stat().st_mtime_ns})
    manifest["final_checkpoint_sha256"] = hashlib.sha256(final_path.read_bytes()).hexdigest()
    for j in requests:
        req, idx = tensor_sample(request_sets[j], n, 20261013+j)
        probes = {"public": public.numpy(), "request": req.numpy()}
        backend = ContributionBackend(factory, flat, flat.flatten(state), u, probes, args.device, 128)
        # Reuse one network and memoize output arrays across dose/view ablations;
        # input indices and all four actual interventions stay unchanged.
        net = factory().to(args.device).eval()
        pred_cache = {}
        backend.propose = lambda removal: dict(removal)
        def predict(removal, view):
            key = (tuple(sorted(removal.items())), view)
            if key not in pred_cache:
                theta = backend.anchor.clone()
                for unit, dose in removal.items():
                    theta -= dose * backend.contributions[int(unit)]
                net.load_state_dict(flat.to_sd(theta))
                xs = backend.probes[view] if view != "evaluation" else ex
                with torch.no_grad():
                    pp = [net(xs[t:t+128].to(args.device)).softmax(1).cpu() for t in range(0, len(xs), 128)]
                pred_cache[key] = torch.cat(pp).numpy()
            return pred_cache[key]
        backend.predict = predict
        # Existing DOSE uses exactly the same model outputs and control pool.
        from existing_guard_components import DoseComparator
        from structural_guard import Policy
        comparator = DoseComparator(backend, "badfu" if args.case == "badfu" else "fuba")
        for dose in args.doses:
            removal = {str(j): dose}
            both = paired_audit(backend, removal)
            public_only = paired_audit(backend, removal, views=("public",))
            baseline = {}
            for v in backend.views:
                controls = [predict({str(peer): dose}, v) for peer in range(len(u)) if peer != j]
                baseline[v] = comparator(predict({}, v), predict(removal, v), controls, Policy())
            joint = {**removal, **both["partners"]}
            evaluation = {"anchor_acc": float((predict({}, "evaluation").argmax(1) == ey).mean()),
                          "request_acc": float((predict(removal, "evaluation").argmax(1) == ey).mean()),
                          "joint_acc": float((predict(joint, "evaluation").argmax(1) == ey).mean()),
                          "ASR": "not measured; no backdoor-removal claim"}
            row = {"requester": j, "ground_truth_role_evaluation_only": role[j], "dose": dose,
                   "request_input_indices": idx, "paired": both, "public_only": public_only,
                   "existing_dose": baseline, "evaluation": evaluation}
            rows.append(row)
            manifest["elapsed_seconds"] = time.time()-start
            save(output, manifest)
            print(json.dumps({"requester": j, "dose": dose, "paired": both["status"], "partners": both["partners"],
                              "restore": both["views"]["request"]["restoration"], "evaluation": evaluation}), flush=True)
        del net, backend
        torch.cuda.empty_cache()
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["synthetic", "fuba", "clean", "badfu"], required=True)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--probe-n", type=int, default=256)
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--doses", type=float, nargs="+", default=[.05, .1, .2])
    a = p.parse_args()
    if a.case == "synthetic":
        synthetic(a.out.resolve())
    else:
        if a.artifact_root is None:
            p.error("--artifact-root required for real data")
        real(a)
