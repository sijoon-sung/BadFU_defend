"""Standalone request-probe experiment on recorded FUBA trajectories or toy demo."""
import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.request_probe import METHODS, SearchConfig, evaluate, freeze, search, split_pool
from reproduction.trajectory import check_state, json_write, sha256, verify


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_candidate(run, clients, dose):
    """Weighted historic parameter subtraction; final BN buffers stay fixed.

This is NOT a counterfactual retraining or an implementation of actual FU.
Only full-client deletion is supported, not within-client sample requests.
"""
    from FUBA.model import Net, resnet18
    run = Path(run)
    protocol = read(run / "protocol.json")
    options = protocol["options"]
    manifest = read(run / "fl/manifest.json")
    if (protocol.get("status") != "complete" or options.get("attack") != "fuba"
            or options.get("dataset") not in ("mnist", "cifar10")
            or manifest["metadata"].get("attack") != "fuba"
            or manifest["metadata"].get("aggregation") != "fuba_uniform"
            or manifest["metadata"].get("dataset") != options["dataset"]
            or manifest["metadata"].get("num_clients") != options["clients"]):
        raise ValueError("complete recorded FUBA MNIST/CIFAR10 run required")
    if (not clients or len(set(clients)) != len(clients)
            or any(i < 0 or i >= options["clients"] for i in clients)):
        raise ValueError("unique in-range client IDs required")
    if not np.isfinite(dose) or dose <= 0:
        raise ValueError("finite positive subtraction dose required")
    print("Checking the complete trajectory and recorded aggregates...", flush=True)
    integrity = verify(run / "fl")
    if not manifest["rounds"]:
        raise ValueError("at least one recorded round required")
    model = Net() if options["dataset"] == "mnist" else resnet18()
    names = {name for name, _ in model.named_parameters()}
    accumulated = {name: torch.zeros_like(p, device="cpu") for name, p in model.named_parameters()}
    counts = {i: 0 for i in clients}
    for entry in manifest["rounds"]:
        path = run / "fl" / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("checkpoint changed after verification")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not np.allclose(state["weights"], 1 / len(state["cms"]), rtol=0, atol=1e-12):
            raise ValueError("recorded FUBA weights must be uniform over participants")
        for client, weight, local in zip(state["client_ids"], state["weights"], state["cms"]):
            if client in counts:
                counts[client] += 1
                for name in names:
                    accumulated[name].add_(local[name] - state["gm"][name], alpha=weight)
        final = state["gm_after"]
    if any(n == 0 for n in counts.values()):
        raise ValueError("requested client has no recorded participation")
    model.load_state_dict(final, strict=True)
    candidate = copy.deepcopy(model)
    with torch.no_grad():
        for name, parameter in candidate.named_parameters():
            parameter.sub_(accumulated[name], alpha=dose)
    check_state(candidate.state_dict(), final)
    provenance = dict(dataset=options["dataset"], request_clients=clients,
        request_unit="whole_client", dose=dose, participation=counts,
        candidate_kind="historical_parameter_subtraction_surrogate",
        candidate_formula="final - dose * sum_t sum_requested weight_i,t * (local_i,t - pre_t)",
        buffer_policy="final FL buffers copied unchanged; eval mode; no recalibration",
        accumulated_parameter_l2=float(sum((v.double() ** 2).sum() for v in accumulated.values()).sqrt()),
        manifest_sha256=sha256(run / "fl/manifest.json"),
        training_protocol_sha256=sha256(run / "protocol.json"), integrity=integrity)
    return model, candidate, provenance


class DemoModel(torch.nn.Module):
    """An explicitly planted cancellation example, not trained FL or FUBA."""
    def __init__(self, released, threshold=.50):
        super().__init__()
        self.released = released
        self.threshold = threshold

    def forward(self, x):
        clean = (x[:, 0, :4, :4].mean((1, 2)) - .5) * 6
        trigger = torch.relu(x[:, 0, 5:, 5:].mean((1, 2)) - self.threshold) * 30
        return torch.stack((clean, -clean, -torch.ones_like(clean) + self.released * trigger), dim=1)


def run_experiment(pre, candidate, images, ids, out, config, search_size, evaluation_size,
                   methods, provenance, actual=None):
    config.validate()
    a, b = split_pool(images, ids, search_size, evaluation_size, config.seed)
    device = provenance["device"]
    chosen = images[np.concatenate((a, b))].copy()
    # Canonical NCHW strides also for C=1. Tensor.contiguous() alone can retain
    # ambiguous channels-last strides and break the upstream MNIST Net.view().
    x = torch.from_numpy(chosen.transpose(0, 3, 1, 2).copy()).float().to(device) / 255
    search_x, evaluation_x = x[:len(a)], x[len(a):]
    pre, candidate = freeze(pre, device), freeze(candidate, device)
    actual = freeze(actual, device) if actual is not None else None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema="request-probe-v1", status="running", config=asdict(config),
        methods=list(methods), search_ids=ids[a].tolist(), evaluation_ids=ids[b].tolist(),
        provenance=provenance, labels_used=False, attack_target_used=False,
        attack_generator_used=False, model_training_calls=0, actual_fu_calls=0,
        actual_fu_checkpoint_evaluated=actual is not None,
        preprocessing="uint8 NHWC -> float32 NCHW / 255; no normalization",
        selection="all classes; restarts and steps selected on search split only",
        source_sha256={name: sha256(ROOT / name) for name in (
            "reproduction/run_request_probe.py", "reproduction/request_probe.py",
            "reproduction/trajectory.py", "reproduction/validate_fuba_detection.py",
            "FUBA/model/__init__.py", "FUBA/model/model.py", "FUBA/model/resnet.py")})
    json_write(out / "protocol.json", protocol)
    rows, arrays = {}, {"evaluation_ids": ids[b], "search_ids": ids[a]}
    for method in methods:
        print(f"Searching {method}; evaluation split stays sealed until selection finishes...", flush=True)
        start = time.perf_counter()
        result = search(pre, candidate, search_x, method, config)
        result["search_seconds"] = time.perf_counter() - start
        metrics, values = evaluate(pre, candidate, evaluation_x, result, config.batch_size)
        row = {k: v for k, v in result.items() if k != "delta"}
        row["surrogate_evaluation"] = metrics
        arrays[method + "__delta"] = result["delta"].cpu().numpy()
        arrays.update({method + "__surrogate__" + k: v for k, v in values.items()})
        if actual is not None:
            row["actual_fu_evaluation"], values = evaluate(pre, actual, evaluation_x, result, config.batch_size)
            arrays.update({method + "__actual_fu__" + k: v for k, v in values.items()})
        rows[method] = row
        json_write(out / "scores.json", rows)
    with (out / "observations.npz").open("xb") as f:
        np.savez_compressed(f, **arrays)
    protocol["status"] = "complete"
    protocol["output_sha256"] = {name: sha256(out / name) for name in ("scores.json", "observations.npz")}
    json_write(out / "protocol.json", protocol)
    lines = ["# Request-conditioned probe experiment", "",
        "Research scores only. No attack verdict, calibrated FPR, accuracy, or ASR is inferred.", "",
        f"Candidate: {provenance['candidate_kind']}",
        f"Actual FU checkpoint evaluated: {actual is not None}", "",
        "| Method | Selected class | Surrogate release margin | Actual FU release margin |",
        "| --- | ---: | ---: | ---: |"]
    for method, row in rows.items():
        real = row.get("actual_fu_evaluation", {}).get("release_margin_mean")
        lines.append(f"| {method} | {row['target']} | {row['surrogate_evaluation']['release_margin_mean']:.6f} | "
                     + (f"{real:.6f}" if real is not None else "not measured") + " |")
    lines += ["", "The same split must not be reused for tuning and confirmatory testing.",
        "Compare independent benign and attack runs; images are not independent training runs.",
        "Fixed corner/random controls have no optimization steps; optimized controls share the search budget."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--run", type=Path, help="recorded FUBA run with protocol.json and fl/manifest.json")
    parser.add_argument("--pool", type=Path, help="unlabeled NPZ with uint8 NHWC images and unique string ids")
    parser.add_argument("--request-clients", nargs="+", type=int)
    parser.add_argument("--fu-state", type=Path, help="optional actual-FU full state_dict; evaluation only")
    parser.add_argument("--out", type=Path, required=True, help="new output directory; never overwrite")
    parser.add_argument("--dose", type=float, default=1.)
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=2/255)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--search-size", type=int, default=80)
    parser.add_argument("--evaluation-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    config = SearchConfig(args.epsilon, args.steps, args.step_size, args.restarts, args.batch_size, args.seed)
    config.validate()
    if args.out.exists() or args.threads < 1 or len(set(args.methods)) != len(args.methods):
        parser.error("new --out, positive threads and unique methods required")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    actual = None
    if args.demo:
        if any(v is not None for v in (args.run, args.pool, args.request_clients, args.fu_state)):
            parser.error("--demo cannot be combined with real inputs")
        pre, candidate = DemoModel(0), DemoModel(1)
        rng = np.random.default_rng(args.seed)
        images = rng.integers(40, 210, (args.search_size + args.evaluation_size, 8, 8, 1), dtype=np.uint8)
        images[:, 5:, 5:] = rng.integers(125, 140, (len(images), 3, 3, 1), dtype=np.uint8)
        ids = np.array([f"synthetic:{i}" for i in range(len(images))])
        provenance = dict(candidate_kind="synthetic_known_release_demo_NOT_FL", dataset="synthetic")
    else:
        if args.run is None or args.pool is None or args.request_clients is None:
            parser.error("--run, --pool and --request-clients are required without --demo")
        pre, candidate, provenance = build_candidate(args.run, args.request_clients, args.dose)
        from reproduction.validate_fuba_detection import load_pool
        images, ids = load_pool(args.pool, provenance["dataset"])
        provenance["pool_sha256"] = sha256(args.pool)
        if args.fu_state is not None:
            state = torch.load(args.fu_state, map_location="cpu", weights_only=True)
            check_state(state, pre.state_dict())
            actual = copy.deepcopy(pre)
            actual.load_state_dict(state, strict=True)
            provenance["actual_fu_state_sha256"] = sha256(args.fu_state)
            provenance["actual_fu_provenance"] = "user-supplied; request correspondence must be independently checked"
    provenance.update(device=device, threads=args.threads, torch=str(torch.__version__),
        hardware=torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU")
    run_experiment(pre, candidate, images, ids, args.out, config, args.search_size,
                   args.evaluation_size, args.methods, provenance, actual)
    print(f"Completed: {args.out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
