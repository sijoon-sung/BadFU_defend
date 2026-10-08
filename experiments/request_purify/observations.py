"""Forward-only shadow observations. No trigger, target, attack ID or FU calls.

These are candidate measurements, not a validated detector or a purification rule.
The only labelled input is the pre-existing clean auxiliary/validation dataset.
"""
import copy
import hashlib

import torch
from torch import nn

from .core import scores
from .data import object_hash


VERSION = "clean_probe_shadow_v1"
PAIR_FEATURES = ("raw", "clean_residual", "margin", "margin_bn", "representation", "representation_bn")
BN_SUFFIXES = (".running_mean", ".running_var", ".num_batches_tracked")


def remove_clean_axis(updates, gradient, eps=1e-12):
    """Remove each class row's component along an independent clean-loss gradient.

    Unlike pair centering, this never estimates the reference from q or i.
    Zero gradient => identity. It can remove attack signal too: an ablation only.
    """
    denominator = gradient.square().sum(-1, keepdim=True)
    coefficient = (updates * gradient).sum(-1, keepdim=True) / denominator.clamp_min(eps)
    return updates - coefficient * gradient


def class_cosine(left, right, eps=1e-12):
    # [..., classes, probes]; zero effect is missing evidence, not opposition.
    denominator = left.norm(dim=-1) * right.norm(dim=-1)
    value = (left * right).sum(-1) / denominator.clamp_min(eps)
    return value.clamp(-1, 1), denominator > eps


def clean_head_gradient(logits, features, labels):
    """Exact mean CE gradient of the linear head in eval mode; no backward pass."""
    error = logits.softmax(1) - nn.functional.one_hot(labels, logits.shape[1])
    augmented = torch.cat((features, features.new_ones((len(features), 1))), 1)
    return error.T @ augmented / len(labels)


class ProbeObserver:
    def __init__(self, model, dataset, head_prefix, args, source_ids=None):
        if args.probe_size > len(dataset):
            raise ValueError(f"probe-size {args.probe_size} exceeds clean auxiliary size {len(dataset)}")
        # A private model avoids changes to training state, BN buffers, hooks or modes.
        self.model = copy.deepcopy(model).eval()
        self.device, self.batch = args.device, args.batch_size
        self.head = self.model.get_submodule(head_prefix)
        if not isinstance(self.head, nn.Linear):
            raise ValueError("Observation adapter requires a linear classification head")
        gen = torch.Generator().manual_seed(args.probe_seed)
        indices = torch.randperm(len(dataset), generator=gen)[:args.probe_size].tolist()
        items = [dataset[i] for i in indices]
        self.x = torch.stack([item[0] for item in items])
        self.y = torch.tensor([int(item[1]) for item in items])
        width = self.head.in_features
        self.projection = torch.linalg.qr(torch.randn(width, min(width, args.probe_dimensions),
                                                     generator=gen)).Q
        self.bn_keys = [k for k in self.model.state_dict() if k.endswith(BN_SUFFIXES)]
        h = hashlib.sha256()
        for tensor in (self.x, self.y, self.projection):
            h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.contiguous().numpy().tobytes())
        self.manifest = {"version": VERSION, "probe_size": len(indices),
                         "probe_seed": args.probe_seed, "auxiliary_indices": indices,
                         "source_ids": [int(source_ids[i]) for i in indices] if source_ids is not None else None,
                         "probe_and_projection_sha256": h.hexdigest(),
                         "representation_dimensions": self.projection.shape[1],
                         "class_counts": torch.bincount(self.y, minlength=self.head.out_features).tolist(),
                         "bn_control_available": bool(self.bn_keys),
                         "data_source": "fixed subset of held-out clean training auxiliary data",
                         "guard_pool_overlap": True,
                         "guard_note": "Guard retains its original full validation pool; probes are not an independent guard test",
                         "forward_examples": 0, "forward_batches": 0,
                         "backward_passes": 0, "optimizer_steps": 0, "diagnostic_fu_runs": 0}
        self.training = {k: [] for k in PAIR_FEATURES if k != "raw"}
        self.fu_tensors, self.fu_events = [], []

    @torch.no_grad()
    def snapshot(self, state, bn_reference=None):
        # The controlled measurement swaps BN buffers ONLY in the private copy.
        observed = dict(state)
        if bn_reference is not None:
            observed.update({k: bn_reference[k] for k in self.bn_keys})
        self.model.load_state_dict(observed)
        captured = []
        handle = self.head.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach().cpu()))
        outputs = []
        try:
            for x in self.x.split(self.batch):
                outputs.append(self.model(x.to(self.device)).cpu())
                self.manifest["forward_batches"] += 1
                self.manifest["forward_examples"] += len(x)
        finally:
            handle.remove()
        logits, features = torch.cat(outputs), torch.cat(captured)
        if not torch.isfinite(logits).all() or not torch.isfinite(features).all():
            raise ValueError("Nonfinite probe observation")
        margins = []
        for c in range(logits.shape[1]):
            other = logits.clone(); other[:, c] = -torch.inf
            margins.append(logits[:, c] - other.max(1).values)
        # One fixed random projection per probe manifest. Normalization is explicit.
        embedding = nn.functional.normalize(features, dim=1) @ self.projection
        return {"margin": torch.stack(margins), "probability": logits.softmax(1).T,
                "representation": embedding.reshape(1, -1),
                "gradient": clean_head_gradient(logits, features, self.y)}

    def deltas(self, start, local):
        reference = self.snapshot(start)
        native = [self.snapshot(state) for state in local]
        controlled = [self.snapshot(state, start) for state in local] if self.bn_keys else native
        effects = {}
        for feature in ("margin", "representation"):
            effects[feature] = torch.stack([s[feature] - reference[feature] for s in native])
            effects[feature + "_bn"] = torch.stack([s[feature] - reference[feature] for s in controlled])
        return reference, effects

    def training_round(self, start, local, head_updates):
        reference, effects = self.deltas(start, local)
        effects["clean_residual"] = remove_clean_axis(head_updates, reference["gradient"])
        for name, value in effects.items():
            self.training[name].append(value)

    def training_result(self, final):
        final_observation = self.snapshot(final)
        return {"manifest": dict(self.manifest),
                "features": {name: torch.stack(values) for name, values in self.training.items()},
                "final": final_observation}

    def fu_round(self, start, local, retained_ids, applied, history_observations, round_index, requester):
        reference, effects = self.deltas(start, local)
        final = self.snapshot(applied)
        final_bn = self.snapshot(applied, start) if self.bn_keys else final
        original = history_observations["final"]
        event = {"round": round_index + 1, "actual_requester": requester,
                 "retained_ids": list(retained_ids), "phase": "actual_FU_after_guard",
                 "alarm": None, "alarm_note": "Uncalibrated shadow measurements; no FU alarm or intervention"}
        for suffix, observed in (("", final), ("_bn", final_bn)):
            shift = observed["margin"] - reference["margin"]
            event["step_margin_rms" + suffix] = shift.square().mean(-1).sqrt().tolist()
            event["step_margin_mean" + suffix] = shift.mean(-1).tolist()
            event["step_representation_rms" + suffix] = float((observed["representation"] - reference["representation"]).square().mean().sqrt())
            historical = history_observations["features"]["margin" + suffix][round_index]
            # [all possible observed q, retained client, class]. Actual FU q never changes.
            cosine, valid = class_cosine(historical[:, None], effects["margin" + suffix][None])
            event["request_effect_cosine" + suffix] = cosine.tolist()
            event["request_effect_valid" + suffix] = valid.tolist()
            event["retained_margin_rms" + suffix] = effects["margin" + suffix].square().mean(-1).sqrt().tolist()
        event["margin_rms_from_original_final"] = (final["margin"] - original["margin"]).square().mean(-1).sqrt().tolist()
        event["probability_mean"] = final["probability"].mean(-1).tolist()
        event["probe_prediction_counts"] = torch.bincount(final["probability"].argmax(0), minlength=final["probability"].shape[0]).tolist()
        self.fu_tensors.append({"round": round_index + 1, "local_effects": effects,
                                "start": reference, "applied": final, "applied_bn": final_bn})
        self.fu_events.append(event)
        return event


def candidate_scores(history, cfg):
    """Return every q and every class; no target or attacker selection."""
    observation = history["observations"]
    if observation["manifest"]["version"] != VERSION:
        raise ValueError("Unsupported observation version")
    features = {"raw": history["heads"], **observation["features"]}
    weights = history["weights"]
    result = {}
    for name in PAIR_FEATURES:
        values = features[name]
        by_request = []
        for q in range(values.shape[1]):
            score, raw, coverage = scores(values, weights, q, cfg)
            window = values[-cfg.window:] * weights[-cfg.window:, :, None, None]
            cosine, usable = class_cosine(window, window[:, q:q + 1])
            usable[:, q] = False
            count = usable.sum(0)
            denominator = count.clamp_min(1)
            by_request.append({"requester": q, "score": float(score.max()),
                               "client_class_scores": score.tolist(), "coverage": coverage.tolist(),
                               "usable_class_counts": count.tolist(),
                               "mean_cosine": ((cosine * usable).sum(0) / denominator).tolist(),
                               "negative_cosine_fraction": (((cosine < 0) & usable).sum(0) / denominator).tolist(),
                               "mean_raw_score": (raw.sum(0) / coverage.clamp_min(1)[:, None]).tolist(),
                               "persistent_rounds": ((raw >= cfg.round_threshold) & usable).sum(0).tolist()})
        result[name + "_pair"] = {"uses_request_effect": True, "requests": by_request}
    # Request-effect-independent comparisons: include all clients so changing q
    # alone cannot alter these values. They are abnormalities, not risk proof.
    for name in ("margin", "representation"):
        values = features[name][-cfg.window:]
        rms = values.square().mean(-1).sqrt().mean(0)
        result[name + "_magnitude"] = {"uses_request_effect": False,
            "requests": [{"requester": q, "score": float(rms.max()), "client_class_scores": rms.tolist()}
                         for q in range(values.shape[1])]}
    return result


def fit_policy(histories, cfg, quantile=.99):
    if not histories or not 0 < quantile <= 1:
        raise ValueError("Nonempty normal histories and a valid quantile are required")
    protocols = {h["manifest"]["protocol_signature"] for h in histories}
    if len(protocols) != 1 or any(not h["manifest"]["benign"] for h in histories):
        raise ValueError("Observation calibration requires benign histories under one protocol")
    sources = [{"trajectory_id": h["manifest"]["trajectory_id"], "seed": h["manifest"]["seed"]} for h in histories]
    if len({s["seed"] for s in sources}) != len(sources):
        raise ValueError("Repeated calibration seeds are not independent")
    reports = [candidate_scores(h, cfg) for h in histories]
    candidates = {}
    for name in reports[0]:
        # Max over q within a trajectory: calibration sample unit is one seed.
        values = [max(r["score"] for r in report[name]["requests"]) for report in reports]
        tau = max(float(torch.quantile(torch.tensor(values, dtype=torch.float64), quantile)), 1e-12)
        candidates[name] = {"threshold": tau, "normal_trajectory_maxima": values}
    result = {"version": VERSION, "protocol_signature": next(iter(protocols)), "sources": sources,
              "quantile": quantile, "candidates": candidates,
              "calibration_unit": "one normal seed; maximum over requesters, clients and classes",
              "warning": "Empirical thresholds only; few independent seeds do not establish a population FPR. Candidate thresholds are separate, not a calibrated OR ensemble."}
    result["policy_sha256"] = object_hash(result)
    return result


def apply_policy(history, cfg, policy):
    meta = history["manifest"]
    if meta["protocol_signature"] != policy["protocol_signature"]:
        raise ValueError("Observation policy protocol mismatch")
    if meta["seed"] in {s["seed"] for s in policy["sources"]}:
        raise ValueError("Calibration/evaluation seed leakage")
    report = candidate_scores(history, cfg)
    for name, candidate in report.items():
        threshold = policy["candidates"][name]["threshold"]
        for request in candidate["requests"]:
            request.update({"threshold": threshold, "alarm": request["score"] > threshold})
    return {"version": VERSION, "seed": meta["seed"], "benign": meta["benign"],
            "actual_requester": meta["requester"], "trajectory_id": meta["trajectory_id"],
            "policy_sha256": policy["policy_sha256"], "mode": "shadow_only",
            "candidates": report, "diagnostic_fu_runs": 0,
            "request_ablation_note": "Only the observation input q changes. Actual deletion is unchanged; other q values have no counterfactual post-FU ground truth."}
