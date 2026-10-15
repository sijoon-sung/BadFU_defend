"""Replay a ledger or audit FUBA/BadFU checkpoints with the common policy.

The checkpoint adapter uses approximate contribution subtraction. It intentionally
provides neither a forgetting certificate nor a retraining implementation, so its
output can be audited_only / held / inconclusive, never released / recovered.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from structural_guard import CancellationMonitor, Policy, StructuralGuard


class ContributionBackend:
    """Existing PyTorch models behind the attack-independent protocol.

    Arrays in probes.npz are already transformed model INPUTS, not labels or
    attack success scores. `request` must be supplied from authorized deletion
    inputs; the adapter never substitutes attack ground truth or a public view.
    """

    def __init__(self, factory, flat, theta, contributions, probes, device="cpu", batch_size=128):
        import torch
        self.torch, self.factory, self.flat = torch, factory, flat
        self.anchor = theta.detach().cpu().clone()
        self.contributions = [u.detach().cpu().clone() for u in contributions]
        if any(u.shape != self.anchor.shape or not torch.isfinite(u).all() for u in self.contributions):
            raise ValueError("contribution vectors must be finite and match the anchor")
        if not torch.isfinite(self.anchor).all():
            raise ValueError("anchor must be finite")
        self.units = tuple(str(i) for i in range(len(contributions)))
        self.views = tuple(sorted(probes))
        self.probes = {}
        for name, xs in probes.items():
            xs = np.asarray(xs)
            if xs.ndim < 2 or len(xs) == 0 or not np.isfinite(xs).all():
                raise ValueError(f"{name}: expected finite, nonempty batched inputs")
            self.probes[name] = torch.as_tensor(xs, dtype=torch.float32).cpu()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.device, self.batch_size = device, batch_size

    def propose(self, removals):
        theta = self.anchor.clone()
        for unit, fraction in removals.items():
            theta -= float(fraction) * self.contributions[int(unit)]
        return theta

    def predict(self, model, view):
        net = self.factory()
        net.load_state_dict(self.flat.to_sd(model))
        net.to(self.device).eval()
        results = []
        with self.torch.no_grad():
            xs = self.probes[view]
            for start in range(0, len(xs), self.batch_size):
                logits = net(xs[start:start + self.batch_size].to(self.device))
                results.append(logits.softmax(dim=1).cpu())
        return self.torch.cat(results).numpy()


def load_backend(args):
    # Historical helper imports change cwd. Keep the new entry point location-safe.
    original_cwd = os.getcwd()
    try:
        from release_map import load_fuba, load_badfu
    finally:
        os.chdir(original_cwd)
    if args.fmt == "badfu":
        if not args.traj_dir:
            raise ValueError("badfu requires --traj_dir")
        if not args.counts and not args.uniform_weights:
            raise ValueError("badfu requires actual --counts or explicit --uniform_weights")
        if args.counts and (min(args.counts) <= 0 or sum(args.counts) <= 0):
            raise ValueError("counts must be positive")
    if args.rounds < (2 if args.fmt == "fuba" else 1) or args.K <= 0:
        raise ValueError("invalid rounds or K")
    factory, state, contributions, flat, _evaluation_only = (
        load_fuba(args) if args.fmt == "fuba" else load_badfu(args))
    if len(contributions) != args.K:
        raise ValueError("--K differs from the trajectory client count")
    with np.load(args.probes, allow_pickle=False) as archive:
        probes = {key: archive[key] for key in archive.files}
    return ContributionBackend(factory, flat, flat.flatten(state), contributions,
                               probes, args.device, args.batch_size)


def replay_cache(path, threshold=0.8, persist=3, screen="structural"):
    """Accept existing class-mean caches, or richer vs[event, input, class] arrays."""
    with np.load(path, allow_pickle=False) as cache:
        k, t = int(cache["K"]), int(cache["T"])
        ts, js, vs = cache["ts"], cache["js"], cache["vs"]
    if k <= 0 or t <= 0 or ts.ndim != 1 or js.ndim != 1 or vs.ndim < 2:
        raise ValueError("invalid ledger dimensions")
    if len(ts) != len(js) or len(ts) != len(vs) or not np.isfinite(vs).all():
        raise ValueError("invalid ledger events")
    if not np.isin(ts, np.arange(1, t + 1)).all() or not np.isin(js, np.arange(k)).all():
        raise ValueError("round or client IDs outside cache dimensions")
    if len(set(zip(ts.tolist(), js.tolist()))) != len(ts):
        raise ValueError("duplicate client-round event")
    monitor, alarms = CancellationMonitor(threshold=threshold, persist=persist), []
    ledger, n_present, streak = np.zeros((k, vs.shape[-1])), np.zeros(k, int), {}
    if screen not in ("ccd", "structural"):
        raise ValueError("unknown screening method")
    if screen == "ccd" and vs.ndim != 2:
        raise ValueError("original CCD requires class-mean vectors; use --screen structural for tensors")
    from ledger_rules import judge
    ccd = SimpleNamespace(min_present=2, neg_set=False, tau=3.0, kappa=2.0,
                          h0=threshold, min_pos=2, max_share=0.6)
    for r in range(1, t + 1):
        mask = ts == r
        if screen == "structural":
            alarms.extend(monitor.observe(r, {str(int(j)): v for j, v in zip(js[mask], vs[mask])}))
        else:
            for j, v in zip(js[mask], vs[mask]):
                ledger[int(j)] += v
                n_present[int(j)] += 1
            if r < 5:
                continue
            verdicts = judge(ledger, n_present, ccd)
            streak = {c: streak.get(c, 0) + 1 if v["coalition"] else 0 for c, v in verdicts.items()}
            for c, v in verdicts.items():
                if streak[c] >= persist:
                    alarms.append({"units": sorted(str(j) for j in set(v["negs"] + v["pos"])),
                                   "coordinate": [int(c)], "score": v["H"], "round": r,
                                   "action": "audit_priority"})
    return {"cache": str(Path(path).resolve()), "K": k, "T": t,
            "screen": screen,
            "feature_shape": list(vs.shape[1:]), "candidates": alarms,
            "n_candidates": len(alarms), "action": "request_audit_required",
            "note": "Screening only. No candidates does not imply safe deletion."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    monitor = sub.add_parser("monitor", help="NumPy-only replay of a ledger cache")
    monitor.add_argument("--cache", required=True)
    monitor.add_argument("--threshold", type=float, default=0.8)
    monitor.add_argument("--persist", type=int, default=3)
    monitor.add_argument("--screen", choices=["ccd", "structural"], default="ccd",
                         help="ccd: original ledger rules; structural: experimental single/multiple-leg screen")
    monitor.add_argument("--out", required=True)
    audit = sub.add_parser("audit", help="checkpoint audit; approximate FU, no release certificate")
    audit.add_argument("--fmt", choices=["fuba", "badfu"], required=True)
    audit.add_argument("--name", required=True)
    audit.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    audit.add_argument("--rounds", type=int, required=True)
    audit.add_argument("--K", type=int, required=True)
    audit.add_argument("--ckpt_dir", default="checkpoints")
    audit.add_argument("--traj_dir")
    audit.add_argument("--counts", type=int, nargs="+")
    audit.add_argument("--uniform_weights", action="store_true")
    audit.add_argument("--probes", required=True, help="NPZ: public and request input tensors, transformed for this model")
    audit.add_argument("--requesters", nargs="+", required=True, help="one or more client indices")
    audit.add_argument("--fraction", type=float, default=1.0)
    audit.add_argument("--step", type=float, default=0.2)
    audit.add_argument("--device", default="cpu")
    audit.add_argument("--batch_size", type=int, default=128)
    audit.add_argument("--max_controls", type=int, default=32)
    audit.add_argument("--components", choices=["existing", "distribution", "response"], default="existing",
                       help="existing: DOSE + TRACE; distribution: experimental multi-statistic; response: experimental held-out soft flow + TRACE")
    audit.add_argument("--stat", choices=["flow", "mass", "changed", "shift"], default="flow")
    audit.add_argument("--paired_confirm", action="store_true",
                       help="experimental held-out joint-removal review; can add a hold, never clear an existing alarm")
    audit.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    output = Path(args.out).resolve()
    if args.command == "monitor":
        result = replay_cache(args.cache, args.threshold, args.persist, args.screen)
    else:
        if len(set(args.requesters)) != len(args.requesters):
            parser.error("requesters must be unique")
        for name in ("ckpt_dir", "traj_dir", "probes"):
            if getattr(args, name):
                setattr(args, name, str(Path(getattr(args, name)).resolve()))
        policy = Policy(max_step=args.step, max_controls=args.max_controls)
        backend = load_backend(args)
        components = {}
        if args.components == "existing":
            # A missing view stays inconclusive; do not initialize a legacy engine
            # with fabricated replacement inputs.
            if "public" in backend.views:
                from existing_guard_components import DoseComparator, trace_attributor
                components = {"comparator": DoseComparator(backend, args.fmt, stat=args.stat),
                              "attributor": trace_attributor}
        elif args.components == "response":
            from response_audit import ResponseComparator
            from existing_guard_components import trace_attributor
            components = {"comparator": ResponseComparator(), "attributor": trace_attributor}
        if args.paired_confirm:
            from paired_audit import paired_audit
            components["confirmation"] = paired_audit
        guard = StructuralGuard(backend, policy, **components)
        result = asdict(guard.request(dict.fromkeys(args.requesters, args.fraction)))
        result.update({"fmt": args.fmt, "name": args.name, "policy": asdict(policy),
                       "components": args.components,
                       "operator": "approximate_contribution_subtraction",
                       "coverage": {"observed_views": list(backend.views), "trigger_coverage": "unverified"},
                       "limitation": "No full retraining or independent forgetting verification in this adapter."})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result.get("status", result.get("action")), "out": str(output)}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
