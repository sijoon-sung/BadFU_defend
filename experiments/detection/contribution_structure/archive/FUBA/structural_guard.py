"""Attack-independent, observation-based guard for dormant FU backdoors.

This is an experimental policy, not a detector with a proved false-positive rate.
No attack labels, target labels, or assumed coalition sizes enter the policy.
Backends own the FU operator; this module owns comparisons and release state.
"""
from dataclasses import dataclass, field
from itertools import permutations
import math
from typing import Any, Callable, Mapping, Protocol

import numpy as np


FEATURES = ("js_mean", "tv_tail", "target_mass", "class_flow")


@dataclass(frozen=True)
class Policy:
    z_limit: float = 3.0
    scale_floor: float = 0.005
    min_effect: float = 0.02
    retain_budget: float = 0.15
    max_step: float = 0.2
    min_controls: int = 3
    max_controls: int = 32
    tail_fraction: float = 0.1
    required_views: tuple = ("public", "request")
    attribution_gain: float = 0.15
    seed: int = 522

    def __post_init__(self):
        for name in ("z_limit", "scale_floor", "min_effect", "retain_budget",
                     "max_step", "tail_fraction", "attribution_gain"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_step", "tail_fraction", "retain_budget", "attribution_gain"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1")
        if self.min_controls < 3 or self.max_controls < self.min_controls:
            raise ValueError("require 3 <= min_controls <= max_controls")
        if not self.required_views or len(set(self.required_views)) != len(self.required_views):
            raise ValueError("required_views must be nonempty and unique")


def probabilities(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("predictions must have shape (nonempty inputs, classes >= 2)")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("predictions must be finite probabilities")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("each prediction must sum to one")
    return p


def features(before, after, tail_fraction=0.1):
    """Distribution change, local change, targeted mass, and multi-source flow."""
    p, q = probabilities(before), probabilities(after)
    if p.shape != q.shape:
        raise ValueError("prediction shapes differ")
    eps = 1e-12
    mid = (p + q) / 2
    js = 0.5 * (p * np.log((p + eps) / (mid + eps))
                + q * np.log((q + eps) / (mid + eps))).sum(axis=1)
    tv = np.abs(q - p).sum(axis=1) / 2
    n_tail = max(1, math.ceil(len(tv) * tail_fraction))
    tail = float(np.partition(tv, len(tv) - n_tail)[-n_tail:].mean())
    mass = float(np.maximum(q - p, 0).mean(axis=0).max())
    source, dest = p.argmax(axis=1), q.argmax(axis=1)
    changed = source != dest
    c = p.shape[1]
    flow = np.bincount(source[changed] * c + dest[changed], minlength=c*c)
    flow = flow.reshape(c, c) / len(p)
    col = flow.sum(axis=0)
    j = int(col.argmax())
    src = flow[:, j] / max(float(col[j]), eps)
    entropy = float(-(src * np.log(src + eps)).sum()) / math.log(max(c - 1, 2))
    return np.array([float(js.mean()), tail, mass, float(col[j]) * max(0, entropy)])


def compare(before, candidate, controls, policy):
    """Same-input, same-operator controls; scores are heuristics, not p-values."""
    if len(controls) < policy.min_controls:
        raise ValueError("insufficient matched controls")
    actual = features(before, candidate, policy.tail_fraction)
    null = np.stack([features(before, p, policy.tail_fraction) for p in controls])
    med = np.median(null, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(null - med), axis=0), policy.scale_floor)
    z = (actual - med) / scale
    flagged = (z > policy.z_limit) & ((actual - med) > policy.min_effect)
    return {
        "risk": bool(flagged.any()),
        "features": dict(zip(FEATURES, actual.tolist())),
        "control_median": dict(zip(FEATURES, med.tolist())),
        "z": dict(zip(FEATURES, z.tolist())),
        "flagged": [name for name, flag in zip(FEATURES, flagged) if flag],
    }


class Backend(Protocol):
    """Units may be clients OR disjoint deletion shards, including same-client shards.

    propose always starts from an immutable anchor; it must not mutate that anchor.
    A unit is a deletion object, not a presumed adversarial role. Matched controls
    must be scientifically comparable (data size/exposure/partition/operator).
    """
    units: tuple
    views: tuple

    def propose(self, removals: Mapping[str, float]) -> Any: ...
    def predict(self, model: Any, view: str) -> np.ndarray: ...


@dataclass(frozen=True)
class Verification:
    """Evidence from independent, predeclared checks, outside discovery probes."""
    forgetting: bool
    retention: bool
    independent_audit: bool
    evidence: str

    @property
    def passed(self):
        return (self.forgetting is True and self.retention is True
                and self.independent_audit is True and bool(self.evidence.strip()))


@dataclass
class Decision:
    status: str
    requested: dict
    committed: dict
    steps: list = field(default_factory=list)
    suspects: list = field(default_factory=list)
    reason: str = ""
    verification: dict = field(default_factory=dict)


def matched_controls(units, removals, policy):
    """Match the WHOLE cumulative request, including cardinality and fractions.

    Matching fraction does not establish exchangeability. Backends/callers must
    restrict units to an appropriate peer pool when exposures differ materially.
    """
    available = sorted(set(units) - set(removals))
    doses = [removals[k] for k in sorted(removals)]
    q = len(doses)
    if q == 0 or len(available) < q:
        return []
    total = math.perm(len(available), q)
    if total <= policy.max_controls:
        groups = list(permutations(available, q))
    else:
        rng = np.random.default_rng(policy.seed)
        groups = set()
        for _ in range(policy.max_controls * 20):
            groups.add(tuple(rng.choice(available, q, replace=False).tolist()))
            if len(groups) >= policy.max_controls:
                break
        groups = sorted(groups)
    # Equal doses make permutations identical: do not count duplicates as evidence.
    unique = {tuple(sorted(zip(group, doses))) for group in groups}
    return [dict(items) for items in sorted(unique)]


class StructuralGuard:
    """Observe -> intervene -> attribute -> recover -> independently verify.

    verify(model, cumulative_removals, suspects) returns Verification.
    recover(cumulative_removals, suspects) returns a reconstructed model. It must
    honor all deletions and must not distill from a teacher retaining forgotten data.
    Without these callbacks, the guard audits but never declares deletion complete.
    """

    def __init__(self, backend: Backend, policy=None, verify: Callable | None = None,
                 recover: Callable | None = None, comparator: Callable | None = None,
                 attributor: Callable | None = None, confirmation: Callable | None = None):
        self.backend, self.policy = backend, policy or Policy()
        if not backend.units or any(not isinstance(u, str) or not u for u in backend.units):
            raise ValueError("units must be nonempty string identifiers")
        if len(set(backend.units)) != len(backend.units):
            raise ValueError("unit identifiers must be unique")
        if len(set(backend.views)) != len(backend.views):
            raise ValueError("view identifiers must be unique")
        self.verify, self.recover = verify, recover
        self.comparator, self.attributor = comparator or compare, attributor
        self.confirmation = confirmation
        self.committed = {}
        self.recovered = False
        self.serving_model = backend.propose({})
        self.anchor_probs = {v: probabilities(backend.predict(self.serving_model, v)).copy()
                             for v in backend.views}

    def audit(self, model, removals):
        """Also callable after subsequent training, using appropriately refreshed controls."""
        missing = sorted(set(self.policy.required_views) - set(self.backend.views))
        if missing:
            return {"status": "inconclusive", "reason": f"missing views: {missing}", "views": {}}
        controls = matched_controls(self.backend.units, removals, self.policy)
        if len(controls) < self.policy.min_controls:
            return {"status": "inconclusive", "reason": "insufficient distinct matched controls", "views": {}}
        pred = {v: probabilities(self.backend.predict(model, v)) for v in self.backend.views}
        control_pred = {v: [] for v in self.backend.views}
        for removal in controls:
            control = self.backend.propose(removal)
            for v in self.backend.views:
                control_pred[v].append(self.backend.predict(control, v))
        views = {v: self.comparator(self.anchor_probs[v], pred[v], control_pred[v], self.policy)
                 for v in self.backend.views}
        if any(v.get("inconclusive", False) for v in views.values()):
            return {"status": "inconclusive", "reason": "comparison_evidence_unavailable", "views": views}
        public_change = 0.0
        if "public" in pred:
            public_change = float(np.abs(pred["public"] - self.anchor_probs["public"]).sum(axis=1).mean() / 2)
        collapsed = public_change > self.policy.retain_budget
        risk = any(v["risk"] for v in views.values())
        paired = self.confirmation(self.backend, removals) if self.confirmation is not None else None
        # An unconfirmed pair must NEVER veto a positive existing detector.
        paired_risk = paired is not None and paired["status"] == "review_priority"
        if paired is not None and paired["status"] == "inconclusive" and not (risk or collapsed):
            return {"status": "inconclusive", "reason": "paired_confirmation_unavailable",
                    "views": views, "paired_confirmation": paired, "n_controls": len(controls),
                    "public_tv": public_change}
        risk = risk or paired_risk
        return {"status": "hold" if risk or collapsed else "pass_probes",
                "reason": "retain_budget" if collapsed else ("paired_review" if paired_risk else
                          "activation" if risk else "within_probe_envelope"),
                "views": views, "n_controls": len(controls), "public_tv": public_change,
                **({"paired_confirmation": paired} if paired is not None else {})}

    def attribute(self, removals, model):
        """Greedy counterfactual localization; candidates are NOT confirmed attackers.

        Find retained units whose joint removal reduces anomalous changes across
        observed views. This supports one/many carriers and multiple targets.
        Exclusion always requires independent recovery verification afterwards.
        """
        if self.attributor is not None:
            return self.attributor(self.backend, self.anchor_probs, removals, model, self.policy)

        def distance(candidate):
            return max(features(self.anchor_probs[v], self.backend.predict(candidate, v),
                                self.policy.tail_fraction)[1] for v in self.backend.views)

        current = distance(model)
        original = current
        selected, trials = [], dict(removals)
        for _ in range(len(self.backend.units) - len(removals)):
            choices = []
            for unit in sorted(set(self.backend.units) - set(trials)):
                intervention = {**trials, unit: 1.0}
                score = distance(self.backend.propose(intervention))
                choices.append((score, unit))
            if not choices:
                break
            best, unit = min(choices)
            gain = (current - best) / max(original, 1e-12)
            if gain < self.policy.attribution_gain:
                break
            selected.append({"unit": unit, "gain": float(gain)})
            trials[unit], current = 1.0, best
        return selected

    def request(self, increments):
        """Transactional deletion. Failed/partial proposals never replace the serving model.

        The common anchor and cumulative removals survive split and sequential
        requests. No reference reset to an already partly activated model occurs.
        """
        if self.recovered:
            return Decision("inconclusive", {}, dict(self.committed),
                            reason="refresh_backend_after_recovery")
        if not increments or set(increments) - set(self.backend.units):
            raise ValueError("request must contain known deletion units")
        target = dict(self.committed)
        for unit, amount in increments.items():
            if not np.isfinite(amount) or not 0 < amount <= 1:
                raise ValueError("increments must be finite and in (0, 1]")
            target[unit] = target.get(unit, 0) + float(amount)
            if target[unit] > 1 + 1e-9:
                raise ValueError("request exceeds remaining contribution")
            target[unit] = min(target[unit], 1.0)
        decision = Decision("inconclusive", target, dict(self.committed))
        n_steps = math.ceil(max(increments.values()) / self.policy.max_step)
        candidate = None
        for i in range(1, n_steps + 1):
            removal = {unit: self.committed.get(unit, 0) + (total - self.committed.get(unit, 0)) * i / n_steps
                       for unit, total in target.items()}
            candidate = self.backend.propose(removal)
            audit = self.audit(candidate, removal)
            decision.steps.append({"removals": removal, **audit})
            if audit["status"] == "inconclusive":
                decision.reason = audit["reason"]
                return decision
            if audit["status"] == "hold":
                decision.suspects = self.attribute(removal, candidate)
                decision.status, decision.reason = "held", audit["reason"]
                if self.recover is None:
                    return decision
                # Recover the FULL requested deletion, not only the interrupted dose.
                suspects = tuple(s["unit"] for s in decision.suspects)
                candidate = self.recover(dict(target), suspects)
                recovered_audit = self.audit(candidate, target)
                decision.steps.append({"removals": dict(target), "recovery": True, **recovered_audit})
                if recovered_audit["status"] != "pass_probes":
                    decision.reason = "recovery_failed_audit"
                    return decision
                return self._finish(candidate, target, decision, suspects, recovered=True)
        return self._finish(candidate, target, decision, (), recovered=False)

    def _finish(self, candidate, target, decision, suspects, recovered):
        if self.verify is None:
            decision.status, decision.reason = "audited_only", "independent_verification_unavailable"
            return decision
        proof = self.verify(candidate, dict(target), suspects)
        if not isinstance(proof, Verification):
            raise TypeError("verifier must return Verification")
        decision.verification = vars(proof).copy()
        if not proof.passed:
            decision.status, decision.reason = "held", "independent_verification_failed"
            return decision
        self.committed = dict(target)
        self.serving_model = candidate
        self.recovered = recovered
        decision.committed = dict(target)
        decision.status = "recovered" if recovered else "released"
        decision.reason = "independent_checks_passed"
        return decision


class CancellationMonitor:
    """Optional online screen on ANY signed functional vectors; never a release gate.

    Unlike a class-mean ledger this can retain input, class, and view dimensions.
    A stable cancelling group is only an audit priority, not proof of malice.
    """

    def __init__(self, threshold=0.8, min_energy=0.01, persist=3):
        if not 0 < threshold <= 1 or not np.isfinite(min_energy) or min_energy <= 0 or persist < 1:
            raise ValueError("invalid monitor settings")
        self.threshold, self.min_energy, self.persist = threshold, min_energy, persist
        self.ledger, self.streak = {}, {}
        self.last_round = -1
        self.shape = None

    def observe(self, round_id, contributions):
        if not isinstance(round_id, int) or round_id <= self.last_round:
            raise ValueError("round IDs must be strictly increasing integers")
        values = {str(k): np.asarray(v, dtype=float) for k, v in contributions.items()}
        shape = self.shape or (next(iter(values.values())).shape if values else None)
        if any(v.ndim == 0 or v.shape != shape or v.size == 0 or not np.isfinite(v).all() for v in values.values()):
            raise ValueError("functional vectors must be finite, nonempty, and have fixed shape")
        if len(values) != len(contributions):
            raise ValueError("unit identifiers collide")
        if round_id != self.last_round + 1:
            self.streak = {}
        self.last_round, self.shape = round_id, shape
        for unit, value in values.items():
            self.ledger[unit] = self.ledger.get(unit, np.zeros_like(value)) + value
        candidates = {}
        units = sorted(self.ledger)
        if units:
            matrix = np.stack([self.ledger[j].reshape(-1) for j in units])
            # Keep each coordinate: averaging across inputs/classes can erase a
            # local or multi-target cancellation. Either side may contain 1..K units.
            for coordinate, column in enumerate(matrix.T):
                gross = float(np.abs(column).sum())
                if gross < self.min_energy:
                    continue
                score = 1 - abs(float(column.sum())) / gross
                median = float(np.median(column))
                scale = max(1.4826 * float(np.median(np.abs(column - median))), self.min_energy)
                negative = [units[i] for i, value in enumerate(column)
                            if value < 0 and value < median - 3 * scale]
                positive = [units[i] for i, value in enumerate(column)
                            if value > 0 and value > median + 2 * scale]
                # With very small cohorts robust ranks cannot separate two legs;
                # the resulting candidate is still only a request-audit priority.
                if len(units) < 4:
                    negative = [units[i] for i, value in enumerate(column) if value < -self.min_energy]
                    positive = [units[i] for i, value in enumerate(column) if value > self.min_energy]
                group = tuple(sorted(negative + positive))
                # Partial participation is legitimate. Require fresh evidence from
                # at least one member, not simultaneous participation by everyone.
                if negative and positive and score >= self.threshold and set(group) & set(values):
                    candidates[(coordinate, group)] = score
        self.streak = {group: self.streak.get(group, 0) + 1 for group in candidates}
        return [{"units": list(key[1]), "coordinate": [int(i) for i in np.unravel_index(key[0], self.shape)],
                 "score": float(candidates[key]),
                 "round": round_id, "action": "audit_priority"}
                for key in sorted(candidates) if self.streak[key] >= self.persist]
