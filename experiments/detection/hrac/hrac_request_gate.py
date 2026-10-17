"""Two-channel history/request assessment, independent of attack names.

Inter uses the recorded history and requested deletion fractions. Intra needs
separately measured request effects and within-owner atoms in common coordinates.
Missing evidence is not a benign verdict. No model is unlearned here.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "FUBA"))
from experiments.detection.hrac.hrac import attribution_residual, decide, inter_profile


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def evaluate(history, requests, observation_id, *, windows=(0, 10, 20)):
    """history: [round, probe repetition, client, perturbed view, output].

    The entries are paired positive-update responses, already weighted by the
    actual aggregate. Removal fractions describe client-level deletion units.
    Partial deletion inter effects are a declared linear attribution surrogate.
    Within-client subsets require independent atoms for the intra channel.
    """
    h = np.asarray(history, float)
    if h.ndim != 5 or 0 in h.shape or not np.isfinite(h).all():
        raise ValueError("finite [round, repetition, client, view, output] history required")
    t, reps, k, views, classes = h.shape
    rows, identifiers = [], set()
    if not requests:
        raise ValueError("at least one declared request required")
    for item in requests:
        ident = item["id"]
        if not isinstance(ident, str) or not ident or ident in identifiers:
            raise ValueError("unique nonempty request IDs required")
        identifiers.add(ident)
        stop = item.get("round", t)
        removal = np.asarray(item["removal"], float)
        if (not isinstance(stop, int) or not 1 <= stop <= t or removal.shape != (k,)
                or not np.isfinite(removal).all() or np.any(removal < 0) or np.any(removal > 1)
                or not removal.any()):
            raise ValueError("valid request round and client deletion fractions in [0,1] required")
        inter = max(float(inter_profile(h[:stop, rep, :, view], removal, windows).max())
                    for rep in range(reps) for view in range(views))
        intra, status = None, "unobserved_independent_request_effect"
        if "intra" in item:
            data = item["intra"]
            meta = data["provenance"]
            if (meta.get("independent_observation") is not True
                    or meta.get("observation_id") != observation_id
                    or meta.get("request_round") != stop
                    or not meta.get("history_source") or not meta.get("request_source")
                    or meta["history_source"] == meta["request_source"]):
                raise ValueError("intra needs independently sourced observations at the declared round in identical coordinates")
            atoms = np.asarray(data["atoms"], float)
            observed = np.asarray(data["observed"], float)
            if atoms.ndim != 2 or atoms.shape[1] != reps * views * classes or observed.shape != (atoms.shape[1],):
                raise ValueError("intra coordinates must flatten [probe repetition, perturbed view, output]")
            owners = np.asarray(data["atom_owners"])
            if (owners.shape != (len(atoms),) or owners.dtype.kind not in "iu"
                    or np.any(owners < 0) or np.any(owners >= k)):
                raise ValueError("each historical atom requires a valid client owner")
            # Feasibility must come from request metadata, not be fitted to U.
            lower, upper = np.asarray(data["lower"], float), np.asarray(data["upper"], float)
            if lower.shape != (len(atoms),) or upper.shape != lower.shape:
                raise ValueError("request-derived atom bounds required")
            if np.any(upper[removal[owners] == 0] != 0):
                raise ValueError("request cannot attribute effects to another owner's atoms")
            if (meta.get("constraints_source") != "request_metadata"
                    or meta.get("complete_within_requested_owners") is not True):
                raise ValueError("bounds/mass constraints must come from request metadata")
            masses = np.asarray(data["atom_masses"], float)
            if masses.shape != (len(atoms),) or not np.isfinite(masses).all() or np.any(masses <= 0):
                raise ValueError("positive historical atom masses required")
            owners_requested = np.flatnonzero(removal)
            if any(not np.any(owners == j) for j in owners_requested):
                raise ValueError("missing historical atoms for a requested owner")
            mass_rows = np.stack([masses * (owners == j) for j in owners_requested])
            mass_values = mass_rows.sum(-1) * removal[owners_requested]
            if ("constraint_matrix" in data) != ("constraint_value" in data):
                raise ValueError("both additional equality matrix and value required")
            if "constraint_matrix" in data:
                mass_rows = np.vstack([mass_rows, np.asarray(data["constraint_matrix"], float)])
                mass_values = np.concatenate([mass_values, np.asarray(data["constraint_value"], float)])
            kwargs = dict(constraint_matrix=mass_rows, constraint_value=mass_values)
            if "covariance" in data:
                kwargs["covariance"] = data["covariance"]
            intra = attribution_residual(atoms, observed, lower=lower, upper=upper, ridge=.001, **kwargs)
            status = "measured_feasibility_residual"
        rows.append(dict(id=ident, round=stop, removal=removal.tolist(), inter=inter,
                         intra=intra, intra_status=status,
                         decision=decide({}, {"inter": inter, "intra": None if intra is None else intra["score"]})))
    scores = {"inter": max(r["inter"] for r in rows),
              "intra": max(r["intra"]["score"] for r in rows) if all(r["intra"] is not None for r in rows) else None}
    return dict(requests=rows, episode_scores=scores, decision=decide({}, scores),
                both_channels_observed=all(r["intra"] is not None for r in rows),
                missing_channel_is_not_zero=True,
                provenance_note="Source and independence declarations are auditable metadata, not cryptographic proof.",
                interpretation="Consistency/conditional risk, not proof of intent. A malicious but feasible request can have zero intra residual.")
