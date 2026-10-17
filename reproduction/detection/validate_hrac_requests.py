"""Apply the SAME HRAC request gate to exported BadFU or FUBA observations.

Only small evidence files are needed. An independent request/atom bundle enables
intra. No independent request means unknown, never a manufactured zero residual.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproduction.detection import validate_detection as common
from reproduction.detection.hrac_request_gate import evaluate, fingerprint


def load_evidence(folder):
    inventory = common.read(folder / "inventory.json")
    for name, expected in inventory.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or common.probes.sha(path) != expected:
            raise ValueError(f"Evidence hash mismatch: {name}")
    training = common.read(folder / "training_protocol.json")
    attack = training["options"]["attack"]
    if attack == "badfu":
        p = common.read(folder / "full6/protocol.json")
        manifest = folder / "full6/manifest.json"
        arrays = folder / "full6/response_means.npz"
        manifest_hash = p["manifest_sha256"]
    elif attack == "fuba":
        p = common.read(folder / "observation_protocol.json")
        manifest = folder / "manifest.json"
        arrays = folder / "response_means.npz"
        manifest_hash = p["source_manifest_sha256"]
    else:
        raise ValueError("Supported evidence: BadFU or FUBA")
    if common.probes.sha(manifest) != manifest_hash:
        raise ValueError("Observation/trajectory manifest mismatch")
    with np.load(arrays, allow_pickle=False) as data:
        raw = data["responses"]
        mask = data["participation"] if attack == "fuba" else np.ones(raw.shape[:1]+raw.shape[2:3], bool)
    if (raw.ndim != 5 or raw.shape[-1] != 10 or raw.shape[3] != len(p["views"])
            or raw.shape[1] != len(p["probe_seeds"]) or not np.isfinite(raw).all()
            or len(raw) != len(common.read(manifest)["rounds"])
            or mask.shape != (len(raw), raw.shape[2])):
        raise ValueError("Invalid compact observation layout")
    return raw[:, :, :, 1:] - raw[:, :, :, :1], mask, p, attack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True, help="extracted evidence directory")
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="new analysis directory")
    args = parser.parse_args()
    h, mask, protocol, attack = load_evidence(args.evidence.resolve())
    observation_id = fingerprint(protocol)
    if args.requests:
        request_data = common.read(args.requests)
        if request_data.get("schema") != 1 or request_data.get("observation_id") != observation_id:
            raise ValueError("Request bundle must bind to this exact observation_id")
        requests = request_data["requests"]
    else:
        requests = [dict(id=f"client:{j}", round=len(h), removal=np.eye(h.shape[2])[j].tolist())
                    for j in range(h.shape[2]) if mask[:, j].any()]
    result = evaluate(h, requests, observation_id)
    args.out.mkdir(parents=True, exist_ok=False)
    common.probes.save_json(args.out / "request_coordinates.json", dict(
        observation_id=observation_id, dimensions=h.shape[1]*h.shape[3]*h.shape[4],
        flatten_order=["probe repetition", "perturbed view (exclude clean)", "output"],
        manifest_sha256=protocol.get("manifest_sha256", protocol.get("source_manifest_sha256")),
        note="No atoms/request effects are fabricated. See RUN.md for the independent request data contract."))
    result.update(attack=attack, input_inventory_sha256=common.probes.sha(args.evidence / "inventory.json"),
                  request_input_sha256=None if args.requests is None else common.probes.sha(args.requests),
                  observation_id=observation_id, training_calls=0, model_unlearning_calls=0,
                  code_sha256={name: common.probes.sha(ROOT/name) for name in (
                      "reproduction/detection/validate_hrac_requests.py", "reproduction/detection/hrac_request_gate.py", "FUBA/research/hrac.py", "FUBA/research/tension_audit.py")})
    common.probes.save_json(args.out / "hrac.json", result)
    print(f"{attack}: inter observed; intra observed={result['both_channels_observed']}; decision={result['decision']['status']}")
    print(args.out / "hrac.json")


if __name__ == "__main__":
    main()
