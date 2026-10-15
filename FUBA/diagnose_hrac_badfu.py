"""Retrospective audit of saved BadFU observations and FU checkpoints.

This is outcome evaluation, never an input to HRAC. No training or FU runs.
The existing FU report describes a single adapted FU step, not full forgetting.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "BadFU_src" / "BadFU-main"))
from request_pipeline import load_tensor, metrics, sha
from badfu_common import ResNet18
from hrac import inter_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "artifacts/badfu/imported_traj_full/manifest.json")
    p.add_argument("--cache", type=Path, default=ROOT / "FUBA/logs/hrac/badfu_traj_full_v1.npz")
    p.add_argument("--fu-report", type=Path, default=ROOT / "FUBA/logs/badfu_request/traj_full_step14_eval_fixed.json")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(8)
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    collected = json.loads(a.cache.with_suffix(".json").read_text(encoding="utf-8"))
    fu = json.loads(a.fu_report.read_text(encoding="utf-8"))
    manifest_hash = sha(a.manifest)
    if not (manifest_hash == collected["source_manifest_sha256"] == fu["manifest_sha256"]):
        raise ValueError("Detection and FU records do not identify the same trajectory")
    if manifest["dataset"]["sha256"] != fu["dataset_sha256"]:
        raise ValueError("Evaluation data differ")
    if collected["source_rounds"] != manifest["rounds"]:
        raise ValueError("Detection round references differ")
    archive = load_tensor(manifest["dataset"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ResNet18(pretrained=False).to(device)
    before = load_tensor(manifest["rounds"][-1]["after"])
    model.load_state_dict(before["gm_after"])
    final_metrics = metrics(model, archive, device, 64)
    step = fu["settings"]["step"]
    if fu["scope"] != "one_actual_step" or step is None:
        raise ValueError("This audit expects the archived single-step FU experiment")
    history = load_tensor(manifest["rounds"][step]["history"])
    model.load_state_dict(history["gm"])
    starting = metrics(model, archive, device, 64)
    reevaluated = {}
    for j in (5, 1):
        previous = fu["results"][f"request_{j}/no_defense"]
        model.load_state_dict(load_tensor(previous["checkpoint"]))
        observed = metrics(model, archive, device, 64)
        for key in ("accuracy", "asr", "clean_target_rate"):
            if abs(observed[key] - previous["candidate_metrics"][key]) > 1e-7:
                raise ValueError(f"Saved metrics not reproduced: request {j}, {key}")
        reevaluated[str(j)] = {
            "metrics": observed, "checkpoint": previous["checkpoint"],
            "asr_change_from_pre_step": observed["asr"] - starting["asr"],
            "asr_change_from_final_fl": observed["asr"] - final_metrics["asr"],
        }
    for observed, previous in ((final_metrics, fu["before_fu"]),
                               (starting, fu["results"]["request_5/no_defense"]["starting_metrics"])):
        for key in ("accuracy", "asr", "clean_target_rate"):
            if abs(observed[key] - previous[key]) > 1e-7:
                raise ValueError(f"Baseline metrics not reproduced: {key}")
    histograms = []
    for j, client in enumerate(archive["clients"]):
        labels = client["labels"]
        count = torch.bincount(labels, minlength=10).numpy()
        top = np.argsort(count)[-2:][::-1]
        histograms.append({"unit": j, "counts": count.tolist(), "largest_classes": top.tolist(),
                           "largest_two_share": float(count[top].sum() / count.sum())})
    clean0 = len(archive["clean_indices"][0])
    same_prefix = all(torch.equal(archive["clients"][0][key][:clean0],
                                 archive["clients"][5][key][:clean0]) for key in ("images", "labels"))
    cache = np.load(a.cache, allow_pickle=False)
    k, t = int(cache["K"]), int(cache["T"])
    h = np.zeros((t, k, 10))
    h[cache["ts"].astype(int)-1, cache["js"].astype(int)] = cache["logprob_vs"].mean(1)
    rows = []
    for j in range(k):
        scores = inter_profile(h, np.eye(k)[j], (0, 10, 20))
        rows.append({"unit": j, "scores_by_output_class": scores.tolist(),
                     "history_sum_by_output_class": h[:, j].sum(0).tolist(),
                     "target0_negative_rounds": int((h[:, j, 0] < 0).sum()),
                     "top_coordinate": int(scores.argmax()), "score": float(scores.max())})
    rows.sort(key=lambda r: -r["score"])
    report = {
        "schema": 1, "mode": "retrospective_saved_checkpoint_evaluation",
        "source_hashes": {str(x): sha(x) for x in (Path(__file__), a.manifest, a.cache, a.fu_report)},
        "matched_detection_and_fu_manifest": True,
        "fu_scope": fu["scope"], "fu_settings": fu["settings"],
        "bn_policy": fu["bn_policy"], "forgetting_verified": False,
        "model_unlearning_calls": 0, "training_calls": 0,
        "original_metrics_reproduced": True,
        "final_fl_metrics": final_metrics, "pre_step_metrics": starting,
        "post_step": reevaluated, "client_label_histograms": histograms,
        "client5_clean_prefix_shared_with_client0": same_prefix,
        "client5_clean_prefix_count": clean0,
        "client5_remaining_cover_count": manifest["counts"][5] - clean0,
        "detection_ranking": rows,
        "limitations": [
            "The saved FU models are an adapted one-step policy, not author full FedEraser or verified forgetting.",
            "The baseline for the step is its pre-round checkpoint; final FL is reported separately.",
            "The same attack-bearing FL history also shows ASR growth after a benign request.",
            "No no-deletion FU-step control is available, so the deletion-specific causal effect is not isolated.",
            "The 512 evaluation inputs are reused retrospective data, not a new independent test set.",
            "Class histograms and positive clean-input target effects do not isolate within-client cover causality.",
            "HRAC rankings are not calibrated detection decisions; intra observations are unavailable.",
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"final_fl": final_metrics, "pre_step": starting,
                      "post_step": reevaluated, "ranking": rows,
                      "histograms": histograms, "clean_prefix": clean0,
                      "shared_prefix": same_prefix}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
