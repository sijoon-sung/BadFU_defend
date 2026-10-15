"""Derive ablations/figures from frozen JSON; never fits an audit threshold."""
import hashlib
import json
from pathlib import Path
import numpy as np
from experiment_paired import save


def main():
    directory = Path(__file__).resolve().parent/"logs/paired_audit"
    load = lambda name: json.loads((directory/name).read_text(encoding="utf-8"))
    cases = {name: load(f"{name}_local.json") for name in ("fuba", "badfu", "clean")}
    evaluations = {name: load(f"{name}_evaluation.json") for name in ("fuba", "badfu")}
    syn = load("synthetic.json")
    summary = {"generated_from_frozen_artifacts": True, "cases": {}, "synthetic": syn["summary"],
               "synthetic_rank_test": {k: syn["calibration"][k] for k in ("n_bank", "n_test", "alpha", "false_alerts")},
               "source_sha256": {}}
    for file in ("paired_audit.py", "trace_solver.py", "experiment_paired.py", "evaluate_paired.py",
                 "existing_guard_components.py", "structural_guard.py", "run_structural_guard.py",
                 "tests/test_paired_audit.py"):
        p = Path(__file__).resolve().parent/file
        summary["source_sha256"][file] = hashlib.sha256(p.read_bytes()).hexdigest()
    # Snapshot hashes are generated after the runs. They describe these saved
    # reproducibility sources, not a committed release or new training run.
    summary["source_hash_note"] = "post-run working-source snapshot; no git commit created"
    for name, data in cases.items():
        records = []
        for row in data["rows"]:
            audit = row["paired"]
            flagged = audit["status"] == "review_priority"
            dose = any(m.get("risk", False) for m in row["existing_dose"].values())
            energy = any(m["energy_z"] >= 3 and m["request_energy"] > 1e-6 for m in audit["views"].values())
            records.append({"requester": row["requester"], "dose": row["dose"],
                            "evaluation_role": row["ground_truth_role_evaluation_only"],
                            "DOSE_alarm": dose, "energy_only_alarm": energy,
                            "paired_alarm": flagged, "DOSE_or_paired_alarm": dose or flagged,
                            "linear_fit_ge_half": audit["discovery_linear_restoration"] >= .5,
                            "paired_status": audit["status"],
                            "public_only_alarm": row["public_only"]["status"] == "review_priority",
                            "partners": audit["partners"]})
        summary["cases"][name] = {"rows": records, "elapsed_seconds": data["elapsed_seconds"],
                                  "training_seeds": 1, "n_audits": len(records)}
    save(directory/"summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    fuba = [r for r in evaluations["fuba"]["rows"] if r["requester"] == 4]
    x = [r["dose"] for r in fuba]
    ax[0].plot(x, [100*r["request"]["asr_non_target"] for r in fuba], "o-", label="Request only")
    ax[0].plot(x, [100*r["joint"]["asr_non_target"] for r in fuba], "s-", label="Joint intervention")
    ax[0].axhline(100*evaluations["fuba"]["anchor"]["asr_non_target"], color="gray", linestyle="--", label="Anchor")
    ax[0].set(title="FUBA: diagnostic models, one seed", xlabel="Removal fraction", ylabel="ASR (%)", ylim=(0, 105))
    ax[0].legend(fontsize=8)
    badfu = [r for r in evaluations["badfu"]["rows"] if r["requester"] == 5]
    ax[1].plot(x, [100*r["request"]["asr_non_target"] for r in badfu], "o-", color="tab:orange", label="Request = joint (no partner)")
    ax[1].axhline(100*evaluations["badfu"]["anchor"]["asr_non_target"], color="gray", linestyle="--", label="Anchor")
    ax[1].set(title="BadFU surrogate: pairing failed", xlabel="Removal fraction", ylabel="ASR (%)", ylim=(0, 105))
    ax[1].legend(fontsize=8)
    modes = ["cancel", "nonlinear", "overfit", "honest_opposition"]
    ix = np.arange(len(modes))
    ax[2].bar(ix-.18, [syn["summary"][m]["linear_fit_ge_half"] for m in modes], .36, label="Discovery fit >= 0.5")
    ax[2].bar(ix+.18, [syn["summary"][m]["review_priority"] for m in modes], .36, label="Actual + held-out check")
    ax[2].set_xticks(ix, ["Cancel", "Nonlinear", "Overfit", "Honest\nopposition"], fontsize=8)
    ax[2].set(title="Synthetic mechanism checks", ylabel="Positive / 50 episodes", ylim=(0, 60))
    ax[2].legend(fontsize=7)
    for item in ax:
        item.grid(alpha=.2, axis="y")
    fig.savefig(directory/"paired_results.png", dpi=180)
    fig.savefig(directory/"paired_results.pdf")
    print(json.dumps({"audits": sum(d["n_audits"] for d in summary["cases"].values()),
                      "synthetic_episodes": 300+199+200, "summary": str(directory/"summary.json")}))


if __name__ == "__main__":
    main()
