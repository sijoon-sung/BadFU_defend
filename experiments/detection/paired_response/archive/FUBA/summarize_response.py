"""Summarize frozen response audits without pooling probes into training seeds."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(folder):
    output = {"date": "2026-10-14", "rows": [], "artifacts": {}, "source_versions": {}}
    details = []
    for case in ("fuba", "badfu", "clean"):
        path = folder / f"{case}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        output["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        output["source_versions"][case] = data["source_hashes"]
        for protocol in ("matched", "fresh"):
            for role in sorted({r["role_evaluation_only"] for r in data["rows"]}):
                rows = [r for r in data["rows"] if r["protocol"] == protocol and r["role_evaluation_only"] == role]
                summary = {"case": case, "protocol": protocol, "role": role, "n": len(rows),
                           "retention_holds": sum(r["retention_hold"] for r in rows)}
                for method in ("existing_dose", "hard_heldout", "soft_heldout"):
                    summary[method] = sum(any(v["risk"] for v in r["methods"][method].values()) for r in rows)
                    summary[method+"_unknown"] = sum(any(v.get("inconclusive", False) for v in r["methods"][method].values()) for r in rows)
                output["rows"].append(summary)
        for row in data["rows"]:
            details.append({k: row[k] for k in ("protocol", "requester", "dose", "role_evaluation_only", "public_tv", "retention_hold")} |
                           {"case": case, "methods": row["methods"]})
    output["n_audits"] = sum(r["n"] for r in output["rows"])
    output["independent_saved_training_trajectories"] = 3
    output["limits"] = ["matched/fresh inputs share trained models", "no population FPR or significance claim",
                        "retention holds are reported separately from detector alarms",
                        "BadFU is a client-level surrogate, not original shard unlearning"]
    # Keep exact comparator versions for replay. The later binary-task guard does
    # not affect these C=10 experiments, but silently replacing run hashes would.
    source = Path(__file__).with_name("response_audit.py").read_bytes()
    block = ("        if base.shape[1] < 3:\n"
             "            # Incoming-source entropy is identically zero for binary tasks.\n"
             "            return {**output, \"reason\": \"requires_at_least_three_classes\"}\n")
    newline = b"\r\n" if b"\r\n" in source else b"\n"
    previous = source.replace(block.replace("\n", newline.decode()).encode(), b"")
    versions = {hashlib.sha256(b).hexdigest(): b for b in (source, previous)}
    snapshots = folder / "source_snapshots"
    snapshots.mkdir(exist_ok=True)
    for hashes in output["source_versions"].values():
        digest = hashes["response_audit.py"]
        if digest not in versions:
            raise ValueError("unavailable recorded response comparator version")
        (snapshots / f"response_audit_{digest[:12]}.py").write_bytes(versions[digest])
    (folder/"summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (folder/"diagnostics.json").write_text(json.dumps(details, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("FUBA/logs/response_audit"))
    summarize(parser.parse_args().folder)
