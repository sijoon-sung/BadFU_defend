"""Copy existing compact BadFU evidence into the Git-tracked export directory.

No training, observation, or FU. This closes the gap where a temporal result
referenced only an absolute path on the remote computer.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproduction.detection import temporal_history as th
from reproduction.detection.validate_temporal_history import write_json


def export(source, export_root):
    source = Path(source).resolve()
    full = th.load_evidence(str(source)+"::full6")
    clean = th.load_evidence(str(source)+"::clean5")
    if full["group"] != clean["group"] or full["coordinates"] != clean["coordinates"]:
        raise ValueError("BadFU phases do not share a run and observation coordinates")
    with zipfile.ZipFile(source) as z:
        # An observation evidence bundle must not contain model/image files.
        for info in z.infolist():
            if info.is_dir():
                continue
            if (not info.filename.startswith("evidence/") or ".." in Path(info.filename).parts
                    or Path(info.filename).suffix not in (".json", ".npz", ".csv", ".txt")):
                raise ValueError(f"Not a compact observation evidence file: {info.filename}")
        training = json.loads(z.read("evidence/training_protocol.json"))
    sha = th.digest(source.read_bytes())
    dest = Path(export_root) / f"badfu_author_s{training['options']['seed']}" / sha[:16]
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "badfu_detection_evidence.zip"
    if target.exists() and th.digest(target.read_bytes()) != sha:
        raise ValueError("Existing evidence differs; preserved without overwrite")
    if not target.exists():
        shutil.copyfile(source, target)
    if th.digest(target.read_bytes()) != sha:
        raise ValueError("Evidence copy failed hash verification")
    write_json(dest / "export.json", dict(schema=1, archive=target.name, sha256=sha,
        inventory_sha256=full["inventory_sha256"], training_group=full["group"],
        rounds=len(full["history"]), smoke=training["options"].get("smoke"),
        training_calls=0, model_unlearning_calls=0, observation_calls=0))
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--export-root", type=Path, default=ROOT / "reproduction/validation/exports")
    a = p.parse_args()
    print(export(a.evidence, a.export_root))


if __name__ == "__main__":
    main()
