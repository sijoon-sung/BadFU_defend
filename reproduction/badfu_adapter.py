"""Instrumentation for the pinned author's sequential BadFU experiment."""
import copy
import json
from pathlib import Path
import torch
from torch.utils.data import Subset
from reproduction.trajectory import Recorder, cpu, json_write, save, sha256


class DiskHistory:
    def __init__(self, owner, phase, kind):
        self.owner, self.phase, self.kind, self.length = owner, phase, kind, 0

    def append(self, ignored=None):
        if len(self.owner.recorders[self.phase].manifest["rounds"]) != self.length + 1:
            raise ValueError("History index differs from recorded round")
        self.length += 1

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        rec = self.owner.recorders[self.phase]
        entry = rec.manifest["rounds"][index]
        path = rec.root / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("History checkpoint changed")
        state = torch.load(path, map_location="cpu", weights_only=True)

        def model(sd):
            m = copy.deepcopy(self.owner.templates[self.phase])
            m.load_state_dict(sd)
            return m.to(self.owner.options["device"])

        if self.kind == "global":
            return model(state["gm_after"])
        # The author's six-client FU omits client 5; the five-client FU pops 1 later.
        return [model(s) for s in state["cms"][:5]]


class Adapter:
    def __init__(self, options):
        self.options = options
        self.root = Path(options["out"])
        self.recorders, self.templates = {}, {}
        self.phase = None
        self.evaluations = []
        self.model_names = {}

    def resolve_image(self, path):
        root = Path(self.options["record_root"])
        parts = str(path).replace("\\", "/").split("/")
        if "badnet_dataset" in parts:
            candidate = root.joinpath(*parts[parts.index("badnet_dataset") + 1:])
        else:
            candidate = root / str(path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing original record image: {candidate}")
        return candidate

    def prepare_record(self, data):
        containers = [data["bd_train"]["bd_data_container"]["data_dict"],
                      data["bd_test"]["bd_data_container"]["data_dict"], data["cv_pert"]["data_dict"]]
        files = {}
        for container in containers:
            for info in container.values():
                path = self.resolve_image(info["path"])
                files[str(path.relative_to(self.options["record_root"]))] = sha256(path)
        json_write(self.root / "record_inputs.json", {
            "pert_result_sha256": sha256(Path(self.options["record_root"]) / "pert_result.pt"),
            "container_counts": [len(c) for c in containers], "image_sha256": files,
        })
        return data

    def limit(self, dataset):
        if not self.options["smoke"]:
            return dataset
        # Deterministic, spread-out indices; no global RNG consumption.
        n = min(32, len(dataset))
        indices = [i * len(dataset) // n for i in range(n)]
        return Subset(dataset, indices)

    def save_split(self, clean_indices, poison_indices, cover_indices):
        json_write(self.root / 'split.json', {
            'clean_cifar10_indices': clean_indices,
            'poison_cifar10_indices': list(map(int, poison_indices)),
            'cover_cifar10_indices': list(map(int, cover_indices)),
            'composition': {'clean5': 'clean i', 'bd5': 'clean i; client 0 adds poison',
                'cover5': 'clean i; client 0 adds poison + cover',
                'full6': 'clean i; client 0 adds poison; client 5 duplicates clean 0 and adds cover'},
            'smoke': self.options['smoke'],
            'smoke_selection': '32 evenly spaced indices per composed client dataset' if self.options['smoke'] else None,
        })

    def start(self, phase, model, counts, ids=None, fu=False):
        self.phase = phase
        self.model_names[id(model)] = phase
        self.templates[phase] = copy.deepcopy(model).cpu()
        rec = Recorder(self.root / phase, model.state_dict(),
                       self.options["rounds"] // 2 if fu else self.options["rounds"], {
            "attack": "badfu", "phase": phase, "smoke": self.options["smoke"],
            "aggregation": "author_federaser" if fu else "badfu_sample_weighted",
            "counts": counts, "client_ids": ids or list(range(len(counts))),
            "asr_definition": "author: predicted target 0 / all bd_test records",
            "upstream_commit": self.options["upstream_commit"],
        })
        self.recorders[phase] = rec
        self.ids = ids or list(range(len(counts)))

    def wrap_average(self, original):
        def average(model, states, counts):
            before = cpu(model.state_dict())
            result = original(model, states, counts)
            self.recorders[self.phase].append(before, states, result.state_dict(),
                                              self.ids, [n / sum(counts) for n in counts])
            return result
        return average

    def wrap_unlearning(self, original):
        def unlearn(old_clients, new_clients, old_global, new_global):
            before = cpu(new_global.state_dict())
            result = original(old_clients, new_clients, old_global, new_global)
            self.model_names[id(result)] = self.phase
            self.recorders[self.phase].append(before, [m.state_dict() for m in new_clients],
                result.state_dict(), self.ids, [1/len(new_clients)] * len(new_clients),
                {"weights_meaning": "uniform recalibration, not final FedAvg"})
            return result
        return unlearn

    def wrap_evaluate(self, original):
        def evaluate(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            self.evaluations.append({"phase": self.phase, "model": self.model_names.get(id(model), "local"),
                "completed_rounds": len(self.recorders[self.phase].manifest["rounds"]),
                "clean_acc_pct": result[0], "class_acc_pct": result[1], "asr_pct": result[2]})
            json_write(self.root / "metrics.json", self.evaluations)
            return result
        return evaluate

    def history(self, phase, kind):
        return DiskHistory(self, phase, kind)

    def finish(self):
        for rec in self.recorders.values():
            rec.complete()


def patch(source, options, replace):
    source = "import json\nfrom reproduction.badfu_adapter import Adapter\n_adapter = Adapter(json.loads(" + repr(json.dumps(options)) + "))\n" + source
    for old, new in [
        ("local_epochs = 5 ", f"local_epochs = {options['local_epochs']} "),
        ("num_rounds = 40 ", f"num_rounds = {options['rounds']} "),
        ('device = "cuda:0" if torch.cuda.is_available() else "cpu"', f"device = {options['device']!r}"),
        ("seed = 42", f"seed = {options['seed']}"),
        ('data_path = "./record/badnet_dataset/pert_result.pt"', f"data_path = {str(Path(options['record_root']) / 'pert_result.pt')!r}"),
        ("data_dict = torch.load(data_path)", "data_dict = _adapter.prepare_record(torch.load(data_path, map_location='cpu', weights_only=False))"),
        ("image_path = info['path']", "image_path = _adapter.resolve_image(info['path'])"),
    ]:
        source = replace(source, old, new)
    source = replace(source, "root='./data'", f"root={options['data_root']!r}", 2)
    source = replace(source, "download=True", f"download={options['download']!r}", 2)
    source = replace(source, "num_workers=2", "num_workers=0", 2)
    source = replace(source, "DataLoader(client_data, batch_size=64", "DataLoader(_adapter.limit(client_data), batch_size=64", 4)
    source = replace(source, "DataLoader(bd_test_dataset,", "DataLoader(_adapter.limit(bd_test_dataset),")
    source = replace(source, "DataLoader(test_dataset,", "DataLoader(_adapter.limit(test_dataset),")
    source = replace(source, "client_clean_data = [[] for _ in range(num_clients)]", "client_clean_data = [[] for _ in range(num_clients)]\n_clean_index_map = [[] for _ in range(num_clients)]")
    source = replace(source, "        client_clean_data[dominant_client_id].append(clean_dataset[idx])", "        client_clean_data[dominant_client_id].append(clean_dataset[idx])\n        _clean_index_map[dominant_client_id].append(int(kept_indices[idx]))")
    source = replace(source, "            client_clean_data[cid].append(clean_dataset[idx])", "            client_clean_data[cid].append(clean_dataset[idx])\n            _clean_index_map[cid].append(int(kept_indices[idx]))")
    source = replace(source, "for cid, samples in enumerate(client_clean_data):", "_adapter.save_split(_clean_index_map, bd_ind, cv_ind)\nfor cid, samples in enumerate(client_clean_data):")
    for suffix, phase in [("clean", "clean5"), ("bd", "bd5"), ("cv", "cover5"), ("for_ul", "full6")]:
        line = f"global_model_{suffix} = ResNet18ForCIFAR10().to(device)"
        source = replace(source, line, line + f"\n_adapter.start('{phase}', global_model_{suffix}, client_data_counts)")
    source = replace(source, "def fed_avg_inner(", "fed_avg = _adapter.wrap_average(fed_avg)\n\ndef fed_avg_inner(")
    source = replace(source, "bd_test_loader = DataLoader", "evaluate_model = _adapter.wrap_evaluate(evaluate_model)\n\nbd_test_loader = DataLoader")
    for name, phase, kind in [("old_CMs_b", "cover5", "clients"), ("old_GMs_b", "cover5", "global"),
                              ("old_CMs", "full6", "clients"), ("old_GMs", "full6", "global")]:
        source = replace(source, f"{name} = []", f"{name} = _adapter.history('{phase}', '{kind}')")
        import re
        source, n = re.subn(rf"{name}\.append\(copy\.deepcopy\([^\n]+\)\)", f"{name}.append()", source)
        if n != 1:
            raise ValueError(f"Upstream append changed: {name}")
    source = replace(source, "global_model_ul = old_GMs[0]", "unlearning_step_once = _adapter.wrap_unlearning(unlearning_step_once)\nglobal_model_ul = old_GMs[0]\n_adapter.start('fu_remove5', global_model_ul, [len(x.dataset) for x in bd_client_loaders], fu=True)")
    source = replace(source, "global_model_ul_b = old_GMs_b[0]", "global_model_ul_b = old_GMs_b[0]\n_adapter.start('fu_remove1', global_model_ul_b, [len(x.dataset) for i, x in enumerate(cv_client_loaders) if i != 1], ids=[0, 2, 3, 4], fu=True)")
    return source + "\n_adapter.finish()\n"
