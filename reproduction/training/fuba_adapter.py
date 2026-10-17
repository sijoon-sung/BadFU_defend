"""Record original MPI FUBA; never substitute a sequential attack implementation."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from reproduction.common.trajectory import Recorder, cpu, json_write


class Adapter:
    def __init__(self, options):
        self.options = options
        self.root = Path(options["out"])
        self.recorder = None
        random.seed(options["seed"])
        np.random.seed(options["seed"])
        torch.manual_seed(options["seed"])
        torch.set_num_threads(1)

    def limit_test(self, dataset):
        if self.options["smoke"]:
            return torch.utils.data.Subset(dataset, range(min(64, len(dataset))))
        return dataset

    def splits(self, splits, rank, config):
        indices = [list(map(int, s.indices)) for s in splits]
        if self.options["smoke"]:
            splits = [torch.utils.data.Subset(s.dataset, list(s.indices)[:64]) for s in splits]
        counts = [len(s) for s in splits]
        if min(counts) < config.batch_size:
            raise ValueError("A client has fewer than batch_size samples with drop_last=True; change split/alpha or batch size explicitly")
        json_write(self.root / f"split_rank{rank}.json", {
            "full_indices": indices, "effective_counts": counts, "smoke": self.options["smoke"]})
        return splits

    def start(self, config, state):
        if config.rank == 0:
            self.recorder = Recorder(self.root / "fl", state, config.communication_rounds, {
                "attack": "fuba", "dataset": config.dataset, "smoke": self.options["smoke"],
                "num_clients": config.num_clients, "aggregation": "fuba_uniform",
                "upstream_commit": self.options["upstream_commit"],
                "asr_definition": "author evaluator: excludes target labels, last individual generator in train mode",
            })

    def before(self, state):
        self.before_state = cpu(state)

    def record(self, states, after, selected, generators, global_generator):
        if sorted(i for i, s in enumerate(states) if s is not None) != sorted(selected):
            raise ValueError("MPI returned different clients from the selected set")
        self.recorder.append(self.before_state, [states[i] for i in selected], after, selected,
            [1/len(selected)] * len(selected), {
                "individual_generators": cpu({i: s for i, s in enumerate(generators) if s is not None}),
                "aggregate_generator": cpu(global_generator),
            })

    def metrics(self, r, acc, asr, generators, warm_up):
        available = [i for i in range(5) if generators[i] is not None] if r >= warm_up else []
        json_write(self.root / f"metrics_round_{r+1:04d}.json", {
            "round": r+1, "clean_acc_pct": acc, "asr_pct": asr,
            "author_generator_client": available[-1] if available else None,
            "generator_source": "last_individual" if available else ("random_initialization" if r >= warm_up else "warmup_no_asr"),
            "generator_mode": "train", "target_excluded": True,
        })

    def finish(self, config):
        if config.rank == 0:
            self.recorder.complete()


def patch_files(runtime, options, patch_file):
    prefix = "import json\nfrom reproduction.training.fuba_adapter import Adapter\n_adapter = Adapter(json.loads(" + repr(json.dumps(options)) + "))\n"

    def train(source, replace):
        source = prefix + source
        source = replace(source, "root='./data'", f"root={options['data_root']!r}", 6)
        source = replace(source, "download=True", f"download={options['download']!r}", 4)
        source = replace(source, "num_workers=12", "num_workers=0", 2)
        source = replace(source, "    # test pretrained model", "    testset = _adapter.limit_test(testset)\n\n    # test pretrained model")
        source = replace(source, "        # Create loaders for each subset", "        trainset_split = _adapter.splits(trainset_split, config.rank, config)\n        # Create loaders for each subset")
        source = replace(source, '    print("start")', '    _adapter.start(config, global_model_dict)\n    print("start")')
        source = replace(source, "            # Apply chosen defense method for aggregation", "            _adapter.before(global_model_dict)\n            # Apply chosen defense method for aggregation")
        source = replace(source, "            # Send aggregated models to tester node", "            _adapter.record(local_model_dicts, global_model_dict, selected_clients, backdoor_model_dicts, global_backdoor_model_dict)\n            # Send aggregated models to tester node")
        source = replace(source, "                # Print results", "                _adapter.metrics(r, ACC, ASR, backdoor_model_dicts, config.warm_up)\n                # Print results")
        # Run the unchanged legacy serializer synchronously: failures must reach the launcher.
        start = source.index("            # Save results in separate thread")
        source = source[:start] + '''            # Synchronous legacy export for the author's unlearn.py.
            if config.save_results:
                save_models(global_model_dict_current, local_model_dicts_current, backdoor_model, r + 1, config.name)
    _adapter.finish(config)
'''
        return source

    patch_file(runtime / "train_main.py", train)

    def dataset(source, replace):
        old = "\n".join(["import nvidia.dali.fn as fn", "import nvidia.dali.types as types",
            "from nvidia.dali.pipeline import Pipeline", "from nvidia.dali.plugin.pytorch import DALIGenericIterator",
            "from nvidia.dali.plugin.base_iterator import LastBatchPolicy"])
        new = "try:\n" + "\n".join("    " + line for line in old.splitlines()) + '''
except ModuleNotFoundError:
    class Pipeline:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ImageNet requires NVIDIA DALI; this launcher supports MNIST/CIFAR10")
'''
        return replace(source, old, new)
    patch_file(runtime / "dataset.py", dataset)

    def split_fix(source, replace):
        # torchvision CIFAR10 targets are a Python list: list == int is a scalar False.
        return replace(source, "        y = dataset.targets", "        y = np.asarray(dataset.targets)", 2)
    patch_file(runtime / "utils" / "comm_utils.py", split_fix)
