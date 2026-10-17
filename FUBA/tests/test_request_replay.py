"""Two-round real PyTorch replay with a synthetic dataset, not attack efficacy."""
from pathlib import Path
import json
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import Net
from FUBA.research.experiment_request_replay import run


class ReplayIntegrationTests(unittest.TestCase):
    def test_full_runner_uses_initial_state_retained_archive_and_each_event(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        images = torch.rand(60, 1, 28, 28)
        labels = torch.arange(60) % 10
        units = np.repeat(np.arange(3), 20)
        model = Net()
        previous = {k: v.clone() for k, v in model.state_dict().items()}
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_root) as tmp:
            root = Path(tmp)
            self.assertTrue(root.resolve().is_relative_to(test_root))
            def save(kind, r, value):
                (root/f"{kind}_net_model_fixture_round_{r}.pkl").write_bytes(pickle.dumps(value))
            save("global", 0, previous)
            for r in (1, 2):
                locals_ = []
                for j in range(3):
                    model.load_state_dict(previous)
                    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
                    optimizer.zero_grad()
                    torch.nn.functional.cross_entropy(model(images[j*20:(j+1)*20]), labels[j*20:(j+1)*20]).backward()
                    optimizer.step()
                    locals_.append({k: v.clone() for k, v in model.state_dict().items()})
                previous = {k: torch.stack([state[k] for state in locals_]).mean(0) for k in previous}
                save("global", r, previous)
                save("local", r, locals_)
                (root/f"history_fixture_round_{r}.json").write_text(json.dumps({"selected_clients": [0, 1, 2]}))
            np.savez(root/"retained.npz", images=images[20:].numpy(), labels=labels[20:].numpy(), units=units[20:])
            args = SimpleNamespace(name="fixture", initial=None, checkpoints=str(root), rounds=2,
                retained_archive=str(root/"retained.npz"), data_root=str(root), out=str(root/"report.json"),
                requesters=[0], modes=["no_defense", "retention_only", "temporal"], threads=1, n_gpu=0,
                seed=71, batches=1, batch_size=20, lr=.001, retain=20, evaluate=20, eval_target=None,
                history_window=2, smoothing=1., risk_ratio=.1, epsilon=100., radius_ratio=.25, min_progress=.05)
            fake_test = torch.utils.data.TensorDataset(images, labels)
            with patch("FUBA.research.experiment_request_replay.datasets.MNIST", return_value=fake_test):
                report = run(args)
            self.assertTrue((root/"report.json").exists())
            self.assertFalse(report["forgetting_verified"])
            for result in report["results"].values():
                self.assertTrue(result["completed_all_events"])
                self.assertEqual(result["actual_recalibration_calls"], 4)
                self.assertEqual(len(result["steps"]), 2)
            self.assertTrue(all(s["actual_retention_verified"] for s in report["results"]["temporal"]["steps"]))


if __name__ == "__main__":
    unittest.main()
