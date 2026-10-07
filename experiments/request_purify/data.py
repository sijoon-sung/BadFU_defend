"""Data preparation without modifying the upstream record or client partition."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def resolve_image(stored, bundle):
    """Resolve relocated upstream paths, rejecting conflicting copies."""
    path = Path(str(stored).replace("\\", "/"))
    candidates = []
    if path.is_absolute() and path.is_file():
        candidates.append(path)
    parts = path.parts
    if bundle.name in parts:
        pos = parts.index(bundle.name)
        candidates.append(bundle.joinpath(*parts[pos + 1:]))
    candidates.append(bundle / path)
    if len(parts) >= 2:
        # The original prepare_data.sh moves non-target class images here.
        for folder in ("cv_train_dataset/pert", "bd_train_dataset"):
            candidates.append(bundle / folder / parts[-2] / parts[-1])
    found = sorted({p.resolve() for p in candidates if p.is_file()}, key=str)
    if not found:
        raise FileNotFoundError(f"Missing record image: {stored}; bundle={bundle}")
    if len(found) > 1 and len({file_hash(p) for p in found}) != 1:
        raise ValueError(f"Ambiguous record image, conflicting copies: {found}")
    return found[0]


class RecordImages(Dataset):
    def __init__(self, records, bundle, transform, target=None):
        self.transform = transform
        self.samples = []
        inventory = []
        for key, info in sorted(records.items(), key=lambda kv: int(kv[0])):
            labels = info["other_info"]
            if len(labels) < 2:
                raise ValueError("Record must contain both assigned and original labels")
            original = int(labels[1])
            if target is not None and original == target:
                continue
            path = resolve_image(info["path"], bundle)
            self.samples.append((path, int(labels[0]), original, int(key)))
            inventory.append([int(key), int(labels[0]), original, file_hash(path)])
        if not self.samples:
            raise ValueError("Empty backdoor/camouflage record")
        self.digest = object_hash(inventory)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, _, _ = self.samples[index]
        with Image.open(path) as image:
            x = self.transform(image.convert("RGB"))
        return x, label


class BadFUModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)


class SmokeModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(8, 12), nn.Tanh())
        self.fc = nn.Linear(12, 3)

    def forward(self, x):
        return self.fc(self.features(x))


def load_badfu_source(record, data_root, target, download):
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, Normalize, ToTensor
    record = Path(record).resolve()
    if not record.is_file():
        raise FileNotFoundError(f"Prepare the original BadFU record first: {record}")
    # This format is a user-supplied, trusted PyTorch pickle from the original code.
    source = torch.load(record, map_location="cpu", weights_only=False)
    transform = Compose([ToTensor(), Normalize((.4914, .4822, .4465), (.2023, .1994, .2010))])
    bd = RecordImages(source["bd_train"]["bd_data_container"]["data_dict"], record.parent, transform)
    cv = RecordImages(source["cv_pert"]["data_dict"], record.parent, transform)
    test_bd = RecordImages(source["bd_test"]["bd_data_container"]["data_dict"],
                           record.parent, transform, target=target)
    if any(label != target for _, label, _, _ in bd.samples + test_bd.samples):
        raise ValueError("--target disagrees with the prepared BadFU labels")
    train = CIFAR10(data_root, train=True, transform=transform, download=download)
    test = CIFAR10(data_root, train=False, transform=transform, download=download)
    ids_bd = {item[3] for item in bd.samples}
    ids_cv = {item[3] for item in cv.samples}
    if ids_bd & ids_cv or not all(0 <= i < len(train) for i in ids_bd | ids_cv):
        raise ValueError("Poison/camouflage source IDs overlap or are out of range")
    source_meta = {"record_sha256": file_hash(record), "bd_sha256": bd.digest,
                   "cv_sha256": cv.digest, "test_trigger_sha256": test_bd.digest,
                   "bd_ids": sorted(ids_bd), "cv_ids": sorted(ids_cv),
                   "clean_train_sha256": hashlib.sha256(train.data.tobytes()).hexdigest()}
    return train, test, bd, cv, test_bd, source_meta


def badfu_case(source, seed, benign, val_size, dominant, duplicate_clean=True):
    train, test, bd, cv, test_bd, source_meta = source
    rng = np.random.default_rng(seed)
    excluded = set(source_meta["bd_ids"] + source_meta["cv_ids"])
    eligible = np.array([i for i in range(len(train)) if i not in excluded])
    if val_size < 0 or val_size >= len(eligible) - 100:
        raise ValueError("Invalid validation size")
    shuffled = rng.permutation(eligible)
    validation_ids = sorted(shuffled[:val_size].tolist())
    eligible = shuffled[val_size:]
    labels = np.asarray(train.targets)
    partitions = [[] for _ in range(5)]
    for cls in range(10):
        ids = rng.permutation(eligible[labels[eligible] == cls])
        cut, owner = int(len(ids) * dominant), cls // 2
        partitions[owner].extend(ids[:cut].tolist())
        others = [i for i in range(5) if i != owner]
        for client, part in zip(others, np.array_split(ids[cut:], 4)):
            partitions[client].extend(part.tolist())
    clean = [Subset(train, ids) for ids in partitions]
    extra = clean[0] if duplicate_clean else Subset(train, [])
    if benign:
        # Keep six clients without introducing trigger-bearing data.
        # Use clean source images for the corresponding bd/cv samples.
        clients = [ConcatDataset([clean[0], Subset(train, source_meta["bd_ids"])])] + clean[1:]
        clients.append(ConcatDataset([extra, Subset(train, source_meta["cv_ids"])]))
    else:
        clients = [ConcatDataset([clean[0], bd])] + clean[1:]
        clients.append(ConcatDataset([extra, cv]))
    clean_clients = [ConcatDataset([clean[0], Subset(train, source_meta["bd_ids"])])] + clean[1:]
    clean_clients.append(ConcatDataset([extra, Subset(train, source_meta["cv_ids"])]))
    meta = {"dataset": "badfu", "seed": seed, "benign": benign,
            "dominant_ratio": dominant, "duplicate_clean": duplicate_clean,
            "validation_ids": validation_ids, "partitions": partitions, **source_meta}
    return {"clients": clients, "clean_clients": clean_clients, "test": test,
            "trigger_test": test_bd, "validation": Subset(train, validation_ids),
            "factory": BadFUModel, "head": "model.fc", "classes": 10, "meta": meta}


def smoke_case(seed, benign, **ignored):
    gen = torch.Generator().manual_seed(seed)
    def samples(count):
        x = torch.randn(count, 8, generator=gen)
        y = x[:, :3].argmax(1)
        x[:, -1] *= .1
        return x, y
    clean = [TensorDataset(*samples(36)) for _ in range(5)]
    x, y = samples(24)
    bd_clean = TensorDataset(x.clone(), y.clone())
    bd = TensorDataset(x.clone(), torch.zeros_like(y)); bd.tensors[0][:, -1] = 4
    x, y = samples(24)
    cv_clean = TensorDataset(x.clone(), y.clone())
    cv = TensorDataset(x.clone(), y.clone()); cv.tensors[0][:, -1] = 4
    clean_clients = [ConcatDataset([clean[0], bd_clean])] + clean[1:] + [ConcatDataset([clean[0], cv_clean])]
    clients = clean_clients if benign else [ConcatDataset([clean[0], bd])] + clean[1:] + [ConcatDataset([clean[0], cv])]
    x, y = samples(90)
    test = TensorDataset(x, y)
    tx = x[y != 0].clone(); tx[:, -1] = 4
    return {"clients": clients, "clean_clients": clean_clients, "test": test,
            "trigger_test": TensorDataset(tx, torch.zeros(len(tx), dtype=torch.long)),
            "validation": TensorDataset(*samples(30)), "factory": SmokeModel,
            "head": "fc", "classes": 3,
            "meta": {"dataset": "smoke", "seed": seed, "benign": benign,
                     "warning": "Synthetic execution check; not a BadFU efficacy result"}}
