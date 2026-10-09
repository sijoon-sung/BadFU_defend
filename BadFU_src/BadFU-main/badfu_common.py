# -*- coding: utf-8 -*-
"""BadFU 'ul' 구성의 데이터·모델·학습 구성요소 (fl_detect_badfu.py 와 같은 내용을 모듈로).

ul 구성 (badfu.py 의 ul_client_datasets):
  client 0 = clean_0 + bd (백도어, 공격자)
  client 1..4 = 정상 (각자 두 클래스를 dominant_ratio 만큼 독점: 1->{2,3}, 2->{4,5}, 3->{6,7}, 4->{8,9})
  client 5 = clean_0 + cv (위장, 언러닝 요청자)
클래스 0,1 은 공격자(client 0)가 독점.
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DOMINANT_MAP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}
ATTACKER, REQUESTER, N_CLEAN_CLIENTS = 0, 5, 5
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
PERT = os.path.join(HERE, "record/badnet_dataset/cv_train_dataset/pert")
BDTR = os.path.join(HERE, "record/badnet_dataset/bd_train_dataset")


def _resolve(path):
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    cls = os.path.basename(os.path.dirname(path))
    for cand in (os.path.join(PERT, cls, base), os.path.join(BDTR, cls, base)):
        if os.path.exists(cand):
            return cand
    return path


class TensorDS(Dataset):
    def __init__(self, data, targets):
        self.data, self.targets = data, targets

    def __getitem__(self, i):
        return self.data[i], self.targets[i]

    def __len__(self):
        return len(self.targets)


class BackdoorDataset(Dataset):
    """pert_result.pt 의 data_dict. other_info[0] = 학습/평가 라벨. orig_label_idx 가 있으면 원래 라벨도 돌려준다."""

    def __init__(self, data_dict, transform=TRANSFORM):
        self.d = data_dict
        self.keys = list(data_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        info = self.d[self.keys[i]]
        img = self.transform(Image.open(_resolve(info["path"])).convert("RGB"))
        return img, torch.tensor(info["other_info"][0])

    def original_labels(self, cifar_targets):
        """키 = 원본 CIFAR 인덱스 (BackdoorBench 관례)."""
        return torch.tensor([cifar_targets[int(k)] for k in self.keys])


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class BadFUData:
    """ul 구성 6 클라 데이터와 평가 로더."""

    def __init__(self, dominant_ratio=0.7, seed=42, batch_size=64):
        np.random.seed(seed)
        d = torch.load(os.path.join(HERE, "record/badnet_dataset/pert_result.pt"), weights_only=False)
        bd_train = d["bd_train"]["bd_data_container"]["data_dict"]
        cv = d["cv_pert"]["data_dict"]
        bd_test = d["bd_test"]["bd_data_container"]["data_dict"]
        remove = set(list(bd_train.keys()) + list(cv.keys()))
        train = datasets.CIFAR10(root=os.path.join(HERE, "data"), train=True, download=True)
        test = datasets.CIFAR10(root=os.path.join(HERE, "data"), train=False, download=True, transform=TRANSFORM)
        kept = [i for i in range(len(train)) if i not in remove]
        x = torch.stack([TRANSFORM(Image.fromarray(train.data[i])) for i in kept])
        y = torch.tensor([train.targets[i] for i in kept])
        clean = TensorDS(x, y)
        tg = y.numpy()
        per = [[] for _ in range(N_CLEAN_CLIENTS)]
        for c in np.unique(tg):
            ci = np.where(tg == c)[0]
            np.random.shuffle(ci)
            dom = DOMINANT_MAP[c]
            sp = int(len(ci) * dominant_ratio)
            per[dom].extend(ci[:sp].tolist())
            rem = ci[sp:]
            others = [k for k in range(N_CLEAN_CLIENTS) if k != dom]
            ch = len(rem) // len(others)
            for i, cid in enumerate(others):
                e = (i + 1) * ch if i < len(others) - 1 else len(rem)
                per[cid].extend(rem[i * ch:e].tolist())
        self.bd = BackdoorDataset(bd_train)
        self.cv = BackdoorDataset(cv)
        self.bd_test = BackdoorDataset(bd_test)
        clean_parts = [TensorDS(x[idx], y[idx]) for idx in per]
        self.client_sets = [ConcatDataset([clean_parts[0], self.bd])] + clean_parts[1:] + \
                           [ConcatDataset([clean_parts[0], self.cv])]
        self.counts = [len(s) for s in self.client_sets]
        self.batch_size = batch_size
        self.test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=0)
        self.bd_test_orig = self.bd_test.original_labels(test.targets)

    def loader(self, k, gen=None):
        return DataLoader(self.client_sets[k], batch_size=self.batch_size, shuffle=True, num_workers=0, generator=gen)


def local_train(model, loader, epochs, device):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            crit(model(data), target).backward()
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def evaluate(model, data, device, target=0, n_classes=10):
    """ACC, 전체 ASR, 원래 클래스별 ASR (bd_test 는 타깃 클래스 원본을 이미 뺀 것이 보통)."""
    model.eval()
    cor = tot = 0
    for x, y in data.test_loader:
        p = model(x.to(device)).argmax(1).cpu()
        cor += (p == y).sum().item()
        tot += len(y)
    preds = []
    for x, _ in DataLoader(data.bd_test, batch_size=256, shuffle=False, num_workers=0):
        preds.append(model(x.to(device)).argmax(1).cpu())
    preds = torch.cat(preds)
    hit = preds == target
    per = []
    for c in range(n_classes):
        m = data.bd_test_orig == c
        per.append(None if (c == target or m.sum() == 0) else round(100.0 * hit[m].float().mean().item(), 1))
    return {"acc": round(100.0 * cor / tot, 2), "asr": round(100.0 * hit.float().mean().item(), 2),
            "asr_by_class": per}


def state_keys():
    return list(ResNet18().state_dict().keys())


def flat(sd, keys):
    return torch.cat([sd[k].detach().float().cpu().flatten() for k in keys])


def unflat(v, ref_sd, keys):
    out, i = {}, 0
    for k in keys:
        n = ref_sd[k].numel()
        t = v[i:i + n].view_as(ref_sd[k])
        out[k] = ref_sd[k].clone() if ref_sd[k].dtype == torch.long else t.to(ref_sd[k].dtype)
        i += n
    return out
