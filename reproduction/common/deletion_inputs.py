"""Read-only model/trajectory adapters for the deletion comparison experiment."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from reproduction.common.trajectory import check_state, sha256, verify

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


class BadFUModel(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        # All state, including pretrained trunk, comes from the saved checkpoint.
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)


class Pixels(nn.Module):
    def __init__(self, base, family):
        super().__init__()
        self.base = base
        mean, std = (([.4914, .4822, .4465], [.2023, .1994, .2010])
                     if family == 'badfu' else ([0.], [1.]))
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None])
        self.register_buffer('std', torch.tensor(std)[None, :, None, None])

    def forward(self, x):
        return self.base(((x - self.mean) / self.std).contiguous())


def make_model(family, dataset):
    from FUBA.model import Net, resnet18
    if family not in ('badfu', 'fuba') or dataset not in ('cifar10', 'mnist'):
        raise ValueError('only recorded BadFU CIFAR10 and FUBA CIFAR10/MNIST supported')
    if family == 'badfu' and dataset != 'cifar10':
        raise ValueError('BadFU requires CIFAR10')
    base = BadFUModel() if family == 'badfu' else Net() if dataset == 'mnist' else resnet18()
    return Pixels(base, family)


class Trajectory:
    def __init__(self, run, phase):
        self.run, self.phase = resolve(run), phase
        if phase not in ('fl', 'full6', 'clean5', 'cover5', 'bd5'):
            raise ValueError('unsupported FL phase')
        self.root = self.run / phase
        self.protocol = read(self.run / 'protocol.json')
        self.manifest = read(self.root / 'manifest.json')
        options, meta = self.protocol['options'], self.manifest['metadata']
        self.family = options['attack']
        self.dataset = options.get('dataset', 'cifar10')
        if (self.protocol.get('status') != 'complete' or meta['attack'] != self.family
                or meta.get('upstream_commit') != self.protocol['upstream']['commit']):
            raise ValueError('incomplete or inconsistent recorded run')
        expected = 'fuba_uniform' if self.family == 'fuba' else 'badfu_sample_weighted'
        if meta.get('aggregation') != expected:
            raise ValueError('only original recorded FL aggregation supported')
        if self.family == 'fuba':
            if phase != 'fl' or meta['dataset'] != self.dataset or meta['num_clients'] != options['clients']:
                raise ValueError('FUBA protocol mismatch')
        elif phase == 'fl' or meta['phase'] != phase:
            raise ValueError('BadFU phase mismatch')
        for name, digest in self.protocol['adapter_sha256'].items():
            path = self.run / 'reproduction' / name
            if path.parent.resolve() != (self.run / 'reproduction').resolve() or sha256(path) != digest:
                raise ValueError(f'training snapshot mismatch: {name}')
        if sha256(self.run / 'runtime.patch') != self.protocol['runtime_patch_sha256']:
            raise ValueError('training patch mismatch')
        print(f'  Verify recorded trajectory: {self.run.name}/{phase}', flush=True)
        self.integrity = verify(self.root)
        self.ids = sorted({i for e in self.manifest['rounds'] for i in e['client_ids']})

    def rounds(self):
        for entry in self.manifest['rounds']:
            path = self.root / entry['path']
            if path.parent.resolve() != self.root.resolve() or sha256(path) != entry['sha256']:
                raise ValueError('trajectory changed after verification')
            state = torch.load(path, map_location='cpu', weights_only=True)
            if self.family == 'fuba':
                expected = np.full(len(state['client_ids']), 1 / len(state['client_ids']))
            else:
                meta = self.manifest['metadata']
                counts = dict(zip(meta['client_ids'], meta['counts']))
                expected = np.asarray([counts[i] for i in state['client_ids']], dtype=float)
                expected /= expected.sum()
            if not np.allclose(state['weights'], expected, rtol=0, atol=1e-10):
                raise ValueError('recorded weights disagree with original aggregation protocol')
            yield state

    def models(self, request, actual_phase=None, actual_state=None, dose=1.):
        if (not request or len(set(request)) != len(request) or not set(request) < set(self.ids)
                or not np.isfinite(dose) or dose <= 0):
            raise ValueError('proper nonempty subset of recorded clients and positive dose required')
        pre = make_model(self.family, self.dataset)
        names = dict(pre.base.named_parameters())
        accumulated = {k: torch.zeros_like(p) for k, p in names.items()}
        participation = dict.fromkeys(request, 0)
        for state in self.rounds():
            for client, weight, local in zip(state['client_ids'], state['weights'], state['cms']):
                if client in participation:
                    participation[client] += 1
                    for k in names:
                        accumulated[k].add_(local[k] - state['gm'][k], alpha=weight)
            final = state['gm_after']
        pre.base.load_state_dict(final, strict=True)
        post = copy.deepcopy(pre)
        info = dict(candidate_kind='historical_parameter_subtraction_surrogate',
                    request_clients=request, request_unit='whole_client', dose=dose,
                    participation=participation, actual_fu_verified=False)
        if actual_phase and actual_state:
            raise ValueError('choose one actual FU source')
        if actual_phase:
            if (self.family, self.phase, actual_phase, request) not in (
                    ('badfu', 'full6', 'fu_remove5', [5]),
                    ('badfu', 'cover5', 'fu_remove1', [1])):
                raise ValueError('actual BadFU request does not match recorded author pipeline')
            fu = self.run / actual_phase
            verify(fu)
            m = read(fu / 'manifest.json')
            if (m['metadata']['aggregation'] != 'author_federaser'
                    or m['metadata']['phase'] != actual_phase):
                raise ValueError('actual FU metadata mismatch')
            entry = m['rounds'][-1]
            path = fu / entry['path']
            if sha256(path) != entry['sha256']:
                raise ValueError('actual FU changed')
            state = torch.load(path, map_location='cpu', weights_only=True)['gm_after']
            check_state(state, final)
            post.base.load_state_dict(state, strict=True)
            info.update(candidate_kind='actual_author_fu', actual_fu_verified=True,
                        fu_phase=actual_phase, fu_manifest_sha256=sha256(fu / 'manifest.json'))
        elif actual_state:
            # Explicit external binding. Provenance is asserted by supplier, not proved.
            if (actual_state.get('request_clients') != request or
                    actual_state.get('pre_manifest_sha256') != sha256(self.root / 'manifest.json')):
                raise ValueError('external FU must bind exact pre-manifest and deletion IDs')
            path = resolve(actual_state['path'])
            if sha256(path) != actual_state['sha256']:
                raise ValueError('external FU checkpoint hash mismatch')
            state = torch.load(path, map_location='cpu', weights_only=True)
            if actual_state.get('key'):
                state = state[actual_state['key']]
            check_state(state, final)
            post.base.load_state_dict(state, strict=True)
            info.update(candidate_kind='external_fu_supplier_attested', external_binding=actual_state)
        else:
            with torch.no_grad():
                for k, p in post.base.named_parameters():
                    p.sub_(accumulated[k], alpha=dose)
            check_state(post.base.state_dict(), final)
            info['buffer_policy'] = 'final FL buffers unchanged; parameters only subtracted'
        return pre, post, info

    def group_updates(self, request):
        """Exact weighted unions of recorded clients, preserving physical rounds."""
        model = make_model(self.family, self.dataset)
        names = dict(model.base.named_parameters())
        for state in self.rounds():
            groups = [{f'base.{k}': torch.zeros_like(p) for k, p in names.items()} for _ in range(2)]
            for client, weight, local in zip(state['client_ids'], state['weights'], state['cms']):
                group = groups[0 if client in request else 1]
                for k in names:
                    group[f'base.{k}'].add_(local[k] - state['gm'][k], alpha=weight)
            yield groups


def public_pool(trajectory, data_root=None, pool=None):
    dataset = trajectory.dataset
    if pool:
        path = resolve(pool)
        with np.load(path, allow_pickle=False) as a:
            if set(a.files) != {'images', 'labels', 'ids'}:
                raise ValueError('public pool must contain images, labels, ids ONLY')
            images, labels, ids = (a[k].copy() for k in ('images', 'labels', 'ids'))
        source = dict(kind='user_supplied_trusted_public_labeled', sha256=sha256(path))
    else:
        from torchvision.datasets import CIFAR10, MNIST
        candidates = ([resolve(data_root)] if data_root else
                      [resolve(trajectory.protocol['options']['data_root']), ROOT / 'FUBA/data'])
        ds, errors = None, []
        for root in candidates:
            try:
                ds = (MNIST if dataset == 'mnist' else CIFAR10)(str(root), train=False, download=False)
                break
            except RuntimeError as e:
                errors.append(str(e))
        if ds is None:
            raise ValueError(f'local public test set missing; use --data-root or labeled pool: {errors}')
        images = ds.data.numpy()[..., None] if dataset == 'mnist' else np.asarray(ds.data)
        labels = np.asarray(ds.targets, dtype=np.int64)
        ids = np.asarray([f'{dataset}:test:{i}' for i in range(len(images))])
        if dataset == 'cifar10':
            excluded = set(read(ROOT / 'FUBA/logs/hrac/unlabeled_excluded_ids_v1.json'))
            keep = np.array([i not in excluded for i in ids])
            images, labels, ids = images[keep], labels[keep], ids[keep]
        source = dict(kind=f'public_{dataset}_test_partition', root=str(root),
                      limitation='reused benchmark test data; not a new untouched final test set')
    shape = (28, 28, 1) if dataset == 'mnist' else (32, 32, 3)
    if images.shape[1:] != shape:
        raise ValueError('public pixels disagree with dataset shape')
    return images, labels, ids, source
