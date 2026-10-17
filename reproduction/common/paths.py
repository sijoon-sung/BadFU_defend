"""Resolve recorded repository paths after layout changes, without editing receipts.

External model/data paths are never redirected. Missing files remain missing.
The explicit move ledger preserves source identity separately from numerical replay.
"""
from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def mappings():
    moves, prefixes = {}, {}
    for name in ('code_layout_20261017.json', 'experiment_layout_20261017.json'):
        path = ROOT/'docs/operations'/name
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            moves.update(data['moves'])
            prefixes.update(data.get('directory_aliases', {}))
    return moves, sorted(prefixes.items(), key=lambda row: -len(row[0]))


def resolve_recorded(path):
    path = Path(path)
    candidate = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    if not candidate.is_relative_to(ROOT):
        return candidate
    relative = candidate.relative_to(ROOT).as_posix()
    moves, prefixes = mappings()
    seen = set()
    while relative not in seen:
        seen.add(relative)
        updated = moves.get(relative)
        if updated is None:
            updated = next((new+relative[len(old):] for old, new in prefixes
                            if relative == old or relative.startswith(old+'/')), relative)
        if updated == relative:
            break
        relative = updated
    result = (ROOT/relative).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError('recorded path mapping escapes repository')
    return result


def research_source(name):
    """Old FUBA helper names remain in historical source-hash inventories."""
    return resolve_recorded('FUBA/research/'+name)


def snapshot_file(root, name):
    """Accept both flat and packaged recorder snapshots, never paths outside them."""
    root, relative = Path(root).resolve(), Path(name)
    path = (root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'invalid training snapshot path: {name}')
    return path
