"""Portable I/O adapters; scientific transformations remain in gagru."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from gagru.random_center import (
    add_source_row_ids, stratified_center_assignment, build_random_center_windows,
    center_and_context_overlap_audit,
)
from gagru.single_well import resegment_after_class_filter

ROOT = Path(__file__).resolve().parent
FEATURES = ['MSFL', 'LLS', 'LLD', 'DEN', 'DT', 'GR', 'NPHI']


def read_config(name):
    return json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_research(filename):
    path = ROOT / 'research_scripts' / filename
    name = 'release_' + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def new_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output is not empty. Choose a new output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def synthetic_frame():
    """Generate artificial values independently of all proprietary records."""
    rng = np.random.default_rng(20260923)
    intervals = np.repeat(np.arange(15), 48)
    labels = intervals % 3
    n = len(labels)
    base = 0.7 + 0.5 * labels + rng.normal(0, 0.1, n)
    frame = pd.DataFrame({
        'well_id': 'SYNTHETIC_01', 'depth': 1000 + np.arange(n) * 0.125,
        'segment_id': 'SYNTHETIC_CONTINUOUS',
        'interval_id': [f'ARTIFICIAL_{i:02d}' for i in intervals], 'class_id': labels,
        'MSFL': 10 ** base,
        'LLS': 10 ** (base + 0.1 + rng.normal(0, 0.03, n)),
        'LLD': 10 ** (base + 0.2 + rng.normal(0, 0.03, n)),
        'DEN': 2.3 + 0.13 * labels + rng.normal(0, 0.04, n),
        'DT': 90 - 8 * labels + rng.normal(0, 3, n),
        'GR': 30 + 25 * labels + rng.normal(0, 8, n),
        'NPHI': 0.1 + 0.04 * labels + rng.normal(0, 0.015, n),
    })
    return frame


def validate_frame(raw):
    required = ['well_id', 'depth', 'segment_id', 'interval_id', 'class_id', *FEATURES]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    f = raw[required].copy()
    if f.empty or f[['well_id', 'segment_id', 'interval_id', 'depth', 'class_id']].isna().any().any():
        raise ValueError('Empty input or missing identifiers, depths, or labels')
    if f.well_id.astype(str).nunique() != 1:
        raise ValueError('Each CSV must contain exactly one well')
    well = str(f.well_id.iloc[0])
    if not re.fullmatch(r'[\w-]+', well) or well.upper() in {'CON', 'PRN', 'AUX', 'NUL'}:
        raise ValueError('Use a well ID containing letters, numbers, underscores or hyphens')
    for col in ['depth', 'class_id', *FEATURES]:
        f[col] = pd.to_numeric(f[col], errors='raise')
    if not np.isfinite(f.depth).all() or f.depth.duplicated().any():
        raise ValueError('Depth must be finite and unique within the well')
    if not np.isfinite(f.class_id).all() or (f.class_id < 0).any() or (f.class_id % 1 != 0).any():
        raise ValueError('class_id must be a nonnegative integer; never use missing-value codes as labels')
    f['class_id'] = f.class_id.astype(np.int64)
    if np.isinf(f[FEATURES].to_numpy(float)).any():
        raise ValueError('Infinite curve values are invalid; use NaN for missing measurements')
    if (f[['MSFL', 'LLS', 'LLD']] <= 0).any().any():
        raise ValueError('Finite resistivity values must be positive')
    if f[FEATURES].isna().all().any():
        raise ValueError('An entire curve is missing; this seven-curve workflow cannot use it')
    if f.groupby('interval_id').class_id.nunique().gt(1).any():
        raise ValueError('Each recorded interval must have one class label')
    f = f.sort_values('depth', kind='stable').reset_index(drop=True)
    runs = f.interval_id.ne(f.interval_id.shift()).cumsum()
    if pd.DataFrame({'interval': f.interval_id, 'run': runs}).groupby('interval').run.nunique().gt(1).any():
        raise ValueError('A recorded interval ID is reused at disconnected depths')
    grid = (f.depth.to_numpy(float) - float(f.depth.iloc[0])) / 0.125
    if not np.allclose(grid, np.round(grid), atol=1e-5, rtol=0):
        raise ValueError('This implementation requires a 0.125-m depth grid; gaps are allowed')
    return f


def prepare_inputs(data_dir, output, split_seeds, *, demo=False):
    source = [('generated independently; synthetic only', synthetic_frame())] if demo else [
        (str(p.resolve()), pd.read_csv(p, encoding='utf-8-sig'))
        for p in sorted(Path(data_dir).glob('*.csv'))
    ]
    if not source:
        raise ValueError('No CSV files found in data directory')
    protocol = {'well_protocols': {}}
    frames, audits, sources = {}, [], []
    base = output / 'protocol'
    for source_name, raw in source:
        f = validate_frame(raw)
        well = str(f.well_id.iloc[0])
        if well in frames:
            raise ValueError(f'Duplicate well ID: {well}')
        counts = f.groupby('class_id').agg(rows=('depth', 'size'), intervals=('interval_id', 'nunique'))
        classes = counts.index[(counts.rows >= 40) & (counts.intervals >= 5)].astype(int).tolist()
        if len(classes) < 2:
            raise ValueError(f'{well}: fewer than two classes meet >=40 rows and >=5 recorded intervals')
        frame = add_source_row_ids(resegment_after_class_filter(f[f.class_id.isin(classes)].copy()))
        snapshot = base / 'data' / well / f'{well}_eligible_source.csv'
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(snapshot, index=False, encoding='utf-8-sig')
        record = {'included_classes': classes,
                  'source_snapshot': snapshot.relative_to(base).as_posix(),
                  'source_snapshot_sha256': sha256(snapshot), 'source_rows': len(frame), 'split_seeds': {}}
        for seed in split_seeds:
            assignment = stratified_center_assignment(frame, window_length=9, seed=seed)
            windows = build_random_center_windows(frame, assignment, FEATURES, 9)
            hashes = {}
            for split in ('train', 'validation', 'test'):
                path = base / 'center_assignments' / f'seed_{seed}' / well / f'{well}_{split}_centers.csv'
                path.parent.mkdir(parents=True, exist_ok=True)
                assignment[assignment.split.eq(split)].to_csv(path, index=False, encoding='utf-8-sig')
                hashes[split] = sha256(path)
            overlap = center_and_context_overlap_audit(windows)
            assert overlap['center_overlap_across_splits'] == 0
            audits.append({'well_id': well, 'split_seed': seed, **overlap})
            record['split_seeds'][str(seed)] = {'assignment_sha256': hashes,
                                               'centers': {s: len(w.y) for s, w in windows.items()}}
        protocol['well_protocols'][well] = record
        frames[well] = frame
        sources.append({'source': source_name, 'sha256': None if demo else sha256(source_name),
                        'well_id': well, 'class_counts_before_filter': counts.reset_index().to_dict('records'),
                        'included_classes': classes, 'excluded_rows': len(f) - len(frame)})
    write_json(base / 'protocol.json', protocol)
    write_json(output / 'input_audit.json', {'synthetic': demo, 'sources': sources, 'context_overlap': audits})
    return protocol, frames
