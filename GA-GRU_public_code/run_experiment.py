"""Run the published configurations on synthetic or locally authorized CSV files."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
import pandas as pd
import torch

from release_support import ROOT, load_research, new_output, prepare_inputs, read_config, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true', help='Artificial data, two epochs, one repeat; not paper results')
    source.add_argument('--data-dir', type=Path, help='One normalized and authorized CSV per well')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--models', nargs='+', help='Model IDs from configs/main_experiment.json')
    args = parser.parse_args(argv)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')
    plan = read_config('main_experiment.json')
    if args.demo:
        plan['split_seeds'] = [plan['split_seeds'][0]]
        plan['training_seeds'] = [17]
        plan['maximum_selection_epochs'] = 2
        plan['selection_patience'] = 2
    selected = args.models or (['ga_gru', 'fixed_gru', 'extra_trees'] if args.demo else [m['model_id'] for m in plan['models']])
    available = {m['model_id'] for m in plan['models']}
    if set(selected) - available:
        parser.error(f'Unknown model IDs: {sorted(set(selected) - available)}')
    plan['models'] = [m for m in plan['models'] if m['model_id'] in selected]
    output = new_output(args.output or ROOT / 'outputs' / ('demo_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    write_json(output / 'run_config.json', {'synthetic': args.demo, 'device': args.device, **plan})
    protocol, frames = prepare_inputs(args.data_dir, output, plan['split_seeds'], demo=args.demo)
    engine = load_research('80_run_final_repeated_evaluation.py')
    engine.PROJECT_DIR = output
    engine.PROTOCOL_DIR = output / 'protocol'
    device = torch.device(args.device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    results = []
    for split_index, split_seed in enumerate(plan['split_seeds']):
        for well_index, (well, frame) in enumerate(frames.items()):
            task = engine.prepare_task(protocol, plan, frame, well, split_seed, 27101 + 100 * split_index + well_index)
            write_json(output / 'audits' / f'{well}_{split_seed}.json', task['audit'])
            for spec in plan['models']:
                for seed in plan['training_seeds']:
                    directory = output / 'runs' / f'split_{split_seed}' / well / spec['model_id'] / f'seed_{seed}'
                    directory.mkdir(parents=True)
                    if spec['family'] == 'recurrent_neural_network':
                        result = engine.recurrent_run(spec, task, device=device, training_seed=seed,
                                                      batch_size=plan['batch_size'], maximum_epochs=plan['maximum_selection_epochs'],
                                                      patience=plan['selection_patience'], directory=directory)
                    else:
                        result = engine.extra_trees_run(spec, task, training_seed=seed)
                    prediction = result.pop('prediction')
                    report = engine.classification_report(task['test_y'], prediction, task['global_classes'])
                    identifiers = {'model_id': spec['model_id'], 'well_id': well, 'split_seed': split_seed, 'training_seed': seed}
                    engine.save_predictions(directory / 'predictions.csv', task, prediction, **identifiers)
                    write_json(directory / 'result.json', {'synthetic': args.demo, **identifiers, **report, **result})
                    results.append({**identifiers, **{k: report[k] for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'weighted_f1')}})
                    print(f"{well} {spec['model_id']} split={split_seed} seed={seed}: accuracy={report['accuracy']:.4f}", flush=True)
    pd.DataFrame(results).to_csv(output / 'run_metrics.csv', index=False)
    per_well = pd.DataFrame(results).groupby(['well_id', 'model_id'], as_index=False)[['accuracy', 'macro_f1', 'balanced_accuracy', 'weighted_f1']].mean()
    per_well.to_csv(output / 'per_well_metrics.csv', index=False)
    per_well.groupby('model_id')[['accuracy', 'macro_f1']].agg(['mean', 'std']).to_csv(output / 'model_summary.csv')
    write_json(output / 'completion.json', {'status': 'COMPLETE', 'synthetic': args.demo, 'runs': len(results),
                                          'interpretation': 'Random-center within-well interpolation; measured contexts can overlap across partitions.',
                                          'paper_reproduction': False})
    print('Completed:', output)
    if args.demo:
        print('Synthetic software test only. These scores are not scientific evidence.')


if __name__ == '__main__':
    main()
