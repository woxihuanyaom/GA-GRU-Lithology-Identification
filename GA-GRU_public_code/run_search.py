"""Portable equal-budget GA/random/TPE search using the original operators."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
import os
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
import numpy as np
import pandas as pd
import torch

from release_support import ROOT, load_research, new_output, prepare_inputs, read_config, write_json
from gagru.local_recurrent import fit_local_recurrent_with_validation
from gagru.search import GRUSearchCandidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true', help='One synthetic well, one seed, seven candidates, one epoch')
    source.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args(argv)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')
    protocol = read_config('repeated_search.json')
    if args.demo:
        protocol['repeats'] = protocol['repeats'][:1]
        protocol['candidate_budget_per_method_per_repeat'] = 7
        protocol['maximum_epochs'] = 1
        protocol['early_stopping_patience'] = 1
    output = new_output(args.output or ROOT / 'outputs' / ('search_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    source_protocol, frames = prepare_inputs(args.data_dir, output, [protocol['split_seed']], demo=args.demo)
    protocol['wells'] = list(frames)
    protocol['balance_seed_by_well'] = {w: protocol['balance_seed_base'] + i for i, w in enumerate(frames)}
    protocol['training_seed_by_well'] = {w: protocol['training_seed_base'] + i for i, w in enumerate(frames)}
    protocol['source_snapshot_sha256'] = {
        str(Path('protocol') / record['source_snapshot']): record['source_snapshot_sha256']
        for record in source_protocol['well_protocols'].values()
    }
    write_json(output / 'search_config.json', {'synthetic': args.demo, 'device': args.device, **protocol})
    engine = load_research('114_run_repeated_optimizer_searches.py')
    engine.PROJECT_DIR = output
    engine.SOURCE_PROTOCOL_DIR = output / 'protocol'
    tasks, audit = engine.prepare_tasks(protocol, source_protocol)
    write_json(output / 'search_preparation_audit.json', audit)
    device = torch.device(args.device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    space = engine.search_space(protocol['search_space'])
    cache, summaries = {}, []
    for repeat in protocol['repeats']:
        initial = [GRUSearchCandidate.from_dict(c) for c in repeat['shared_initial_candidates']]
        for method in protocol['methods']:
            records = []
            directory = output / f"repeat_{repeat['repeat_id']}" / method
            directory.mkdir(parents=True)

            def evaluate(candidate, provenance):
                reused = candidate.key in cache
                if reused:
                    record = copy.deepcopy(cache[candidate.key])
                    record['actual_training_runtime_seconds'] = 0.0
                else:
                    per_well = []
                    for well, task in tasks.items():
                        fit = fit_local_recurrent_with_validation(
                            'gru', candidate, task['X_train'], task['y_train'], task['X_validation'], task['y_validation'],
                            device=device, seed=task['training_seed'], batch_size=protocol['batch_size'],
                            max_epochs=protocol['maximum_epochs'], patience=protocol['early_stopping_patience'])
                        pd.DataFrame(fit.history).to_csv(directory / f'candidate_{len(records)+1:03d}_{well}_history.csv', index=False)
                        per_well.append({'well_id': well, 'macro_f1': fit.validation_macro_f1,
                                         'accuracy': fit.validation_accuracy, 'balanced_accuracy': fit.validation_balanced_accuracy,
                                         'runtime_seconds': fit.runtime_seconds, 'trainable_parameters': fit.trainable_parameters,
                                         'best_epoch': fit.best_epoch})
                    runtime = float(sum(row['runtime_seconds'] for row in per_well))
                    record = {'candidate_key': candidate.key, 'candidate': candidate.to_dict(),
                              'mean_per_well_macro_f1': float(np.mean([r['macro_f1'] for r in per_well])),
                              'mean_per_well_accuracy': float(np.mean([r['accuracy'] for r in per_well])),
                              'mean_per_well_balanced_accuracy': float(np.mean([r['balanced_accuracy'] for r in per_well])),
                              # Original operator API field name; sums over the supplied wells.
                              'total_parameters_across_six_models': int(sum(r['trainable_parameters'] for r in per_well)),
                              'charged_training_runtime_seconds': runtime, 'actual_training_runtime_seconds': runtime,
                              'per_well': per_well}
                    cache[candidate.key] = copy.deepcopy(record)
                record.update({'candidate_index': len(records) + 1,
                               'provenance': {**provenance, 'evaluation_reused': reused}})
                records.append(record)
                write_json(directory / 'candidate_evaluations.json', records)
                print(f"Repeat {repeat['repeat_id']} {method}: {len(records)}/{protocol['candidate_budget_per_method_per_repeat']}", flush=True)
                return record

            options = {'space': space, 'budget': protocol['candidate_budget_per_method_per_repeat'],
                       'seed': repeat['method_search_seed'], 'evaluate': evaluate}
            started = time.perf_counter()
            if method == 'genetic_algorithm':
                evaluated = engine.run_ga(initial, settings=protocol['ga'], **options)
            elif method == 'random_search':
                evaluated = engine.run_random(initial, **options)
            else:
                evaluated = engine.run_tpe(initial, startup_candidates=protocol['tpe']['startup_candidates'], **options)
            winner = max(evaluated, key=engine.rank)
            summary = {'repeat_id': repeat['repeat_id'], 'method': method, 'synthetic': args.demo,
                       'candidates': len(evaluated), 'best_validation_macro_f1': winner['mean_per_well_macro_f1'],
                       'winner': winner['candidate'], 'elapsed_wall_seconds': time.perf_counter() - started,
                       'accounted_fit_validation_seconds': sum(r['charged_training_runtime_seconds'] for r in evaluated),
                       'actual_fit_validation_seconds': sum(r['actual_training_runtime_seconds'] for r in evaluated)}
            write_json(directory / 'summary.json', summary)
            summaries.append(summary)
    write_json(output / 'search_summary.json', {'status': 'COMPLETE', 'synthetic': args.demo,
                                              'test_metrics_used': False, 'searches': summaries})
    print('Completed:', output)
    if args.demo:
        print('Synthetic software test only; shortened budgets are not manuscript experiments.')


if __name__ == '__main__':
    main()
