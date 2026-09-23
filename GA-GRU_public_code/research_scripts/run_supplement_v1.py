"""Additional-depression within-well evaluation using the main-study configurations.

Spyder: run this file without arguments to prepare, resume evaluation, and summarize.
No new hyperparameter search or modification of the main experiment is performed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime

PROJECT = Path(os.environ.get('GAGRU_RESEARCH_PROJECT', Path(__file__).resolve().parent))
DATA_ROOT = Path(os.environ.get('GAGRU_SUPPLEMENT_DATA', Path(__file__).resolve().parent.parent / 'private_data' / 'supplement'))
ROOT = DATA_ROOT / '补充实验_v1'
PROTOCOL_DIR = ROOT / 'protocol'
PLAN_PATH = PROTOCOL_DIR / 'evaluation_plan.json'
WELLS = ['乌20','乌22','希9','乌28','希18']
EXCLUDED_WELLS = {'乌21':'MSFL/LLS原始曲线重复待核查，无已整理七曲线样本',
                  '希3':'按主实验每类至少40点和5个标注区间的规则，仅一类满足条件'}
REPEATS = [(20260917,17),(20260918,29),(20260919,43)]
FEATURES = ['MSFL','LLS','LLD','DEN','DT','GR','NPHI']
for directory in [ROOT,ROOT/'tmp',ROOT/'runs',ROOT/'analysis',PROTOCOL_DIR]:
    directory.mkdir(parents=True,exist_ok=True)
os.environ['TMP']=str(ROOT/'tmp')
os.environ['TEMP']=str(ROOT/'tmp')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
sys.path.insert(0,str(PROJECT))

import numpy as np
import pandas as pd
import torch
from gagru.random_center import (add_source_row_ids, eligible_center_ids, stratified_center_assignment,
                                build_random_center_windows,center_and_context_overlap_audit)
from gagru.single_well import resegment_after_class_filter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def csv(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temp,index=False,encoding='utf-8-sig')
    os.replace(temp,path)


def load_engine():
    path=PROJECT/'80_run_final_repeated_evaluation.py'
    spec=importlib.util.spec_from_file_location('supplement_main_evaluation',path)
    engine=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=engine
    spec.loader.exec_module(engine)
    engine.PROJECT_DIR=ROOT
    engine.PROTOCOL_DIR=PROTOCOL_DIR
    return engine


def source_csv(well):
    folder='新增两井整理_20260921' if well in ['乌28','希18'] else '新增五井整理_20260921'
    return DATA_ROOT/folder/'02_已整理标注样本'/f'{well}_七曲线.csv'


def prepare():
    if PLAN_PATH.exists():
        plan=json.loads(PLAN_PATH.read_text(encoding='utf-8'))
        check_sources(plan)
        print('Existing supplementary protocol verified.',flush=True)
        return plan
    original_path=PROJECT/'outputs/independent_wells_v5/final_evaluation/final_evaluation_plan.json'
    original=json.loads(original_path.read_text(encoding='utf-8'))
    expected_models=['ga_gru','random_search_gru','tpe_gru','fixed_gru','tuned_rnn','tuned_lstm','extra_trees']
    assert [m['model_id'] for m in original['models']]==expected_models
    models=copy.deepcopy(original['models'])
    for m in models:
        m['supplement_configuration_policy']='hyperparameters inherited from main study; all weights/trees fitted on this well'
    well_protocols={};counts=[];class_rows=[];overlaps=[];input_hashes={};file_hashes={};overview=[]
    for well in WELLS:
        path=source_csv(well)
        input_hashes[str(path)]=digest(path)
        f=pd.read_csv(path).rename(columns={'depth_m':'depth'})
        assert not f.empty and set(f.well_id)=={well}
        assert not f.depth.duplicated().any()
        f['class_id']=pd.factorize(f.lithology_group,sort=True)[0]
        all_classes=f.groupby(['class_id','lithology_group'],as_index=False).agg(
            rows=('depth','size'),independent_intervals=('interval_id','nunique'))
        all_classes['eligible']=(all_classes.rows>=40)&(all_classes.independent_intervals>=5)
        selected=all_classes.loc[all_classes.eligible,'class_id'].astype(int).tolist()
        assert len(selected)>=2,(well,'insufficient classes')
        g=add_source_row_ids(resegment_after_class_filter(f[f.class_id.isin(selected)].copy()))
        eligible=g[g.source_row_id.isin(eligible_center_ids(g,9))]
        all_classes['eligible_L9_centers']=all_classes.class_id.map(eligible.class_id.value_counts()).fillna(0).astype(int)
        assert all(all_classes.loc[all_classes.eligible,'eligible_L9_centers']>=3)
        class_map={int(r.class_id):r.lithology_group for r in all_classes.itertuples()}
        for r in all_classes.to_dict('records'):
            class_rows.append({'well_id':well,**r})
        snapshot=PROTOCOL_DIR/'data'/well/f'{well}_eligible_source.csv'
        csv(snapshot,g)
        file_hashes[str(snapshot)]=digest(snapshot)
        split_records={}
        for split_seed,training_seed in REPEATS:
            assignment=stratified_center_assignment(g,window_length=9,seed=split_seed)
            windows=build_random_center_windows(g,assignment,FEATURES,9)
            audit=center_and_context_overlap_audit(windows)
            assert audit['center_overlap_across_splits']==0
            hashes={}
            for split in ['train','validation','test']:
                part=assignment[assignment.split.eq(split)]
                p=PROTOCOL_DIR/'center_assignments'/f'seed_{split_seed}'/well/f'{well}_{split}_centers.csv'
                csv(p,part);hashes[split]=digest(p);file_hashes[str(p)]=hashes[split]
                for cid in selected:
                    count=int(part.class_id.eq(cid).sum())
                    assert count >= (2 if split=='train' else 1),(well,split,cid,count)
                    counts.append({'well_id':well,'split_seed':split_seed,'training_seed':training_seed,
                                   'split':split,'class_id':cid,'class_name':class_map[cid],'centers':count})
            split_records[str(split_seed)]={'assignment_sha256':hashes,'centers':{s:len(windows[s].y) for s in windows}}
            overlaps.append({'well_id':well,'split_seed':split_seed,**audit})
        well_protocols[well]={'included_classes':selected,'class_names':{str(c):class_map[c] for c in selected},
            'source_snapshot':str(snapshot.relative_to(PROTOCOL_DIR)),'source_snapshot_sha256':digest(snapshot),
            'source_rows':len(g),'prepared_rows':len(f),'excluded_sparse_class_rows':len(f)-len(g),
            'split_seeds':split_records,'depression':str(f.depression.iloc[0]),
            'evidence_role':'core' if len(eligible)>=500 else 'small_sample_descriptive',
            'eligible_centers':len(eligible)}
        overview.append({'well_id':well,'depression':str(f.depression.iloc[0]),'prepared_rows':len(f),
                         'eligible_rows':len(g),'classes':len(selected),'L9_centers':len(eligible),
                         'train_centers':len(windows['train'].y),'validation_centers':len(windows['validation'].y),
                         'test_centers':len(windows['test'].y),'evidence_role':well_protocols[well]['evidence_role']})
        print('Prepared',json.dumps(overview[-1],ensure_ascii=False),flush=True)
    code_paths=[Path(__file__),PROJECT/'80_run_final_repeated_evaluation.py',*sorted((PROJECT/'gagru').glob('*.py'))]
    plan={'status':'FROZEN_BEFORE_NEW_WELL_MODEL_RESULTS',
          'created_at':datetime.now().astimezone().isoformat(timespec='seconds'),
          'task':'additional-depression, separately retrained within-well random-center interpolation',
          'scope':'preselected main-study hyperparameter configurations applied without new search',
          'cross_well_transfer':False,'weights_shared_across_wells':False,
          'new_optimizer_search_performed':False,'new_test_outcomes_used_for_configuration_selection':False,
          'parent_main_plan':str(original_path),'parent_main_plan_sha256':digest(original_path),
          'models':models,'wells':WELLS,'excluded_wells':EXCLUDED_WELLS,'well_protocols':well_protocols,
          'features':FEATURES,'window_length':9,'representation':'engineered','imbalance_strategy':'smote_tomek',
          'class_eligibility':{'minimum_rows':40,'minimum_source_intervals':5,'minimum_classes_per_well':2,
                               'inherited_from_main_study':True,'new_lithologies_allowed_if_eligible':True},
          'class_code_scope':'class_id is a well-specific group code in this supplement; thesis and historical codes retained separately',
          'paired_repeats':[{'repeat_id':i+1,'split_seed':s,'training_seed':t} for i,(s,t) in enumerate(REPEATS)],
          'repeat_interpretation':'three paired split/training repetitions, not a 3x3 factorial; no independent-sample inference from repetitions',
          'fractions':{'train':.70,'validation':.15,'test':.15},
          'batch_size':512,'maximum_selection_epochs':60,'selection_patience':8,
          'selection_metric':'validation macro-F1 for epoch only',
          'refit':'refit from scratch on train+validation for validation-selected epoch count',
          'preprocessing':'selection statistics on train; refit statistics on train+validation; test transform-only',
          'balance':'SMOTE-Tomek on training inputs only, same resampled arrays across all seven models',
          'overlap_disclosure':'target centers disjoint; observed curve context can overlap; no unseen-well or unseen-continuous-interval claim',
          'evidence_role_rule':'under 500 eligible centers: small-sample descriptive; applied before training',
          'reporting':'per-well mean and sample SD over three paired repeats; equal-well means by depression and evidence role; retain every result',
          'majority_reference':'training+validation majority class before balancing, evaluated on the same test centers',
          'timing':'new-well epoch selection, refit and inference separately; historical hyperparameter search not charged as new search',
          'expected_runs':len(WELLS)*len(models)*len(REPEATS),'input_hashes':input_hashes,
          'protocol_file_hashes':file_hashes,'code_hashes':{str(p):digest(p) for p in code_paths}}
    csv(PROTOCOL_DIR/'well_eligibility.csv',pd.DataFrame(overview))
    csv(PROTOCOL_DIR/'class_eligibility.csv',pd.DataFrame(class_rows))
    csv(PROTOCOL_DIR/'partition_class_counts.csv',pd.DataFrame(counts))
    write_json(PROTOCOL_DIR/'context_overlap_audit.json',overlaps)
    write_json(PLAN_PATH,plan)
    check_sources(plan)
    return plan


def check_sources(plan):
    for key in ['input_hashes','protocol_file_hashes','code_hashes']:
        for path,expected in plan[key].items():
            if digest(path)!=expected:
                raise RuntimeError(f'Frozen input/code changed: {path}')
    assert digest(plan['parent_main_plan'])==plan['parent_main_plan_sha256']


def result_files():
    return sorted((ROOT/'runs').glob('repeat_*/*/*/result.json'))


def evaluate(plan):
    check_sources(plan)
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    engine=load_engine()
    environment={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
                 'device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                 'cpu_threads':torch.get_num_threads(),'start_at':datetime.now().astimezone().isoformat(timespec='seconds')}
    write_json(ROOT/'environment.json',environment)
    allpaths=[];started=time.perf_counter();new_runs=0
    for rep in plan['paired_repeats']:
        rid=rep['repeat_id'];split_seed=rep['split_seed'];seed=rep['training_seed']
        for wi,well in enumerate(plan['wells']):
            pending=[]
            for spec in plan['models']:
                directory=ROOT/'runs'/f'repeat_{rid}'/well/spec['model_id']
                resultpath=directory/'result.json';allpaths.append(resultpath)
                if resultpath.exists():
                    saved=json.loads(resultpath.read_text(encoding='utf-8'))
                    engine.validate_existing_result(saved,model_id=spec['model_id'],well_id=well,split_seed=split_seed,training_seed=seed)
                    assert saved['supplement_plan_sha256']==digest(PLAN_PATH)
                else:
                    pending.append((spec,directory,resultpath))
            if not pending:
                continue
            print(f'Preparing task repeat={rid}, well={well}',flush=True)
            snapshot=PROTOCOL_DIR/plan['well_protocols'][well]['source_snapshot']
            frame=pd.read_csv(snapshot)
            balance_seed=27101+(rid-1)*100+wi
            preparation_start=time.perf_counter()
            task=engine.prepare_task(plan,plan,frame,well,split_seed,balance_seed)
            prep_seconds=time.perf_counter()-preparation_start
            assert np.isfinite(task['refit_X']).all() and np.isfinite(task['test_X']).all()
            print(f'Task ready {well}: balanced train={len(task["selection_y_train"])}, refit={len(task["refit_y"])}, test={len(task["test_y"])}, preprocessing={prep_seconds:.2f}s',flush=True)
            audit_path=ROOT/'task_audits'/f'repeat_{rid}'/f'{well}.json'
            audit={**task['audit'],'well_id':well,'repeat_id':rid,'preprocessing_seconds':prep_seconds,
                   'test_center_ids':task['test_windows'].center_ids.tolist(),'plan_sha256':digest(PLAN_PATH)}
            write_json(audit_path,audit)
            before=task['audit']['refit_balance']['class_counts_before']
            majority=int(max(sorted(before),key=lambda k:before[k]))
            majority_prediction=np.full(len(task['test_y']),majority,dtype=int)
            majority_report=engine.classification_report(task['test_y'],majority_prediction,task['global_classes'])
            write_json(ROOT/'majority_references'/f'repeat_{rid}'/f'{well}.json',{
                'well_id':well,'repeat_id':rid,'training_majority_local_class':majority,
                'metrics':majority_report,'test_samples':len(task['test_y'])})
            for spec,directory,resultpath in pending:
                directory.mkdir(parents=True,exist_ok=True)
                run_start=time.perf_counter()
                print(f'START repeat={rid} well={well} model={spec["model_id"]}',flush=True)
                if spec['family']=='recurrent_neural_network':
                    run=engine.recurrent_run(spec,task,device=device,training_seed=seed,batch_size=plan['batch_size'],
                        maximum_epochs=plan['maximum_selection_epochs'],patience=plan['selection_patience'],directory=directory)
                else:
                    run=engine.extra_trees_run(spec,task,training_seed=seed)
                prediction=np.asarray(run.pop('prediction'),dtype=np.int64)
                metrics=engine.classification_report(task['test_y'],prediction,task['global_classes'])
                for row in metrics['per_class']:
                    row['class_name']=plan['well_protocols'][well]['class_names'][str(row['global_class_id'])]
                predictions=directory/'test_predictions.csv'
                engine.save_predictions(predictions,task,prediction,model_id=spec['model_id'],well_id=well,split_seed=split_seed,training_seed=seed)
                result={'status':'COMPLETE','model_id':spec['model_id'],'display_name':spec['display_name'],
                        'well_id':well,'repeat_id':rid,'split_seed':split_seed,'training_seed':seed,
                        'depression':plan['well_protocols'][well]['depression'],
                        'evidence_role':plan['well_protocols'][well]['evidence_role'],
                        'class_count':len(task['global_classes']),'test_samples':len(task['test_y']),
                        'model_spec':spec,'metrics':metrics,'majority_reference_accuracy':majority_report['accuracy'],
                        'supplement_plan_sha256':digest(PLAN_PATH),
                        'predictions_file':str(predictions.relative_to(ROOT)), 'predictions_sha256':digest(predictions),
                        'total_wall_seconds':time.perf_counter()-run_start,'finished_at':datetime.now().astimezone().isoformat(timespec='seconds'),**run}
                write_json(resultpath,result)
                new_runs+=1
                completed=len(result_files())
                elapsed=time.perf_counter()-started
                write_json(ROOT/'progress.json',{'completed':completed,'expected':plan['expected_runs'],
                    'last_run':{'well':well,'repeat':rid,'model':spec['model_id']},'elapsed_this_invocation_seconds':elapsed,
                    'rough_remaining_seconds':elapsed/max(1,new_runs)*(plan['expected_runs']-completed),
                    'updated_at':datetime.now().astimezone().isoformat(timespec='seconds')})
                print(f'DONE {completed}/{plan["expected_runs"]} repeat={rid} {well} {spec["model_id"]}: accuracy={metrics["accuracy"]:.4f}, macroF1={metrics["macro_f1"]:.4f}, selected_epoch={run["selected_epoch"]}, seconds={result["total_wall_seconds"]:.1f}',flush=True)
            del task
            if device.type=='cuda':torch.cuda.empty_cache()
    assert len(allpaths)==plan['expected_runs'] and all(p.exists() for p in allpaths)
    write_json(ROOT/'execution_completion.json',{'status':'COMPLETE','expected_runs':plan['expected_runs'],
        'completed_runs':len(allpaths),'plan_sha256':digest(PLAN_PATH),'result_hashes':{str(p.relative_to(ROOT)):digest(p) for p in allpaths},
        'completed_at':datetime.now().astimezone().isoformat(timespec='seconds')})
    return analyze(plan)


def analyze(plan):
    engine=load_engine()
    resultpaths=result_files()
    if len(resultpaths)!=plan['expected_runs']:
        raise RuntimeError(f'Incomplete experiment: {len(resultpaths)}/{plan["expected_runs"]}')
    rows=[];classes=[];comparisons=[]
    for p in resultpaths:
        r=json.loads(p.read_text(encoding='utf-8'))
        engine.validate_existing_result(r,model_id=r['model_id'],well_id=r['well_id'],split_seed=r['split_seed'],training_seed=r['training_seed'])
        predictions=pd.read_csv(ROOT/r['predictions_file'])
        check=engine.classification_report(predictions.true_local_class_id.to_numpy(),predictions.predicted_local_class_id.to_numpy(),
                                           tuple(plan['well_protocols'][r['well_id']]['included_classes']))
        for metric in ['accuracy','macro_f1','balanced_accuracy','weighted_f1']:
            assert np.isclose(check[metric],r['metrics'][metric],rtol=0,atol=1e-12)
        row={k:r[k] for k in ['well_id','repeat_id','split_seed','training_seed','depression','evidence_role','model_id','display_name','class_count','test_samples','majority_reference_accuracy',
                              'selected_epoch','selection_epochs_completed','selection_runtime_seconds','refit_runtime_seconds','inference_runtime_seconds','trainable_parameters']}
        row.update({m:r['metrics'][m] for m in ['accuracy','macro_f1','balanced_accuracy','weighted_f1']})
        row['fit_total_seconds']=r['selection_runtime_seconds']+r['refit_runtime_seconds']
        rows.append(row)
        classes.extend({'well_id':r['well_id'],'model_id':r['model_id'],'repeat_id':r['repeat_id'],**c} for c in r['metrics']['per_class'])
    frame=pd.DataFrame(rows)
    assert not frame.duplicated(['well_id','repeat_id','model_id']).any()
    for (w,rid),part in frame.groupby(['well_id','repeat_id']):
        arrays=[pd.read_csv(ROOT/'runs'/f'repeat_{rid}'/w/m/'test_predictions.csv') for m in part.model_id]
        base=arrays[0][['center_row_id','true_local_class_id','depth']]
        assert all(base.equals(x[['center_row_id','true_local_class_id','depth']]) for x in arrays[1:])
    csv(ROOT/'analysis/all_runs.csv',frame)
    csv(ROOT/'analysis/per_class_runs.csv',pd.DataFrame(classes))
    metrics=['accuracy','macro_f1','balanced_accuracy','weighted_f1','fit_total_seconds','inference_runtime_seconds']
    keys=['well_id','depression','evidence_role','model_id','display_name','class_count','test_samples']
    grouped=frame.groupby(keys,as_index=False).agg(**{f'{m}_{stat}':pd.NamedAgg(column=m,aggfunc=stat) for m in metrics for stat in ['mean','std']},
                                                 runs=('repeat_id','size'),majority_accuracy_mean=('majority_reference_accuracy','mean'))
    assert grouped.runs.eq(3).all()
    csv(ROOT/'analysis/per_well_summary.csv',grouped)
    groups=grouped.groupby(['depression','evidence_role','model_id'],as_index=False).agg(
        wells=('well_id','nunique'),accuracy_mean=('accuracy_mean','mean'),macro_f1_mean=('macro_f1_mean','mean'),
        balanced_accuracy_mean=('balanced_accuracy_mean','mean'),mean_fit_seconds=('fit_total_seconds_mean','mean'))
    csv(ROOT/'analysis/depression_summary.csv',groups)
    for well,part in grouped.groupby('well_id'):
        ga=part[part.model_id.eq('ga_gru')].iloc[0]
        for r in part[~part.model_id.eq('ga_gru')].itertuples():
            comparisons.append({'well_id':well,'evidence_role':ga.evidence_role,'comparator':r.model_id,
                'accuracy_gain_percentage_points':100*(ga.accuracy_mean-r.accuracy_mean),
                'macro_f1_gain_percentage_points':100*(ga.macro_f1_mean-r.macro_f1_mean),
                'fit_time_ratio_ga_over_comparator':ga.fit_total_seconds_mean/r.fit_total_seconds_mean})
    csv(ROOT/'analysis/ga_paired_differences.csv',pd.DataFrame(comparisons))
    draw_results(grouped,plan)
    write_json(ROOT/'analysis/verification.json',{'status':'PASS','runs':len(frame),
        'all_metrics_recomputed_from_saved_predictions':True,'same_test_centers_across_models':True,
        'repeat_count_per_model_well':3,'held_out_target_overlap':0,
        'context_overlap_expected_and_reported':True,'new_hyperparameter_search':False})
    print('FINAL SUMMARY',flush=True)
    print(grouped[grouped.model_id.isin(['ga_gru','fixed_gru','extra_trees'])][['well_id','model_id','test_samples','accuracy_mean','macro_f1_mean','fit_total_seconds_mean']].to_string(index=False),flush=True)
    return grouped


def draw_results(frame,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    aliases={'乌20':'Wu20','乌22':'Wu22','希9':'Xi9','乌28':'Wu28','希18':'Xi18'}
    colors=['#B94040','#376E95','#38816A','#8B6DA4','#9E7950','#697C84','#D2A62C']
    fig,axes=plt.subplots(2,1,figsize=(11,7.4),sharex=True,layout='constrained')
    wells=plan['wells'];models=plan['models'];x=np.arange(len(wells));width=.105
    for ax,metric,title in zip(axes,['accuracy','macro_f1'],['Test accuracy (%)','Test macro-F1 (%)']):
        for j,spec in enumerate(models):
            part=frame[frame.model_id.eq(spec['model_id'])].set_index('well_id').loc[wells]
            ax.bar(x+(j-3)*width,100*part[f'{metric}_mean'],width,color=colors[j],label=spec['display_name'],
                   yerr=100*part[f'{metric}_std'],error_kw={'elinewidth':.7,'capsize':2})
        ax.set_ylim(0,105);ax.set_ylabel(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.axvline(2.5,color='#555555',linestyle=':',linewidth=1)
    axes[0].legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.25),frameon=False,fontsize=9)
    axes[-1].set_xticks(x,[aliases[w]+(' *' if plan['well_protocols'][w]['evidence_role']!='core' else '') for w in wells])
    axes[-1].set_xlabel('* Small sample cases. Error bars: SD over three paired split/training repetitions.')
    fig.savefig(ROOT/'analysis/model_comparison.png',dpi=220)
    fig.savefig(ROOT/'analysis/model_comparison.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','run','analyze','all'],default='all')
    args,_=parser.parse_known_args()
    plan=prepare()
    if args.stage in ['all','run']:
        evaluate(plan)
    elif args.stage=='analyze':
        analyze(plan)


if __name__=='__main__':
    main()
