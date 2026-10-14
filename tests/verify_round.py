"""Independently audit a completed research round, using CPU and trusted local files.

No training or GPU inference runs here. The sole output is a new audit report;
existing sources, results, inputs and checkpoints are read-only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'src' / 'pipeline'))
import data
from metrics import pick_threshold


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def graph_sha(incidence):
    digest = hashlib.sha256()
    digest.update(np.asarray(incidence['shape'], dtype=np.int64).tobytes())
    for key in ('indptr', 'indices', 'data'):
        digest.update(np.asarray(incidence[key]).tobytes())
    return digest.hexdigest()


def close(a, b, atol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(a, b, rtol=0, atol=atol, equal_nan=True))


def binary_values(y, p, threshold):
    if not len(y):
        return {'auc': float('nan'), 'bacc': float('nan'),
                'sen': float('nan'), 'spe': float('nan')}
    cm = confusion_matrix(y, p >= threshold, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sen = float(tp / (tp + fn)) if tp + fn else float('nan')
    spe = float(tn / (tn + fp)) if tn + fp else float('nan')
    return {'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan'),
            'bacc': (sen + spe) / 2, 'sen': sen, 'spe': spe}


def metrics_from_npz(d):
    y, val, test, p = d['y'], d['val'], d['test'], d['probability'][:, 1]
    threshold = pick_threshold(y[val], p[val])
    result = binary_values(y[test], p[test], threshold)
    result['val_auc'] = float(roc_auc_score(y[val], p[val]))
    pet = d['observed_amy'] | d['observed_tau']
    for name, has in [('pet', pet), ('nopet', ~pet)]:
        chosen = test[has[test]]
        result[f'auc_{name}'] = binary_values(y[chosen], p[chosen], threshold)['auc']
    return result, threshold


class Audit:
    def __init__(self):
        self.checked = Counter()
        self.failures = []

    def check(self, group, condition, detail):
        self.checked[group] += 1
        if not condition:
            self.failures.append({'group': group, 'detail': detail})

    def equal_array(self, group, a, b, detail):
        self.check(group, np.array_equal(a, b), detail)


def check_fitting(audit, cp, td, tag, cohort_rows):
    split = cp['split']
    train, val, test = map(lambda k: np.asarray(split[k]), ('train','val','test'))
    background = np.flatnonzero(td.y < 0)
    forbidden = np.union1d(np.union1d(val, test), background)
    audit.check('fit_scope', not np.intersect1d(train, forbidden).size, f'{tag}: train isolation')
    pp = cp['preprocessing']
    audit.equal_array('fit_scope', pp['train_rows'], train, f'{tag}: preprocessing training scope')
    eligible = np.union1d(train, background)
    audit.equal_array('mae_scope', split['mae_eligible'], eligible, f'{tag}: MAE training/background only')
    for m, artifact in cp['encoders'].items():
        rows = np.asarray(artifact['fit_rows'])
        expected = train[td.observed[m][train]]
        audit.equal_array('encoder_fit_scope', rows, expected, f'{tag}/{m}: encoder fit rows')
        audit.check('encoder_fit_scope', not np.intersect1d(rows, forbidden).size,
                    f'{tag}/{m}: encoder excludes validation/test/background')
        if artifact['kind'] == 'zero':
            audit.check('encoder_fallback', len(rows) < 2, f'{tag}/{m}: justified zero encoder')
        elif artifact['kind'] == 'pca':
            audit.check('encoder_fallback', len(artifact['components']) <= len(rows)-1,
                        f'{tag}/{m}: PCA dimension supported by training observations')
        else:
            audit.check('encoder_state', artifact['kind'] == 'mlp', f'{tag}/{m}: known encoder kind')
            audit.check('encoder_state', all(v.device.type == 'cpu' and torch.isfinite(v).all().item()
                                             for v in artifact['body_state'].values()),
                        f'{tag}/{m}: finite saved CPU encoder')

    for m, entry in pp['modalities'].items():
        pooled_expected = train[td.observed[m][train]]
        artifact_entries = [('pooled', entry['pooled'], pooled_expected)]
        if m in td.tracer:
            expected_groups = set(int(x) for x in np.unique(td.tracer[m][pooled_expected]))
            audit.check('standardizer_fit_scope', set(entry['groups']) == expected_groups,
                        f'{tag}/{m}: only training tracer groups fitted')
            for g, stats in entry['groups'].items():
                own = pooled_expected[td.tracer[m][pooled_expected] == g]
                artifact_entries.append((f'tracer{g}', stats, own if len(own)>=20 else pooled_expected))
        for name, stats, expected in artifact_entries:
            rows = np.asarray(stats['fit_rows'])
            audit.equal_array('standardizer_fit_scope', rows, expected, f'{tag}/{m}/{name}: rows')
            audit.check('standardizer_fit_scope', not np.intersect1d(rows, forbidden).size,
                        f'{tag}/{m}/{name}: excludes validation/test/background')
            if len(rows):
                x = td.blocks[m][rows].astype(np.float64)
                med = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
                med = np.where(np.isfinite(med), med, 0.)
                xf = np.where(np.isfinite(x), x, med)
                mean, std = xf.mean(0), xf.std(0)
                std = np.where(std >= 1e-8, std, 1.)
                zero = False
            else:
                d = td.blocks[m].shape[1]
                med, mean, std, zero = np.zeros(d), np.zeros(d), np.ones(d), True
            for key, target in [('median',med), ('mean',mean), ('std',std)]:
                audit.check('standardizer_numeric', np.allclose(stats[key], target, atol=1e-12, rtol=0),
                            f'{tag}/{m}/{name}: independently recomputed {key}')
            audit.check('standardizer_numeric', stats['zero_fallback'] == zero,
                        f'{tag}/{m}/{name}: explicit empty-training fallback')
    audit.equal_array('patient_alignment', split['node_ids'], cohort_rows, f'{tag}: checkpoint cohort rows')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=STUDY/'results'/'round01_core')
    args = parser.parse_args()
    out = args.out.resolve()
    if STUDY.resolve() not in out.parents:
        raise ValueError('Audit output must belong to the research folder.')
    protocol = read_json(out/'protocol.json')
    records = read_json(out/'records.json')
    summary = read_json(out/'summary.json')
    runner_report = read_json(out/'verification.json') if (out/'verification.json').exists() else None
    expected = protocol['expected_runs']
    if len(records) != expected or not summary.get('complete') or runner_report is None:
        raise SystemExit(f'Round still incomplete: {len(records)}/{expected}; no audit report written.')
    if not runner_report.get('all_passed') or runner_report.get('completed_runs') != expected:
        raise SystemExit('Runner completion audit is absent/failed; no independent audit report written.')
    torch.set_num_threads(1)  # CPU only, no CUDA initialization.
    audit = Audit()
    frozen_src = protocol['source_sha256']
    actual_src = {p.relative_to(STUDY/'src').as_posix():sha(p)
                  for p in sorted((STUDY/'src').rglob('*.py'))}
    snapshot_src = {p.relative_to(out/'source').as_posix():sha(p)
                    for p in sorted((out/'source').rglob('*.py'))}
    audit.check('source_integrity', actual_src == frozen_src, 'Current sources match frozen protocol')
    audit.check('source_integrity', snapshot_src == frozen_src, 'Archived sources match frozen protocol')
    audit.check('input_integrity', sha(protocol['cohort']) == protocol['cohort_sha256'], 'Cohort unchanged')
    origin = read_json(STUDY/'origin_manifest.json')
    for entry in origin['preserved_files']:
        path = Path(entry['path'])
        audit.check('original_integrity', path.exists() and sha(path) == entry['sha256'], str(path))
    cohort = data.load_cohort(Path(protocol['cohort']))
    task_data = {task:data.make_task(cohort, task, scope='dev') for task in protocol['tasks']}
    expected_keys = set(itertools.product(protocol['tasks'], protocol['seeds'], protocol['folds'], protocol['arms']))
    keys = [(r['task'],r['seed'],r['fold'],r['arm']) for r in records]
    audit.check('run_coverage', len(set(keys)) == len(keys) == expected, 'Unique completed run keys')
    audit.check('run_coverage', set(keys) == expected_keys, 'All predefined runs present')
    by_key = {key:r for key,r in zip(keys,records)}
    tags = {f'{t}_seed{s}_fold{f}_{a}' for t,s,f,a in expected_keys}
    audit.check('file_coverage', {p.stem for p in (out/'checkpoints').glob('*.pt')} == tags,
                'One checkpoint per predefined run')
    audit.check('file_coverage', {p.stem for p in (out/'predictions').glob('*.npz')} == tags,
                'One prediction NPZ per predefined run')
    selection_tags = {f'{t}_seed{s}_fold{f}_{a}' for t,s,f,a in expected_keys if not a.startswith('baseline_')}
    audit.check('file_coverage', {p.stem for p in (out/'selection').glob('*.json')} == selection_tags,
                'One generator selection record per module run')
    recomputed, thresholds, replay_checks, reference_splits = {}, {}, [], {}
    protocol_hash = sha(out/'protocol.json')
    for index, record in enumerate(records, 1):
        task, seed, fold, arm = key = (record['task'],record['seed'],record['fold'],record['arm'])
        tag = f'{task}_seed{seed}_fold{fold}_{arm}'
        td = task_data[task]
        try:
            with np.load(out/'predictions'/f'{tag}.npz', allow_pickle=False) as archive:
                d = {k:archive[k] for k in archive.files}
            probability = d['probability']
            audit.check('prediction_validity', probability.shape == (len(td.y),2),
                        f'{tag}: probability shape')
            audit.check('prediction_validity', np.isfinite(probability).all() and
                        np.all((probability>=0)&(probability<=1)) and
                        np.allclose(probability.sum(1),1,atol=3e-7,rtol=0), f'{tag}: finite probabilities')
            for name,target in [('y',td.y),('node_ids',td.node_ids),
                                ('observed_amy',td.observed['amy']),('observed_tau',td.observed['tau'])]:
                audit.equal_array('patient_alignment', d[name],target,f'{tag}: NPZ {name}')
            for p,q in [('train','val'),('train','test'),('val','test')]:
                audit.check('split_isolation', not np.intersect1d(d[p],d[q]).size,f'{tag}: {p}/{q}')
            metrics, threshold = metrics_from_npz(d)
            recomputed[tag], thresholds[tag] = metrics, threshold
            for m,value in metrics.items():
                audit.check('metric_recomputation', close(record[m],value), f'{tag}: {m}')
            audit.equal_array('patient_alignment', record['test_rows'],d['node_ids'][d['test']],f'{tag}: saved test cohort rows')
            audit.equal_array('patient_alignment', record['test_y'],d['y'][d['test']],f'{tag}: saved test labels')
            audit.equal_array('prediction_validity', record['test_p'],probability[d['test'],1].astype(float),f'{tag}: JSON/NPZ probabilities')
            for name in ('train','val','test'):
                field = 'validation' if name=='val' else name
                audit.check('split_sizes', record[field] == len(d[name]), f'{tag}: {field}')
            a,t = d['observed_amy'][d['test']],d['observed_tau'][d['test']]
            patterns = {'AT':a&t,'A-':a&~t,'-T':~a&t,'--':~a&~t}
            total_pattern_n = 0
            for pattern,mask in patterns.items():
                y = d['y'][d['test']][mask]; p = probability[d['test'],1][mask]
                rec = record['pet_patterns'][pattern]
                target = {'n':len(y),'positive':int((y==1).sum()),'negative':int((y==0).sum()),
                          'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}
                for field,value in target.items():
                    audit.check('pet_subgroups', close(rec[field],value),f'{tag}/{pattern}: {field}')
                total_pattern_n += target['n']
            audit.check('pet_subgroups',total_pattern_n==len(d['test']),f'{tag}: complete PET partition')
            reference_key = (task,seed,fold)
            if reference_key not in reference_splits:
                reference_splits[reference_key] = {name:d[name].copy() for name in ('train','val','test','node_ids','y')}
            for name,value in reference_splits[reference_key].items():
                audit.equal_array('same_fold_alignment',d[name],value,f'{tag}: same-fold {name}')
            cp = torch.load(out/'checkpoints'/f'{tag}.pt',map_location='cpu',weights_only=False)
            audit.check('checkpoint_metadata',cp['study']=={
                'task':task,'seed':seed,'fold':fold,'arm':arm,
                'cohort_sha256':protocol['cohort_sha256'],'protocol_sha256':protocol_hash,
                'source_sha256':frozen_src},f'{tag}: checkpoint provenance')
            audit.check('checkpoint_validity',cp['input'].device.type=='cpu' and
                        torch.isfinite(cp['input']).all().item() and
                        cp['tokens'].device.type=='cpu' and torch.isfinite(cp['tokens']).all().item(),
                        f'{tag}: finite CPU input/tokens')
            audit.check('checkpoint_validity',all(v.device.type=='cpu' and torch.isfinite(v).all().item()
                                                   for v in cp['body_state'].values()),f'{tag}: finite CPU backbone')
            for name in ('train','val','test'):
                audit.equal_array('checkpoint_alignment',cp['split'][name],d[name],f'{tag}: checkpoint {name}')
            actual_graph_sha = graph_sha(cp['incidence'])
            audit.check('graph_integrity',actual_graph_sha==record['graph_sha256'],f'{tag}: checkpoint graph hash')
            baseline = by_key[(task,seed,fold,'baseline_naive')]
            if arm != 'baseline_observed':
                audit.check('graph_integrity',record['graph_sha256']==baseline['graph_sha256'],
                            f'{tag}: fixed baseline incidence across module arms')
                audit.check('graph_integrity',cp['graph_mode']=='naive',f'{tag}: naive fixed graph mode')
            else:
                audit.check('graph_integrity',cp['graph_mode']=='observed',f'{tag}: isolated observed graph arm')
            check_fitting(audit,cp,td,tag,d['node_ids'])
            if arm.startswith('baseline_'):
                audit.check('checkpoint_generator', 'pet_generator' not in cp,f'{tag}: baseline has no generator')
            else:
                audit.check('checkpoint_generator',cp['pet_generator']['arm']==arm,f'{tag}: matching generator')
                audit.check('checkpoint_generator',all(v.device.type=='cpu' and torch.isfinite(v).all().item()
                                                        for v in cp['pet_generator']['state'].values()),
                            f'{tag}: finite CPU generator')
                selection = read_json(out/'selection'/f'{tag}.json')
                audit.check('generator_selection', selection['fit_patients']==len(cp['split']['mae_eligible']),
                            f'{tag}: training+background generator pool size')
                audit.check('generator_selection',selection['preparation']['pca_fit_patients']==len(d['train']),
                            f'{tag}: neighbor PCA fits task-training only')
                for m in ('amy','tau'):
                    audit.check('generator_selection',selection['fit_observed_pet'][m]==int(td.observed[m][cp['split']['mae_eligible']].sum()),
                                f'{tag}/{m}: generator training observations')
                    audit.check('generator_selection',selection['validation_observed_pet'][m]==int(td.observed[m][d['val']].sum()),
                                f'{tag}/{m}: validation observations')
            if 'checkpoint_replay_max_error' in record:
                error = record['checkpoint_replay_max_error']
                replay_checks.append({'tag':tag,'max_error':error})
                audit.check('recorded_checkpoint_replay',0<=error<=2e-6,f'{tag}: runner device replay')
            del cp
        except Exception as exc:
            audit.check('run_readability',False,f'{tag}: {type(exc).__name__}: {exc}')
        if index%25==0:
            print(f'Audited {index}/{expected} saved runs on CPU.',flush=True)
    coverage = {}
    for task,arm,seed in itertools.product(protocol['tasks'],protocol['arms'],protocol['seeds']):
        rows = [r for r in records if (r['task'],r['arm'],r['seed'])==(task,arm,seed)]
        counts = Counter(v for r in rows for v in r['test_rows'])
        target = set(int(v) for v in task_data[task].node_ids[task_data[task].y>=0])
        once = set(counts)==target and all(v==1 for v in counts.values())
        audit.check('patient_cv_coverage',once,f'{task}/{arm}/seed{seed}: every labeled patient tested once')
        audit.check('patient_cv_coverage',{r['fold'] for r in rows}==set(protocol['folds']),
                    f'{task}/{arm}/seed{seed}: all folds')
        coverage[f'{task}/{arm}/seed{seed}'] = {'unique_patients':len(counts),'each_tested_once':once}
        if len(recomputed)==expected:
            item = summary['tasks'][task][arm]
            seed_item = next(v for v in item['per_seed'] if v['seed']==seed)
            for metric in ('auc','bacc','sen','spe','val_auc','auc_pet','auc_nopet'):
                values = [recomputed[f"{task}_seed{r['seed']}_fold{r['fold']}_{arm}"][metric]
                          for r in records if r['task']==task and r['arm']==arm]
                audit.check('summary_recomputation',close(item['metrics'][metric]['mean'],float(np.nanmean(values))),
                            f'{task}/{arm}/{metric}: mean')
                audit.check('summary_recomputation',close(item['metrics'][metric]['std_population'],float(np.nanstd(values))),
                            f'{task}/{arm}/{metric}: population std')
                seed_values = [recomputed[f"{task}_seed{r['seed']}_fold{r['fold']}_{arm}"][metric]
                               for r in rows]
                audit.check('seed_summary_recomputation',close(seed_item['metrics'][metric]['mean'],float(np.nanmean(seed_values))),
                            f'{task}/{arm}/seed{seed}/{metric}: mean')
                audit.check('seed_summary_recomputation',close(seed_item['metrics'][metric]['std_population'],float(np.nanstd(seed_values))),
                            f'{task}/{arm}/seed{seed}/{metric}: population std')
    for task,seed,fold in itertools.product(protocol['tasks'],protocol['seeds'],protocol['folds']):
        tag=f'{task}_seed{seed}_fold{fold}'
        split=reference_splits.get((task,seed,fold))
        if split is None:
            audit.check('split_audits',False,f'{tag}: no readable reference split'); continue
        frozen=read_json(out/'split_audits'/f'{tag}.json')
        td=task_data[task]
        for key in ('train','val','test'):
            audit.equal_array('split_audits',frozen[f'{key}_rows'],td.node_ids[split[key]],f'{tag}: {key}')
        pool=np.union1d(split['train'],np.flatnonzero(td.y<0))
        audit.equal_array('split_audits',frozen['fit_and_donor_rows'],td.node_ids[pool],f'{tag}: donor pool')
        audit.check('split_audits',frozen['graph_sha256']==by_key[(task,seed,fold,'baseline_naive')]['graph_sha256'],
                    f'{tag}: recorded fixed graph')
        for key in ('donor_validation_overlap','donor_test_overlap','mae_eligible_validation_overlap','mae_eligible_test_overlap'):
            audit.check('split_audits',frozen[key]==0,f'{tag}: {key}')
    expected_replays=len(protocol['tasks'])*len(protocol['arms'])
    audit.check('recorded_checkpoint_replay',len(replay_checks)==expected_replays,
                'First split replay recorded for each predefined arm/task')
    zero=read_json(out/'zero_output_check.json')
    audit.check('zero_output_identity',zero['passed'] and zero['max_probability_error']==0,
                'Recorded full-pipeline zero-output sham equals new baseline')
    report={'all_passed':not audit.failures,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'execution':'CPU-only independent audit; no training or GPU inference',
            'completed_runs':len(records),'expected_runs':expected,'recomputed_runs':len(recomputed),
            'checkpoint_count':len(list((out/'checkpoints').glob('*.pt'))),
            'preserved_original_files_checked':len(origin['preserved_files']),
            'checks_by_group':dict(audit.checked),'failures':audit.failures,
            'coverage':coverage,'validation_thresholds':thresholds,
            'recorded_first_split_checkpoint_replay':replay_checks,
            'checkpoint_replay_scope':'Runner recorded same-device replay; audit does not assert CPU/GPU bitwise prediction equivalence',
            'interpretation':'Exploratory same-cohort comparisons; no new untouched external test claim'}
    target=out/'verification_independent.json'
    temporary=target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(target)
    print(json.dumps({'all_passed':report['all_passed'],'runs':len(records),
                      'checks':sum(audit.checked.values()),'failures':len(audit.failures),
                      'report':str(target)},ensure_ascii=False),flush=True)
    if audit.failures:
        raise SystemExit(1)


if __name__=='__main__':
    main()

