"""Post-review operational audit of immutable evaluation-v2 decision archives.

This re-analysis uses no detector refit or target-label policy selection. It
verifies the original SHA-256 manifest, distinguishes empty-set miscoverage
from automatic mistakes, and retains all 24 projects for every method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, binomtest
from vultriage.empirical_bootstrap import percentile_interval


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def action_counts(labels: np.ndarray, sets: np.ndarray) -> dict:
    labels = np.asarray(labels)
    sets = np.asarray(sets)
    if (labels.ndim != 1 or len(labels) == 0 or
            sets.shape != (len(labels), 2) or
            not np.isin(labels, [0, 1]).all() or
            not np.isin(sets, [0, 1]).all()):
        raise ValueError("Invalid decision archive")
    labels = labels.astype(int)
    sets = sets.astype(bool)
    single = sets.sum(1) == 1
    safe = single & sets[:, 0]
    vuln = single & sets[:, 1]
    review = ~single
    fn = int(np.sum(safe & (labels == 1)))
    fp = int(np.sum(vuln & (labels == 0)))
    nv = int(np.sum(labels == 1))
    ns = len(labels) - nv
    n_auto = int(single.sum())
    n_safe_auto = int(safe.sum())
    n_vuln_auto = int(vuln.sum())
    cov_v = int(np.sum(single & (labels == 1)))
    cov_s = int(np.sum(single & (labels == 0)))
    ratio = lambda a, b: a / b if b else np.nan
    counts = dict(n=len(labels), n_vulnerable=nv, n_safe=ns,
                  auto_safe=n_safe_auto, auto_vulnerable=n_vuln_auto,
                  auto_fn=fn, auto_fp=fp, review_count=int(review.sum()),
                  review_vulnerable=int(np.sum(review & (labels == 1))),
                  review_safe=int(np.sum(review & (labels == 0))),
                  empty_count=int(np.sum(sets.sum(1) == 0)),
                  singleton_count=n_auto,
                  coverage=n_auto / len(labels), review_load=review.mean(),
                  auto_fn_rate=ratio(fn, nv), auto_fp_rate=ratio(fp, ns),
                  selective_error=ratio(fn + fp, n_auto),
                  selective_vulnerable_error=ratio(fn, cov_v),
                  selective_safe_error=ratio(fp, cov_s),
                  safe_release_contamination=ratio(fn, n_safe_auto),
                  vulnerable_alert_precision=ratio(n_vuln_auto - fp, n_vuln_auto))
    assert counts['auto_safe'] + counts['auto_vulnerable'] + counts['review_count'] == len(labels)
    return counts


def matched_confidence(probability: np.ndarray, count: int) -> np.ndarray:
    probability = np.asarray(probability, float)
    if (probability.ndim != 1 or not np.isfinite(probability).all() or
            ((probability < 0) | (probability > 1)).any() or
            isinstance(count, (bool, np.bool_)) or
            not isinstance(count, (int, np.integer)) or not 0 <= count <= len(probability)):
        raise ValueError('Invalid probability or automation count')
    confidence = np.maximum(probability, 1 - probability)
    order = np.argsort(-confidence, kind="stable")
    sets = np.ones((len(probability), 2), bool)
    retained = order[:count]
    sets[retained] = False
    # Match NumPy argmax on [1-p, p]: an exact p=.5 tie chooses class 0.
    sets[retained, (probability[retained] > 0.5).astype(int)] = True
    return sets


def auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    n1, n0 = int(y.sum()), len(y) - int(y.sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((rankdata(score)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_summary(frame: pd.DataFrame, outcome: str, baseline: str, candidate: str, rng) -> dict:
    a = frame[frame.method == baseline].set_index('target_group')[outcome]
    b = frame[frame.method == candidate].set_index('target_group')[outcome]
    if set(a.index) != set(b.index):
        raise ValueError('Unmatched project cohorts')
    delta = (b.reindex(a.index) - a).to_numpy(float)
    if not np.isfinite(delta).all():
        raise ValueError('Undefined paired endpoint')
    interval = percentile_interval(delta)
    wins, losses = int((delta < -1e-12).sum()), int((delta > 1e-12).sum())
    return dict(outcome=outcome, baseline=baseline, candidate=candidate,
                projects=len(delta), median_difference=float(np.median(delta)),
                ci_lower=interval[0], ci_upper=interval[1],
                wins=wins, losses=losses, ties=len(delta)-wins-losses,
                descriptive_sign_p=float(binomtest(wins, wins+losses).pvalue) if wins+losses else np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--public-root', type=Path, default=Path('public_results/extension-v2'))
    p.add_argument('--output', type=Path, default=Path('outputs/neurocomputing-revision-v1'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.evaluation / 'evaluation_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    raw_metrics_path = args.public_root / 'evaluation-v2/fold_seed_metrics.csv'
    if sha256(raw_metrics_path) != manifest['metrics_sha256']:
        raise ValueError('Original metric digest mismatch')
    metrics = pd.read_csv(raw_metrics_path)
    primary = metrics[(metrics.alpha_vulnerable == .1) & (metrics.alpha_safe == .2)]
    contract_path = args.public_root / 'contract-v1/contract_manifest.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    qualification_path = args.public_root / 'contract-v1/qualification_summary.csv'
    split_path = args.public_root / 'contract-v1/fixed_split_projects.csv'
    for input_path in [qualification_path, split_path]:
        if sha256(input_path) != contract['outputs'][input_path.name]['sha256']:
            raise ValueError(f'Contract input digest mismatch: {input_path.name}')
    gate_seal_path = args.public_root / 'gate-v1/gate_seal.json'
    gate_seal = json.loads(gate_seal_path.read_text(encoding='utf-8'))
    development_path = args.public_root / 'gate-v1/primevul_gate_development.csv'
    if sha256(development_path) != gate_seal['development_csv_sha256']:
        raise ValueError('Gate development digest mismatch')
    qualification = pd.read_csv(qualification_path)
    registered = qualification[qualification.partition == 'registration'].set_index('detector').qualified.to_dict()
    split = pd.read_csv(split_path).set_index('target_group').partition.to_dict()
    rows, examples, checked = [], [], {}
    key = 'av0p1_as0p2__'
    for relative, expected in sorted(manifest['decision_files'].items()):
        path = args.evaluation / relative
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f'Sealed decision digest mismatch: {relative}')
        checked[relative] = observed
        detector, group = path.parts[-3:-1]
        seed = int(path.stem.split('-')[-1])
        with np.load(path, allow_pickle=False) as a:
            y = a['target_labels'].astype(int)
            probability = a['target_p_vulnerable'].astype(float)
            decisions = {name: a[key+name].astype(bool) for name in
                         ['unweighted_mondrian','msp_matched_unweighted_mondrian','estimated_weight_no_gate','vultriage_full_gate']}
            decisions['msp_matched_weighted'] = matched_confidence(probability, int((decisions['estimated_weight_no_gate'].sum(1)==1).sum()))
            decisions['msp_matched_gate'] = matched_confidence(probability, int((decisions['vultriage_full_gate'].sum(1)==1).sum()))
            forced = np.zeros((len(y),2),bool)
            forced[np.arange(len(y)), (probability>.5).astype(int)] = True
            decisions['forced_argmax'] = forced
            decisions['all_review'] = np.ones((len(y),2),bool)
            decisions['registered_controller'] = decisions['vultriage_full_gate'] if registered[detector] else decisions['all_review']
            for name, sets in decisions.items():
                counts = action_counts(y, sets)
                rows.append(dict(detector=detector,target_group=group,seed=seed,partition=split[group],method=name,**counts))
                if name in ['unweighted_mondrian','msp_matched_unweighted_mondrian','estimated_weight_no_gate','vultriage_full_gate']:
                    orig_name = {'estimated_weight_no_gate':'estimated_weight_no_gate_clip_20','vultriage_full_gate':'vultriage_full_gate_clip_20'}.get(name,name)
                    old = primary[(primary.detector==detector)&(primary.target_group==group)&(primary.seed==seed)&(primary.method==orig_name)].iloc[0]
                    assert counts['singleton_count'] == old.singleton_count
                    # Automatic false-safe/false-alert errors omit empty-set errors.
                    assert counts['auto_fn'] <= old.vulnerable_miscoverage_count
                    assert counts['auto_fp'] <= old.safe_miscoverage_count
                if detector=='hashing' and seed==13 and split[group]=='confirmation' and name=='registered_controller':
                    for action, mask in [('auto-safe',(sets.sum(1)==1)&sets[:,0]),('auto-vulnerable',(sets.sum(1)==1)&sets[:,1]),('review',sets.sum(1)!=1)]:
                        if not any(e['action']==action for e in examples) and mask.any():
                            i=int(np.flatnonzero(mask)[0])
                            examples.append(dict(action=action,target_group=group,seed=seed,target_position=int(a['target_positions'][i]),
                                                 vulnerable_probability=float(probability[i]),prediction_set=np.flatnonzero(sets[i]).tolist(),label=int(y[i])))
    frame = pd.DataFrame(rows)
    if len(checked) != 240 or frame.groupby(['detector','method']).target_group.nunique().ne(24).any():
        raise ValueError('Incomplete original cohort')
    frame.to_csv(args.output/'action_metrics_seed.csv',index=False)
    numerical = [c for c in frame.columns if c not in ['detector','target_group','seed','partition','method']]
    projects = frame.groupby(['detector','target_group','partition','method'])[numerical].mean().reset_index()
    projects.to_csv(args.output/'action_metrics_project.csv',index=False)
    projects.groupby(['detector','method'])[numerical].median().reset_index().to_csv(args.output/'action_summary.csv',index=False)
    rng=np.random.default_rng(20261009)
    paired=[]
    comparisons=[('unweighted_mondrian','estimated_weight_no_gate'),('msp_matched_weighted','estimated_weight_no_gate'),
                 ('msp_matched_gate','vultriage_full_gate'),('forced_argmax','vultriage_full_gate')]
    for detector, df in projects.groupby('detector'):
        for baseline,candidate in comparisons:
            for outcome in ['coverage','auto_fn_rate','auto_fp_rate']:
                paired.append(dict(detector=detector,**paired_summary(df,outcome,baseline,candidate,rng)))
    pd.DataFrame(paired).to_csv(args.output/'action_paired_comparisons.csv',index=False)
    # Cost grid is specified for this post-review audit, not the original experiment.
    costs=[]
    for miss in [1.,5.,10.,20.,50.]:
        for review in [.01,.05,.1,.2,.5]:
            for residual in [0.,.01,.05]:
                c=projects.copy()
                c['cost']=(miss*c.auto_fn+c.auto_fp+review*c.review_count+residual*(miss*c.review_vulnerable+c.review_safe))/c.n
                for detector,df in c.groupby('detector'):
                    for baseline,candidate in comparisons:
                        result=paired_summary(df,'cost',baseline,candidate,rng)
                        costs.append(dict(detector=detector,miss_cost=miss,false_alert_cost=1.,review_cost=review,review_residual_error=residual,**result))
    costs=pd.DataFrame(costs)
    costs.to_csv(args.output/'cost_paired_sensitivity.csv',index=False)
    nominal=projects.copy()
    nominal['cost']=(10*nominal.auto_fn+nominal.auto_fp+.1*nominal.review_count)/nominal.n
    nominal.groupby(['detector','method']).cost.median().reset_index().to_csv(args.output/'nominal_cost_summary.csv',index=False)
    gate_rows=primary[primary.method=='vultriage_full_gate_clip_20'].copy()
    gate_rows['score']=gate_rows.support.map(lambda t:float(json.loads(t)['gate_probability']))
    gate_rows['passed']=gate_rows.support.map(lambda t:bool(json.loads(t)['passed']))
    raw=primary[primary.method=='estimated_weight_no_gate_clip_20'].groupby(['detector','target_group']).max_relative_violation.mean()
    g=gate_rows.groupby(['detector','target_group']).agg(score=('score','mean'),pass_all=('passed','all')).reset_index()
    g['severe']=[int(raw.loc[(r.detector,r.target_group)]>.5) for r in g.itertuples()]
    g.to_csv(args.output/'gate_project_scores.csv',index=False)
    gate_audit=[]
    for detector,df in g.groupby('detector'):
        yy,ss=df.severe.to_numpy(),df.score.to_numpy()
        vals=[]
        for _ in range(10000):
            ix=rng.integers(0,len(df),len(df)); val=auc(yy[ix],ss[ix])
            if np.isfinite(val): vals.append(val)
        gate_audit.append(dict(domain='DiverseVul',detector=detector,projects=len(df),rows=len(df),auroc=auc(yy,ss),
                               auc_ci_lower=float(np.quantile(vals,.025)),auc_ci_upper=float(np.quantile(vals,.975)),
                               valid_bootstrap_samples=len(vals),brier=float(np.mean((ss-yy)**2)),
                               prevalence=float(yy.mean()),prevalence_brier=float(np.mean((yy-yy.mean())**2))))
    dev=pd.read_csv(development_path)
    yy,ss=dev.severe_violation.to_numpy(),dev.crossfit_severe_probability.to_numpy()
    gate_audit.append(dict(domain='PrimeVul crossfit',detector='frozen gate',projects=dev.target_group.nunique(),rows=len(dev),
                           auroc=auc(yy,ss),auc_ci_lower=np.nan,auc_ci_upper=np.nan,valid_bootstrap_samples=0,
                           brier=float(np.mean((ss-yy)**2)),prevalence=float(yy.mean()),prevalence_brier=float(np.mean((yy-yy.mean())**2))))
    pd.DataFrame(gate_audit).to_csv(args.output/'gate_audit.csv',index=False)
    (args.output/'running_examples.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
    receipts={'analysis_status':'post-review re-analysis of frozen evaluation-v2; target outcomes already known',
              'metric_rows':len(frame),'project_method_rows':len(projects),'verified_decision_archives':len(checked),
              'script_sha256':sha256(Path(__file__)),'original_manifest_sha256':sha256(manifest_path),
              'original_metrics_sha256':sha256(raw_metrics_path),'verified_archive_sha256':checked,
              'cost_grid':{'miss':[1,5,10,20,50],'false_alert':1,'review':[.01,.05,.1,.2,.5],'review_residual_error':[0,.01,.05]},
              'paired_median_interval':'exact ordinary even-n empirical bootstrap; percentile inverse CDF',
              'gate_bootstrap_replicates':10000,'seed':20261009,'selection':'no target-label policy tuning; all outcomes retained',
              'new_confidence_baselines':'rank immutable float32 probabilities with stable row-order ties; exact p=.5 argmax tie is safe; match singleton count without labels',
              'input_sha256':{
                  'contract-v1/contract_manifest.json':sha256(contract_path),
                  'contract-v1/qualification_summary.csv':sha256(qualification_path),
                  'contract-v1/fixed_split_projects.csv':sha256(split_path),
                  'gate-v1/gate_seal.json':sha256(gate_seal_path),
                  'gate-v1/primevul_gate_development.csv':sha256(development_path),
              },
              'empirical_bootstrap_helper_sha256':sha256(Path(__file__).parent/'vultriage/empirical_bootstrap.py'),
              'output_sha256':{p.name:sha256(p) for p in sorted(args.output.iterdir())
                               if p.is_file() and p.name != 'analysis_manifest.json'}}
    (args.output/'analysis_manifest.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    print(json.dumps({'archives_verified':len(checked),'rows':len(frame),'project_rows':len(projects),'cost_comparisons':len(costs),'output':str(args.output)},indent=2))


if __name__=='__main__':
    main()
