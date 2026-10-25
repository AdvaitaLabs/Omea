"""Fixed-parameter accounting: all attempts, paired scores, cases and targets."""
import csv
import json
from pathlib import Path
import numpy as np

ARMS = ('zero', 'joint_0p4', 'x0p02_h0p6')
BEST = ARMS[-1]
METRICS = ('vina_kcal_mol', 'cnn_affinity_pK')


def good(row):
    score = row.get('raw_score', {})
    return bool(row['raw_audit'].get('strict_success') and row['raw_audit'].get('new_design')
                and score.get('status') == 'scored' and score.get('posebusters_pass'))


def scored(row):
    return row.get('raw_score', {}).get('status') == 'scored'


def stats(values):
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None)


def scope_accounting(case_ids, records, references, attempts=100):
    wanted = set(case_ids)
    by = {arm: {(r['case_id'],r['attempt']):r for r in records
                if r['arm']==arm and r['case_id'] in wanted} for arm in ARMS}
    expected = {(case,i) for case in wanted for i in range(attempts)}
    assert all(set(rows)==expected for rows in by.values()), 'missing/duplicate accounting attempts'
    arms = {}
    for arm, rows in by.items():
        values = list(rows.values())
        valid = [r for r in values if good(r)]
        dual = [r for r in valid if r['raw_score']['vina_kcal_mol'] < references[r['case_id']]['vina_kcal_mol']
                and r['raw_score']['cnn_affinity_pK'] > references[r['case_id']]['cnn_affinity_pK']]
        arms[arm] = dict(attempts=len(values),reconstructed=sum(r['reconstructed'] for r in values),
                         strict=sum(r['raw_audit']['strict_success'] for r in values),
                         scored=sum(scored(r) for r in values),valid_new=len(valid),
                         valid_new_rate=len(valid)/len(values),dual_vs_reference=len(dual),
                         dual_vs_reference_rate=len(dual)/len(values),
                         unscored=sum(not scored(r) for r in values),
                         score_failures=sum(r.get('raw_score',{}).get('status')=='failed' for r in values),
                         incomplete_geometry=sum(scored(r) and not r['raw_score'].get('posebusters_complete',False) for r in values),
                         absolute_scores_valid_new={m:stats([r['raw_score'][m] for r in valid]) for m in METRICS})
    paired = {}
    for control in ARMS[:-1]:
        pools = dict(all_scored_pairs=[k for k in sorted(expected) if scored(by[BEST][k]) and scored(by[control][k])],
                     both_valid_new=[k for k in sorted(expected) if good(by[BEST][k]) and good(by[control][k])],
                     common_valid_triplets=[k for k in sorted(expected) if all(good(by[a][k]) for a in ARMS)])
        paired[control] = {label:dict(n=len(keys),score_deltas={m:stats([by[BEST][k]['raw_score'][m]-by[control][k]['raw_score'][m]
                                                                       for k in keys]) for m in METRICS})
                           for label,keys in pools.items()}
        paired[control]['unconditional_rate_changes'] = {
            m:arms[BEST][m]-arms[control][m] for m in ('valid_new','valid_new_rate','dual_vs_reference','dual_vs_reference_rate')}
        paired[control]['valid_new_gained'] = sum(good(by[BEST][k]) and not good(by[control][k]) for k in expected)
        paired[control]['valid_new_lost'] = sum(good(by[control][k]) and not good(by[BEST][k]) for k in expected)
    return dict(cases=len(wanted),attempts_per_arm=len(expected),arms=arms,paired_selected_minus_control=paired)


def write_accounting(run, records, tasks, references, comparison):
    manifest=json.loads((run/'manifest.json').read_text())
    assert len({(r['arm'],r['case_id'],r['attempt']) for r in records})==len(records)==9000
    all_ids=[t['case_id'] for t in tasks]
    selection=manifest['selection_case_ids']
    additional=[case for case in all_ids if case not in set(selection)]
    scopes={name:scope_accounting(ids,records,references) for name,ids in
            [('all_30',all_ids),('selection_10',selection),('additional_20',additional)]}
    per_case={case:scope_accounting([case],records,references) for case in all_ids}
    per_target={group:scope_accounting([t['case_id'] for t in tasks if t['target_group']==group],records,references)
                for group in sorted({t['target_group'] for t in tasks})}
    for arm in ARMS:
        for key in ('attempts','reconstructed','strict','valid_new','dual_vs_reference','score_failures'):
            assert scopes['all_30']['arms'][arm][key] == comparison['arms'][arm]['metrics'][key], (arm,key)
        for key in ('attempts','valid_new','dual_vs_reference'):
            assert scopes['all_30']['arms'][arm][key] == sum(v['arms'][arm][key] for v in per_case.values())
            assert scopes['all_30']['arms'][arm][key] == sum(v['arms'][arm][key] for v in per_target.values())
            assert scopes['all_30']['arms'][arm][key] == scopes['selection_10']['arms'][arm][key]+scopes['additional_20']['arms'][arm][key]
    macro={arm:{metric:float(np.mean([g['arms'][arm][metric] for g in per_target.values()]))
                for metric in ('valid_new_rate','dual_vs_reference_rate')} for arm in ARMS}
    result=dict(complete=True,selected_fixed_arm=BEST,selection_frozen_before_additional_cases=True,
                new_outputs=2000,reused_outputs=7000,total_outputs=9000,scopes=scopes,
                per_case=per_case,per_target=per_target,target_macro_average_rates=macro,
                interpretation=['All 100 attempts per case remain denominators, including failures.',
                                'A valid-new output passes the strict task audit, complete 22-check geometry assessment and is new.',
                                'Unscorable outputs have no invented affinity score; paired score pools are reported separately.',
                                'Delta = selected minus control; lower Vina and higher CNN affinity are favorable.',
                                'The selection 10 and additional 20 are separate; the additional cases share existing target groups.',
                                'One seed root on the existing 30-case cohort; not a new-target or multi-seed validation.',
                                'GNINA and Vina are proxy scores, not measured binding free energies.'])
    (run/'accounting.json').write_text(json.dumps(result,indent=2))
    rows=[]
    for level,mapping in [('scope',scopes),('case',per_case),('target',per_target)]:
        for name,data in mapping.items():
            for arm,metrics in data['arms'].items():
                row=dict(level=level,name=name,arm=arm,cases=data['cases'])
                row.update({k:v for k,v in metrics.items() if k!='absolute_scores_valid_new'})
                for metric,values in metrics['absolute_scores_valid_new'].items():
                    for stat,value in values.items():row[f'valid_new_{metric}_{stat}']=value
                rows.append(row)
    with (run/'accounting.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['最优 guidance 参数全30案例核算','固定参数：坐标 0.02，元素 0.6；每例100次；原生重建与固定重原子评分。',
           '合计9000条：新生成2000条，复用7000条。10例参与过挑参，另外20例未参与此次挑参。','']
    for name,label in [('all_30','全部30例'),('selection_10','挑参10例'),('additional_20','新增20例')]:
        data=scopes[name];lines += [label,'参数 | 合格新分子 | 双指标优于参考 | 总尝试数']
        for arm,m in data['arms'].items():lines.append(f"{arm} | {m['valid_new']} | {m['dual_vs_reference']} | {m['attempts']}")
        for control,delta in data['paired_selected_minus_control'].items():
            pair=delta['common_valid_triplets'];v=pair['score_deltas']['vina_kcal_mol']['mean'];c=pair['score_deltas']['cnn_affinity_pK']['mean']
            lines.append(f'相对 {control}，共同合格三元组 n={pair["n"]}：平均ΔVina={v} kcal/mol，平均ΔCNN={c} pK。')
        lines.append('')
    lines += ['按案例和靶标的完整计数、均值/中位数与不同配对口径见 accounting.json / accounting.csv。',
              '低Vina、高CNN表示代理分数改善；单套种子，新增20例仍属于已有靶标组。']
    (run/'accounting.txt').write_text('\n'.join(lines)+'\n')
    return result


def self_test():
    rows=[]
    for arm in ARMS:
        for i in range(2):
            row=dict(arm=arm,case_id='c',attempt=i,reconstructed=True,
                     raw_audit=dict(strict_success=True,new_design=True),
                     raw_score=dict(status='scored',posebusters_pass=True,posebusters_complete=True,
                                    vina_kcal_mol=-6-(arm==BEST),cnn_affinity_pK=6+(arm==BEST)))
            if arm==BEST and i==1:
                row['reconstructed']=False;row['raw_audit']['strict_success']=False;row.pop('raw_score')
            rows.append(row)
    result=scope_accounting(['c'],rows,{'c':dict(vina_kcal_mol=-5,cnn_affinity_pK=5)},attempts=2)
    assert result['arms'][BEST]['valid_new_rate']==.5
    assert result['arms'][BEST]['unscored']==1
    pair=result['paired_selected_minus_control']['zero']['common_valid_triplets']
    assert pair['n']==1 and pair['score_deltas']['vina_kcal_mol']['mean']==-1
    assert pair['score_deltas']['cnn_affinity_pK']['mean']==1
    print('Accounting checks passed: failed attempts retained and paired scores use shared valid keys.')


if __name__=='__main__':
    self_test()
