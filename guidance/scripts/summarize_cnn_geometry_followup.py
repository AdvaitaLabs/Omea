"""All-attempt C/D accounting; selection consumes only CNN and validity counts."""
import csv
import json
from collections import Counter
from cnn_geometry_followup_settings import ARMS, SETTINGS, BASELINE, SOURCE
from score_cnn_clean import assert_clean


def good(row):
    a, s = row['raw_audit'], row.get('raw_score', {})
    return bool(a.get('strict_success') and a.get('new_design') and s.get('status') == 'scored'
                and s.get('posebusters_complete') and s.get('posebusters_pass'))


def choose(clean):
    assert_clean(clean)
    base = clean[BASELINE['name']]
    eligible = [arm for arm in SETTINGS if clean[arm]['valid_new'] >= base['valid_new']
                and clean[arm]['cnn_success'] > base['cnn_success']]
    eligible.sort(key=lambda arm: (-clean[arm]['cnn_success'], arm))
    return dict(selected_arm=eligible[0] if eligible else BASELINE['name'],
        rule='Valid-new >= B; strictly more valid-new CNN improvements; maximize count, then lexicographic arm.',
        inputs=clean, eligible=eligible, uses_external_score=False)


def write_accounting(run, records, tasks, references):
    assert len(records) == len({(r['arm'],r['case_id'],r['attempt']) for r in records}) == 12000
    assert Counter(r['arm'] for r in records) == {arm:3000 for arm in ARMS}
    def flags(row):
        if not good(row): return False, False, False
        s, ref = row['raw_score'], references[row['case_id']]
        c = s['cnn_affinity_pK'] > ref['cnn_affinity_pK']
        v = s['vina_kcal_mol'] < ref['vina_kcal_mol']
        return c, v, c and v
    def scope(cases):
        cases = set(cases)
        expected = {(c,i) for c in cases for i in range(100)}
        by = {a:{(r['case_id'],r['attempt']):r for r in records if r['arm']==a and r['case_id'] in cases} for a in ARMS}
        assert all(set(m)==expected for m in by.values())
        arms = {}
        for arm, mapping in by.items():
            rows = list(mapping.values()); n=len(rows)
            c,v,d = map(sum,zip(*(flags(r) for r in rows)))
            m=dict(attempts=n,reconstructed=sum(r['reconstructed'] for r in rows),
                strict=sum(r['raw_audit']['strict_success'] for r in rows),
                scored=sum(r.get('raw_score',{}).get('status')=='scored' for r in rows),
                score_failures=sum(r.get('raw_score',{}).get('status')=='failed' for r in rows),
                valid_new=sum(good(r) for r in rows),cnn_success=c,vina_success=v,dual_success=d,
                missing_geometry=sum(r.get('raw_score',{}).get('status')=='scored' and not r['raw_score'].get('posebusters_complete') for r in rows))
            for metric in ('valid_new','cnn_success','vina_success','dual_success'):m[metric+'_rate']=m[metric]/n
            arms[arm]=m
        paired={};base=by[BASELINE['name']]
        for arm in SETTINGS:
            gains=[k for k in sorted(expected) if flags(by[arm][k])[2] and not flags(base[k])[2]]
            losses=[k for k in sorted(expected) if flags(base[k])[2] and not flags(by[arm][k])[2]]
            paired[arm]=dict(dual_gained=len(gains),dual_lost=len(losses),dual_net=len(gains)-len(losses),
                delta_percentage_points=100*(len(gains)-len(losses))/len(expected),gained_attempts=gains,lost_attempts=losses)
            assert len(gains)-len(losses)==arms[arm]['dual_success']-arms[BASELINE['name']]['dual_success']
        return dict(cases=len(cases),attempts_per_arm=len(expected),arms=arms,paired_vs_B=paired)
    ids=[t['case_id'] for t in tasks]
    previous=set(json.loads((run/'manifest.json').read_text())['previous_selection_case_ids'])
    scopes={name:scope(cases) for name,cases in [('all_30',ids),
        ('previous_selection_10',[c for c in ids if c in previous]),
        ('previous_additional_20',[c for c in ids if c not in previous])]}
    per_case={case:scope([case]) for case in ids}
    per_target={g:scope([t['case_id'] for t in tasks if t['target_group']==g]) for g in sorted({t['target_group'] for t in tasks})}
    for arm in ARMS:
        for metric in ('attempts','valid_new','cnn_success','vina_success','dual_success'):
            n=scopes['all_30']['arms'][arm][metric]
            assert n==sum(s['arms'][arm][metric] for s in per_case.values())
            assert n==sum(s['arms'][arm][metric] for s in per_target.values())
            assert n==sum(scopes[s]['arms'][arm][metric] for s in ('previous_selection_10','previous_additional_20'))
    old=json.loads((SOURCE/'accounting.json').read_text())['scopes']['all_30']['arms']
    for arm in ARMS[:2]:
        for metric in ('attempts','valid_new','cnn_success','vina_success','dual_success'):
            assert scopes['all_30']['arms'][arm][metric]==old[arm][metric]
    clean={arm:{k:m[k] for k in ('attempts','valid_new','cnn_success')} for arm,m in scopes['all_30']['arms'].items()}
    selection=choose(clean)
    result=dict(complete=True,total_attempts=12000,new_attempts=6000,reused_attempts=6000,
        scopes=scopes,per_case=per_case,per_target=per_target,selection=selection,
        metric_definition='Strict new design and complete 22-check geometry pass, Vina lower AND CNN affinity higher than case reference, divided by all attempts.',
        interpretation=['Vina evaluated for both arms; never used by guidance or selector.',
            'Original fixed-pose judge, native final reconstruction, all failures retained.',
            'Previously examined development cases; not independent holdout validation.'])
    for name,value in [('accounting',result),('selection',selection)]:
        (run/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    csv_rows=[dict(level=level,name=name,arm=arm,**m) for level,mapping in
        [('scope',scopes),('case',per_case),('target',per_target)] for name,s in mapping.items() for arm,m in s['arms'].items()]
    with (run/'accounting.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    lines=['C/D 全量几何引导实验：CNN、Vina 与双改善率',
        '每组30例×100次；所有失败保留在分母。Vina仅用于生成后的固定构象评估。',
        'B为配对对照：492/3000=16.40%。C仅加强几何权重，D限制损害局部几何的坐标更新。','']
    for name,s in scopes.items():
        lines += [name,'设置 | 合格新分子 | CNN改善 | Vina改善 | 双改善 | 分母']
        for arm,m in s['arms'].items():
            lines.append(f"{arm} | {m['valid_new']} | {m['cnn_success']} | {m['vina_success']} | {m['dual_success']} ({m['dual_success_rate']:.2%}) | {m['attempts']}")
        lines.append('')
    lines.append('仅按CNN/有效性预定规则选中：'+selection['selected_arm'])
    (run/'accounting.txt').write_text('\n'.join(lines)+'\n')
    return result
