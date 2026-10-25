"""All-output paired evaluation of direct gradients on continuous element inputs."""
import concurrent.futures as cf
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from rdkit import Chem
from tasks import audit
from score_extra_results import score_one
from local_regeneration_graph import changes


def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def score_arm(run, arm, records, tasks, required_checks, update):
    taskmap={t['case_id']:t for t in tasks};selected=[r for r in records if r['arm']==arm]
    jobs=[r for r in selected if r['raw_audit']['strict_success']]
    assert len(selected)==json.loads((run/'manifest.json').read_text())['outputs_per_arm']
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(score_one,Chem.Mol((Path(r['path'])/'raw.rdmol').read_bytes()),
                             taskmap[r['case_id']]['protein_path'],Path(r['path'])/'judge'):r for r in jobs}
        for done,future in enumerate(cf.as_completed(futures),1):
            r=futures[future];s=dict(future.result())
            missing=sorted(required_checks-set(s.get('posebusters_checks',{})))
            s.update(legacy_posebusters_pass=s.get('posebusters_pass',False),posebusters_complete=not missing,missing_checks=missing)
            s['posebusters_pass']=s['legacy_posebusters_pass'] and not missing and s['status']=='scored'
            r['raw_score']=s;save(Path(r['path'])/'judge'/'verified_score.json',s)
            if done%10==0 or done==len(jobs):
                save(run/'records.json',records);update('scoring',active_arm=arm,completed_scores=done,total_scores=len(jobs))
    save(run/'records.json',records)


def summarize(run, configurations, records, tasks, references):
    taskmap={t['case_id']:t for t in tasks}
    baseline={(r['case_id'],r['attempt']):r for r in records if r['arm']=='zero'}
    manifest=json.loads((run/'manifest.json').read_text())
    expected=manifest['outputs_per_arm']
    attempts_per_case=manifest['attempts_per_case']
    assert len(baseline)==expected
    def good(r):
        return bool(r['raw_audit'].get('strict_success') and r['raw_audit'].get('new_design') and
                    r.get('raw_score',{}).get('status')=='scored' and r['raw_score'].get('posebusters_pass'))
    def dual(r):
        s=r.get('raw_score',{});ref=references[r['case_id']]
        return good(r) and s['vina_kcal_mol']<ref['vina_kcal_mol'] and s['cnn_affinity_pK']>ref['cnn_affinity_pK']
    rows=[];arms={}
    for cfg in configurations:
        arm=cfg['name'];rr=[r for r in records if r['arm']==arm]
        if len(rr)!=expected or any(r['raw_audit']['strict_success'] and 'raw_score' not in r for r in rr):continue
        pairs=[]
        for r in sorted(rr,key=lambda r:(r['case_id'],r['attempt'])):
            base=baseline[r['case_id'],r['attempt']];nc=taskmap[r['case_id']]['fixed_heavy_atoms']
            if r['reconstructed'] and base['reconstructed']:
                change=changes(Chem.Mol((Path(base['path'])/'raw.rdmol').read_bytes()),
                               Chem.Mol((Path(r['path'])/'raw.rdmol').read_bytes()),nc)
            else:change=dict(chemical_graph_changed=False,bond_graph_changed=False,connectivity_changed=False,
                             change_category='unavailable',parent_smiles=None,candidate_smiles=None)
            s,b=r.get('raw_score',{}),base.get('raw_score',{})
            scored=s.get('status')==b.get('status')=='scored'
            cnn=s['cnn_affinity_pK']-b['cnn_affinity_pK'] if scored else None
            vina=s['vina_kcal_mol']-b['vina_kcal_mol'] if scored else None
            wins=good(r) and good(base) and cnn is not None and cnn>=.05 and vina<=-.01
            pair=dict(arm=arm,case_id=r['case_id'],attempt=r['attempt'],seed=r['seed'],
                      valid_new=good(r),baseline_valid_new=good(base),dual_vs_reference=dual(r),baseline_dual=dual(base),
                      pair_scored=scored,cnn_gain=cnn,vina_change=vina,valid_pair_dual_win=wins,
                      valid_chemical_pair_dual_win=wins and change['chemical_graph_changed'],
                      charge_match=r['raw_audit'].get('charge_match',False),path=r['path'],baseline_path=base['path'],**change)
            pairs.append(pair);rows.append(pair)
        eventpath=run/'arms'/arm/'events.jsonl'
        events=[json.loads(s) for s in eventpath.read_text().splitlines()] if eventpath.exists() else []
        guided=[r for r in events if r['status']=='guided']
        chem_pairs=[r for r in pairs if r['chemical_graph_changed']]
        scored=[r for r in pairs if r['pair_scored']]
        metrics=dict(attempts=expected,reconstructed=sum(r['reconstructed'] for r in rr),
                     strict=sum(r['raw_audit']['strict_success'] for r in rr),valid_new=sum(r['valid_new'] for r in pairs),
                     valid_gained=sum(r['valid_new'] and not r['baseline_valid_new'] for r in pairs),
                     valid_lost=sum(not r['valid_new'] and r['baseline_valid_new'] for r in pairs),
                     dual_vs_reference=sum(r['dual_vs_reference'] for r in pairs),
                     dual_charge_matched=sum(r['dual_vs_reference'] and r['charge_match'] for r in pairs),
                     dual_gained=sum(r['dual_vs_reference'] and not r['baseline_dual'] for r in pairs),
                     dual_lost=sum(not r['dual_vs_reference'] and r['baseline_dual'] for r in pairs),
                     chemical_graph_changed=len(chem_pairs),valid_chemical_graph_changed=sum(r['valid_new'] for r in chem_pairs),
                     bond_graph_changed=sum(r['bond_graph_changed'] for r in pairs),
                     valid_pair_dual_wins=sum(r['valid_pair_dual_win'] for r in pairs),
                     valid_chemical_pair_dual_wins=sum(r['valid_chemical_pair_dual_win'] for r in pairs),
                     scored_pairs=len(scored),mean_cnn_gain_scored_pairs=float(np.mean([r['cnn_gain'] for r in scored])) if scored else None,
                     mean_vina_change_scored_pairs=float(np.mean([r['vina_change'] for r in scored])) if scored else None,
                     raw_failure_reasons=dict(Counter(s for r in rr for s in r['raw_audit']['reasons'])),
                     missing_geometry_checks=sum(r.get('raw_score',{}).get('status')=='scored' and not r['raw_score']['posebusters_complete'] for r in rr),
                     score_failures=sum(r.get('raw_score',{}).get('status')=='failed' for r in rr))
        mechanism=dict(event_count=len(events),guided_events=len(guided),chemistry_skips=len(events)-len(guided),
                       coordinate_score_calls=sum(r['coordinate_score_calls'] for r in events),
                       joint_model_forward_calls=sum(r['joint_model_forward_calls'] for r in events),
                       joint_backward_calls=sum(r['joint_backward_calls'] for r in events),
                       scored_element_substitution_candidates=sum(r['scored_element_substitution_candidates'] for r in events),
                       coordinate_x0_max_A=max((r['actual_x0_coordinate_max_change_A'] for r in guided),default=0.),
                       coordinate_reverse_mean_max_A=max((r['actual_reverse_mean_coordinate_max_change_A'] for r in guided),default=0.),
                       feature_x0_max=max((r['actual_x0_feature_max_change'] for r in guided),default=0.),
                       feature_reverse_mean_max=max((r['actual_reverse_mean_feature_max_change'] for r in guided),default=0.),
                       final_discrete_outputs_per_seed=1,terminal_edits=0,trajectory_branches=0)
        arms[arm]=dict(configuration=cfg,metrics=metrics,mechanism=mechanism,paired_rows=pairs,
                       per_case={t['case_id']:dict(attempts=attempts_per_case,valid_new=sum(r['valid_new'] for r in pairs if r['case_id']==t['case_id']),
                                                  dual_vs_reference=sum(r['dual_vs_reference'] for r in pairs if r['case_id']==t['case_id']),
                                                  valid_chemical_pair_dual_wins=sum(r['valid_chemical_pair_dual_win'] for r in pairs if r['case_id']==t['case_id'])) for t in tasks})
    result=dict(complete=len(arms)==len(configurations),arms=arms,
                interpretation=['single terminal output per seed and strength, all failures count; no output selection or resampling',
                                'strict + complete PoseBusters + new design defines valid_new; SMILES alone cannot show pose quality',
                                'raw final generated pose; H-only common preparation; no post-generation GNINA optimization',
                                'element gradients come directly from one continuous-density GNINA score through the ten-element softmax; local chemical-role context held fixed',
                                'scored-pair mean deltas are descriptive and omit unscorable failures; unconditional success counts include every predeclared attempt',
                                f"{len(tasks)} cases and {manifest['target_groups']} target groups; report development replication and expanded-case results separately; no training-independent target holdout claim",
                                'GNINA is optimized, Vina is a second proxy; neither is measured DDG'])
    save(run/'comparison.json',result)
    if rows:
        with (run/'paired_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['连续元素直接梯度实验',f"{len(tasks)} 个案例，{manifest['target_groups']} 个靶标组，每案例 {attempts_per_case} 次。",
           f'每组 {expected} 次生成；相同种子及全部去噪随机量；所有结果直接评价，不择优、不补抽。',
           '所有候选采用相同的加氢处理；主结果没有生成后的 GNINA 构象微调。','',
           f'设置 | 有效新分子/{expected} | 双指标优于参考/{expected} | 改变化学图/{expected} | 合格且改图并双指标优于配对基线/{expected}']
    for arm,v in arms.items():
        m=v['metrics'];lines.append(f"{arm} | {m['valid_new']} | {m['dual_vs_reference']} | {m['chemical_graph_changed']} | {m['valid_chemical_pair_dual_wins']}")
    lines+=['','强度在固定梯度裁剪之后施加，坐标和类型的强度单位不同，不能直接比较数值。',
            '评分是代理指标；生成失败和几何检查缺失均保留在总分母中。']
    (run/'summary.txt').write_text('\n'.join(lines)+'\n')
    return result

