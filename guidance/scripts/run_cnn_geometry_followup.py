"""C/D full runs: frozen paired generation and automatic full dual-metric reporting."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('WANDB_MODE','disabled')
import argparse
from collections import Counter
import concurrent.futures as cf
import copy
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem,RDLogger
from tasks import ROOT,audit
from resume_selected_guidance_full_case_shards import (read,save,sha,now,hardware,integrity,batch,
    TRAJECTORIES,SEED_ROOT,BATCH,validate_core_metadata)
import run_cnn_clean_ablations as transport
from autofragdiff_cnn_geometry_hook import GeometryEngine
from cnn_geometry_followup_settings import SOURCE,PARENT,CHECKS,RUN,BASELINE,SETTINGS,ARMS,NEW_SOURCES
from cnn_clean_settings import element_strength
from score_extra_results import score_one
from score_cnn_clean import assert_clean
from summarize_cnn_geometry_followup import write_accounting
SCRIPT=Path(__file__).resolve()
PYTHON=ROOT/'envs/generators/bin/python'


def key(row):return row['arm'],row['case_id'],row['attempt']


def prepare(run):
    assert not run.exists(),run
    parent=read(PARENT/'execution_manifest.finalize_v1.json');integrity(parent)
    assert read(PARENT/'audit.json')['passed'] and read(SOURCE/'audit.json')['passed']
    source_audit=read(SOURCE/'audit.json')
    for name,digest in source_audit['output_sha256'].items():assert sha(SOURCE/name)==digest
    gates=[read(CHECKS/'cpu_checks.json')]+[read(CHECKS/f'gpu{i}/checks.json') for i in (0,1)]
    for gate in gates:
        assert gate['passed']
        for name,digest in gate['source_sha256'].items():assert sha(ROOT/'scripts'/name)==digest
        for path,digest in gate.get('output_sha256',{}).items():assert sha(path)==digest
    for gpu,gate in enumerate(gates[1:]):
        assert gate['configuration']==list(SETTINGS.values())[gpu] and gate['hardware']==hardware()
        assert gate['baseline_equation_steps']==3500 and gate['technical_outputs']==20
    tasks=read(PARENT/'tasks.json');assert len(tasks)==30
    references=read(SOURCE/'references.json')
    reused=[]
    for row in read(SOURCE/'records.json'):
        if row['arm'] not in ARMS[:2]:continue
        r=copy.deepcopy(row);r.update(reused=True,reused_control_source=str(SOURCE));reused.append(r)
    assert len(reused)==len({key(r) for r in reused})==6000
    sources=copy.deepcopy(parent['source_sha256'])
    sources.update({name:sha(ROOT/'scripts'/name) for name in NEW_SOURCES})
    for gate in gates:sources.update(gate['source_sha256'])
    inputs=copy.deepcopy(parent['input_sha256'])
    for name in ('records.json','references.json','audit.json','manifest.json','accounting.json'):
        path=SOURCE/name;inputs[str(path)]=sha(path)
    for name in ('audit.json','execution_manifest.finalize_v1.json','selection_freeze.json'):
        path=PARENT/name;inputs[str(path)]=sha(path)
    for path in [CHECKS/'cpu_checks.json']+[CHECKS/f'gpu{i}/checks.json' for i in (0,1)]:inputs[str(path)]=sha(path)
    for gate in gates:inputs.update(gate.get('output_sha256',{}))
    run.mkdir(parents=True)
    configs=[dict(BASELINE,name='zero',lambda_x=0.,lambda_h=0.,taper_end=0.,geometry_weight=0.),BASELINE,*SETTINGS.values()]
    for name,value in [('tasks',tasks),('references',references),('reused_records',reused),('configurations',configs),('records',reused)]:save(run/(name+'.json'),value)
    for name in ('tasks.json','references.json','reused_records.json','configurations.json'):inputs[str(run/name)]=sha(run/name)
    workers=[dict(gpu=str(gpu),arm=arm,case_ids=[t['case_id'] for t in tasks],outputs=3000) for gpu,arm in enumerate(SETTINGS)]
    manifest=dict(created_utc=now(),cases=30,target_groups=7,attempts_per_case=100,outputs_per_arm=3000,
        total_outputs=12000,new_sampling_outputs=6000,reused_outputs=6000,seed_root=SEED_ROOT,seed_roots=1,batch_size=BATCH,
        timesteps=500,configurations=configs,workers=workers,case_ids=[t['case_id'] for t in tasks],
        previous_selection_case_ids=read(PARENT/'manifest.json')['previous_selection_case_ids'],source_run=str(SOURCE),hardware=hardware(),
        source_sha256=sources,input_sha256=inputs,
        technical_checks=dict(paths=[str(CHECKS/'cpu_checks.json')]+[str(CHECKS/f'gpu{i}/checks.json') for i in (0,1)],
            technical_outputs=sum(g['technical_outputs'] for g in gates[1:]),counts_as_efficacy=False,
            baseline_equation_steps=sum(g['baseline_equation_steps'] for g in gates[1:])),
        authorization='User accepted the C/D proposal: two full 3000-attempt arms, plus separate core-graph/internal-energy diagnostics. Vina must be evaluated but never used to guide.',
        protocol=dict(objective='CNN affinity only in coordinate and element channels; geometry constraints contain no binding-energy score.',
            C='Same as B except local geometry weight 0.1 -> 0.3; final 100 reverse steps.',
            D='Same B proposal (weight 0.1), followed by projection against bond, angle and severe internal-overlap gradients; uniform cap and backtracking enforce nonincrease of all three frozen-graph penalties for the stored x0 coordinate update.',
            D_overlap='Pairs with shortest graph distance >=3 and at least one R atom. Penalty activates below 0.65*(RDKit vdW radius sum). No receptor terms.',
            D_projection='32 alternating linear projection sweeps; at most 16 halvings; zero coordinate update if no acceptable step. No extra CNN scoring or molecule selection.',
            D_limit='Constraint applies only to the proposed coordinate correction, not to subsequent diffusion noise, changing elements/graphs or final geometry.',
            context='Missing discrete context falls back to unchanged CNN coordinate update and is counted.',
            sampling='Native 500 steps, 502 paired noise draws, fixed core, no redraws, no final repair.',
            scoring='Original H preparation, GNINA 1.1 --scoring vina --cnn_scoring rescore --score_only and complete 22-check PoseBusters. Both metrics retained and reported for both arms automatically.',
            selector='Valid-new >= B and strictly more valid-new CNN successes, maximize CNN successes then lexicographic arm; no Vina fields passed to selector.',
            diagnosis='Core graph and energy-check diagnostics are separate; no generated output, judge or control is modified by those diagnostics.',
            scope='Previously inspected development cohort; no independent validation claim.'))
    save(run/'manifest.json',manifest)
    save(run/'status.json',dict(stage='prepared',created_utc=now(),generated_outputs=0,new_total_outputs=6000,reused_outputs=6000,total_outputs=12000,execution_manifest=str(run/'manifest.json')))
    integrity(manifest);print(json.dumps(dict(prepared=str(run),workers=workers)),flush=True)


def score_case(private,records,task,required,update):
    jobs=[r for r in records if r['case_id']==task['case_id'] and r['raw_audit']['strict_success'] and 'raw_score' not in r]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(score_one,Chem.Mol((Path(r['path'])/'raw.rdmol').read_bytes()),task['protein_path'],Path(r['path'])/'judge'):r for r in jobs}
        for done,future in enumerate(cf.as_completed(pending),1):
            row=pending[future];s=dict(future.result());missing=sorted(required-set(s.get('posebusters_checks',{})))
            finite=all(np.isfinite(s.get(k,float('nan'))) for k in ('vina_kcal_mol','cnn_affinity_pK','cnn_pose_score'))
            s.update(legacy_posebusters_pass=s.get('posebusters_pass',False),posebusters_complete=not missing,missing_checks=missing)
            s['posebusters_pass']=bool(s['legacy_posebusters_pass'] and not missing and finite and s['status']=='scored')
            row['raw_score']=s;save(Path(row['path'])/'judge/verified_score.json',s)
            if done%10==0 or done==len(jobs):
                save(private/'records.json',records);update('scoring',active_case=task['case_id'],completed_scores=done,total_scores=len(jobs))
    save(private/'records.json',records)


def finalize(run,manifest,records,update):
    integrity(manifest)
    tasks=read(run/'tasks.json');taskmap={t['case_id']:t for t in tasks}
    assert len(records)==len({key(r) for r in records})==12000
    assert Counter(r['arm'] for r in records)=={a:3000 for a in ARMS}
    reused={key(r):r for r in read(run/'reused_records.json')}
    baseline={(r['case_id'],r['attempt']):r for r in records if r['arm']=='zero'}
    assert validate_core_metadata(records,taskmap)==1500
    for row in records:
        if row['reused']:assert row==reused[key(row)]
        base=baseline[row['case_id'],row['attempt']]
        assert row['seed']==base['seed'] and row['noise_tape_sha256']==base['noise_tape_sha256'] and row['noise_draws']==502
        mol=None
        if row['reconstructed']:
            path=Path(row['path'])/'raw.rdmol';assert sha(path)==row['raw_state_sha256'];mol=Chem.Mol(path.read_bytes())
        assert audit(mol,taskmap[row['case_id']])==row['raw_audit'],key(row)
        assert not row['raw_audit']['strict_success'] or 'raw_score' in row
    update('auditing_trajectories',verified_records=12000)
    new=[r for r in records if not r['reused']];assert len(new)==6000
    configs={Path(r['batch_path']):SETTINGS[r['arm']] for r in new};assert len(configs)==3000
    counts={a:dict(events=0,batches=0,geometry_applied=0,geometry_context_missing=0,geometry_parameter_missing=0,
                   coordinate_x0_max_A=0.,feature_x0_max=0.,projection_accepted=0,projection_zero=0,projection_missing_context=0) for a in SETTINGS}
    event_count=0
    for index,path in enumerate(sorted(configs)):
        config=configs[path];count=counts[config['name']]
        commit=read(path/'commit.json')
        for name,digest in commit['files'].items():assert sha(path/name)==digest,('batch changed',path,name)
        case=commit['case_id'];offset=commit['offset'];nc=taskmap[case]['fixed_heavy_atoms']
        with np.load(path/'guidance_trace.npz') as t:
            steps=t['steps'].tolist();assert sorted(steps)==list(range(500)) and t['seeds'].tolist()==commit['seeds']
            dx=t['delta_x0_coordinates'];dh=t['delta_x0_features'];before=t['coordinates_before_geometry']
            assert dx.shape[:2]==dh.shape[:2]==(500,2)
            for name in ('predicted_xyz','predicted_features','delta_x0_coordinates','delta_x0_features','posterior_weight','coordinates_before_geometry','effective_h'):
                assert np.isfinite(t[name]).all()
            assert not np.count_nonzero(dx[:,:,:nc]) and not np.count_nonzero(dh[:,:,:nc])
            expected=np.array([element_strength(config,s) for s in steps])
            assert np.array_equal(expected,t['effective_h'])
            assert np.all(np.linalg.norm(dh.astype(np.float64),axis=-1).max(axis=(1,2))<=expected+1e-7)
            assert np.linalg.norm(dx.astype(np.float64),axis=-1).max()<=.02+1e-7
            inactive=np.array([s>=100 or not config['geometry_weight'] for s in steps])
            assert np.array_equal(dx[inactive],before[inactive])
            count['coordinate_x0_max_A']=max(count['coordinate_x0_max_A'],float(np.linalg.norm(dx.astype(np.float64),axis=-1).max()))
            count['feature_x0_max']=max(count['feature_x0_max'],float(np.linalg.norm(dh.astype(np.float64),axis=-1).max()))
        seen=set()
        with (path/'events.jsonl').open() as f:
            for line in f:
                e=json.loads(line);assert_clean(e);k=(e['attempt'],e['step']);assert k not in seen;seen.add(k)
                assert e['case_id']==case and e['attempt'] in (offset,offset+1) and 0<=e['step']<500
                assert e['seed']==baseline[case,e['attempt']]['seed']
                assert e['lambda_x']==.02 and e['lambda_h']==.6 and e['status']=='guided'
                assert e['lambda_h_effective']==element_strength(config,e['step'])
                assert e['geometry_weight']==config['geometry_weight']
                assert e['joint_model_forward_calls']==e['joint_backward_calls']==1
                assert e['scored_element_substitution_candidates']==0 and np.isfinite(e['affinity_pK'])
                if e['step']>=100 or not config['geometry_weight']:assert e['geometry']['status']=='inactive'
                else:
                    assert e['geometry']['status'] in ('applied','no_valid_context','no_available_parameters')
                    count['geometry_applied']+=e['geometry']['status']=='applied'
                    count['geometry_context_missing']+=e['geometry']['status']=='no_valid_context'
                    count['geometry_parameter_missing']+=e['geometry']['status']=='no_available_parameters'
                projection=e['projection']
                if config['geometry_projection'] and e['step']<config['late_steps']:
                    assert projection['status'] in ('accepted','zero_fallback','zero_storage_fallback','no_valid_context')
                    count['projection_accepted']+=projection['status']=='accepted'
                    count['projection_zero']+=projection['status'] in ('zero_fallback','zero_storage_fallback')
                    count['projection_missing_context']+=projection['status']=='no_valid_context'
                    if projection['status']!='no_valid_context':
                        assert np.all(np.array(projection['stored_after'])<=np.array(projection['before'])+np.array(projection['tolerance']))
                        assert abs(projection['applied_max_A']-e['actual_x0_coordinate_max_change_A'])<1e-7
                else:assert projection['status']=='inactive'
                count['events']+=1;event_count+=1
        assert len(seen)==1000;count['batches']+=1
        with np.load(path/'sampling_trace.npz') as t,np.load(TRAJECTORIES/'arms/zero/sampling_trace'/f'{case}_{offset:04d}.npz') as old:
            assert np.array_equal(t['xyz'][:,:nc],old['xyz'][:,:nc])
            assert np.array_equal(t['atomic_numbers'][:,:nc],old['atomic_numbers'][:,:nc])
            for i,row in enumerate(read(path/'records.json')):
                assert t['seeds'][i]==row['seed']
                if row['reconstructed']:
                    mol=Chem.Mol((Path(row['path'])/'raw.rdmol').read_bytes())
                    assert [a.GetAtomicNum() for a in mol.GetAtoms()]==t['atomic_numbers'][i].tolist()
                    assert np.max(np.abs(mol.GetConformer().GetPositions()-t['xyz'][i]))<.000051
        if (index+1)%50==0:update('auditing_trajectories',verified_batches=index+1,verified_events=event_count)
    assert event_count==3000000 and all(c['events']==1500000 and c['batches']==1500 for c in counts.values())
    assert all(c['geometry_applied']>0 for c in counts.values())
    assert counts['cnn_geometry_projected']['projection_accepted']>0
    assert counts['cnn_geometry_weight_0p3']['projection_accepted']==0
    update('accounting');accounting=write_accounting(run,records,tasks,read(run/'references.json'))
    save(run/'mechanism.json',counts);integrity(manifest)
    save(run/'audit.json',dict(passed=True,cases=30,target_groups=7,attempts_per_case=100,new_outputs=6000,reused_outputs=6000,
        total_outputs=12000,core_fixed=True,paired_seeds_and_all_noise=True,native_final_reconstruction=True,
        gradient_events_verified=event_count,guidance_batch_steps_verified=1500000,committed_batches_verified=3000,
        all_failures_retained=True,reused_controls_unchanged=True,accounting_cross_checked=True,
        guidance_and_selector_exclude_external_score=True,external_evaluation_unblinded=True,
        audit_norm_accumulation_dtype='float64',unchanged_update_tolerance=1e-7,
        audit_execution_manifest=str(run/'manifest.json'),
        records_sha256=sha(run/'records.json'),accounting_sha256=sha(run/'accounting.json')))
    save(run/'selection_freeze.json',dict(created_utc=now(),selected_arm=accounting['selection']['selected_arm'],
         manifest_sha256=sha(run/'manifest.json'),selection_sha256=sha(run/'selection.json'),external_evaluation_unblinded=True,
         audit_execution_manifest_sha256=sha(run/'manifest.json')))
    update('complete',completed_utc=now(),result_file=str(run/'accounting.txt'))



def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(RUN));p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--worker',choices=('0','1'));p.add_argument('--finalize-only',action='store_true')
    a=p.parse_args();run=Path(a.run).resolve();RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4)
    # Reuse the frozen transaction/resume machinery; scientific functions are
    # explicitly provided by this separately hashed experiment version.
    transport.SCRIPT=SCRIPT;transport.SETTINGS=SETTINGS;transport.ARMS=ARMS
    transport.CleanEngine=GeometryEngine;transport.score_case=score_case;transport.finalize=finalize
    if a.prepare_only:prepare(run)
    elif a.worker is not None:transport.worker(run,a.worker)
    else:transport.coordinate(run,a.finalize_only)


if __name__=='__main__':main()
