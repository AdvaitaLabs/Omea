"""Expand the frozen winning guidance setting to the entire registered 30 cases.

Two workers own disjoint cases and private output/checkpoint directories. The
coordinator is the only writer of aggregate records. Existing audited outputs
are reused; exactly 2,000 missing outputs are generated, then audited/accounted.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('WANDB_MODE','disabled')
import argparse
from collections import Counter
import concurrent.futures as cf
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
from rdkit import Chem, RDLogger
from run_direct_guidance_parameter_screen_10case import (
    NativeFinalEngine, make_noise, now, sha, save, SOURCE, TRAJECTORIES, SEED_ROOT, BATCH)
from tasks import ROOT, audit, read_mol
from score_extra_results import score_one
from evaluate_direct_guidance import summarize
from summarize_selected_guidance_full import write_accounting, BEST, ARMS

SCREEN=ROOT/'results/direct_guidance_parameter_screen_10case_v1'
PRIOR=ROOT/'results/direct_guidance_parameter_screen_2case_v3'
SCRIPT=ROOT/'scripts/run_selected_guidance_full.py'
PYTHON=ROOT/'envs/generators/bin/python'
ATTEMPTS=100


def read(path):return json.loads(path.read_text())
def key(row):return row['arm'],row['case_id'],row['attempt']
def hardware():return subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True)


def integrity(manifest):
    for name,digest in manifest['source_sha256'].items():
        assert sha(ROOT/'scripts'/name)==digest,('source changed',name)
    for path,digest in manifest['input_sha256'].items():
        assert sha(path)==digest,('input changed',path)


def batch(task,offset):
    attempts=list(range(offset,offset+BATCH))
    seeds=[(SEED_ROOT+int(hashlib.sha256(f"{task['case_id']}/{i}".encode()).hexdigest()[:8],16))%(2**32) for i in attempts]
    return attempts,seeds,make_noise(seeds,task['fixed_heavy_atoms']+task['rgroup_heavy_atoms'])



def validate_core_metadata(records,taskmap):
    """Legacy recovered controls expose source traces; new samplers log core error."""
    cores={};checked=set()
    for row in records:
        assert row['noise_draws']==502
        if 'core_max_normalized_error' in row:
            assert row['core_max_normalized_error']==0
            continue
        assert row['reused'] and row['arm'] in ARMS[:-1]
        case=row['case_id'];offset=row['attempt']//BATCH*BATCH
        expected=TRAJECTORIES/'arms'/row['arm']/'sampling_trace'/f'{case}_{offset:04d}.npz'
        assert Path(row['source_trace'])==expected
        if expected in checked:continue
        task=taskmap[case];nc=task['fixed_heavy_atoms']
        if case not in cores:
            core=read_mol(task['core_path'])
            cores[case]=(core.GetConformer().GetPositions(),np.array([a.GetAtomicNum() for a in core.GetAtoms()]))
        xyz,numbers=cores[case]
        with np.load(expected) as trace:
            assert np.max(np.abs(trace['xyz'][:,:nc]-xyz[None,:,:]))<.000051,('legacy core coordinates',expected)
            assert np.array_equal(trace['atomic_numbers'][:,:nc],np.tile(numbers,(BATCH,1))),('legacy core elements',expected)
            seeds=[(SEED_ROOT+int(hashlib.sha256(f"{case}/{i}".encode()).hexdigest()[:8],16))%(2**32)
                   for i in range(offset,offset+BATCH)]
            assert np.array_equal(trace['seeds'],seeds)
        checked.add(expected)
    return len(checked)


def prepare(run,gpus):
    assert not run.exists(),run
    screening=read(SCREEN/'manifest.json')
    integrity(screening)
    assert read(SCREEN/'status.json')['stage']=='complete' and read(SCREEN/'audit.json')['passed']
    assert read(SCREEN/'selection.json')['selected_global_arm']==BEST
    assert read(SOURCE/'audit.json')['passed'] and read(SOURCE/'comparison.json')['complete']
    assert read(TRAJECTORIES/'manifest.json')['seed_root']==SEED_ROOT
    assert screening['batch_size']==BATCH==2 and screening['seed_roots']==1
    tasks=read(SOURCE/'tasks.json');selection_tasks=read(SCREEN/'tasks.json')
    taskmap={t['case_id']:t for t in tasks}
    assert len(tasks)==30 and len(taskmap)==30 and len({t['target_group'] for t in tasks})==7
    assert all(taskmap[t['case_id']]==t for t in selection_tasks)
    selected_ids=[t['case_id'] for t in selection_tasks]
    additional=[t['case_id'] for t in tasks if t['case_id'] not in set(selected_ids)]
    assert len(selected_ids)==10 and len(additional)==20
    configs=[next(c for c in screening['configurations'] if c['name']==name) for name in ARMS]
    assert configs[-1]['lambda_x']==.02 and configs[-1]['lambda_h']==.6
    references_file=ROOT/'results/direct_guidance_0p4_expansion_v1/expanded_references.json'
    references={r['case_id']:r for r in read(references_file)['rows'] if r['model']=='reference'}
    assert set(references)==set(taskmap) and all(r['status']=='scored' for r in references.values())
    records=[]
    for source,arm_names in [(SOURCE,ARMS[:-1]),(SCREEN,[BEST])]:
        original=read(source/'records.json')
        for row in original:
            if row['arm'] not in arm_names:continue
            r=copy.deepcopy(row)
            r.update(reused=True,reused_source_run=str(source),reused_original_path=row['path'])
            records.append(r)
    assert Counter(r['arm'] for r in records)=={'zero':3000,'joint_0p4':3000,BEST:1000}
    assert len({key(r) for r in records})==7000
    baseline={(r['case_id'],r['attempt']):r for r in records if r['arm']=='zero'}
    for row in records:
        base=baseline[row['case_id'],row['attempt']]
        assert row['seed']==base['seed'] and row['noise_tape_sha256']==base['noise_tape_sha256']
        assert not row['raw_audit']['strict_success'] or 'raw_score' in row
        if row['reconstructed']:assert sha(Path(row['path'])/'raw.rdmol')==row['raw_state_sha256']
    assert validate_core_metadata(records,taskmap)==3000
    assert len(gpus)==2 and len(set(gpus))==2
    hw=hardware();assert set(gpus)<={line.split(',')[0].strip() for line in hw.splitlines()}
    workers=[dict(gpu=gpu,case_ids=additional[i::len(gpus)],outputs=1000) for i,gpu in enumerate(gpus)]
    assert set().union(*(set(w['case_ids']) for w in workers))==set(additional)
    assert not (set(workers[0]['case_ids']) & set(workers[1]['case_ids']))
    names=list(screening['source_sha256'])+['run_selected_guidance_full.py','summarize_selected_guidance_full.py']
    inputs=dict(screening['input_sha256'])
    for source,files in [(SCREEN,['manifest.json','tasks.json','records.json','comparison.json','selection.json','audit.json']),
                         (SOURCE,['manifest.json','tasks.json','records.json','comparison.json','audit.json']),
                         (TRAJECTORIES,['manifest.json'])]:
        for name in files:inputs[str(source/name)]=sha(source/name)
    for source in (PRIOR,SCREEN):
        path=source/'arms'/BEST/'events.jsonl';inputs[str(path)]=sha(path)
    for task in tasks:
        for name in ('core_path','protein_path','reference_path'):inputs[task[name]]=sha(task[name])
    inputs[str(references_file)]=sha(references_file)
    manifest=dict(created_utc=now(),cases=30,target_groups=7,attempts_per_case=ATTEMPTS,
                  outputs_per_arm=3000,total_outputs=9000,new_sampling_outputs=2000,reused_outputs=7000,
                  seed_root=SEED_ROOT,seed_roots=1,batch_size=BATCH,configurations=configs,workers=workers,
                  case_ids=list(taskmap),selection_case_ids=selected_ids,additional_case_ids=additional,
                  selection_run=str(SCREEN),control_run=str(SOURCE),hardware=hw,
                  source_sha256={name:sha(ROOT/'scripts'/name) for name in dict.fromkeys(names)},input_sha256=inputs,
                  protocol=dict(scope='All 30 pre-existing common multi-atom benchmark cases, unchanged input order and inclusion.',
                                selection='Freeze x0p02_h0p6 selected on ten cases before generating the remaining twenty.',
                                sampling='100 attempts per case, seed root 2026101203, paired 502 Gaussian draws and original batch size 2.',
                                generation='Unchanged NativeFinalEngine, 500 steps, full guidance, temperature 0.2, original clipping; native final reconstruction.',
                                controls='Reuse audited native-reconstruction zero and joint_0p4, 3000 records each; selected arm reuses 1000 screened records.',
                                evaluation='Same H-only preparation, fixed-heavy-atom Vina/GNINA, complete 22-check PoseBusters; all failures retained.',
                                accounting='Full30, selected10, additional20, each case and target; paired all-scored, both-valid and common-valid-triplet scores.',
                                limits='No resampling, per-output selection, extra seed roots, post-generation heavy-atom optimization or automatic further expansion.'),
                  limitations=['Additional20 share existing target groups and the single seed root; not an independent-target or multi-seed claim.'])
    run.mkdir(parents=True)
    (run/'source').mkdir()
    for name in manifest['source_sha256']:shutil.copy2(ROOT/'scripts'/name,run/'source'/name)
    for name,data in [('manifest',manifest),('tasks',tasks),('configurations',configs),('references',references),
                      ('reused_records',records),('records',records)]:save(run/(name+'.json'),data)
    for name in ('tasks','configurations','references','reused_records'):
        path=run/(name+'.json');manifest['input_sha256'][str(path)]=sha(path)
    save(run/'manifest.json',manifest)
    save(run/'status.json',dict(stage='prepared',created_utc=now(),updated_utc=now(),generated_outputs=0,
                               new_total_outputs=2000,reused_outputs=7000,total_outputs=9000))
    print(json.dumps(dict(stage='prepared',run=str(run),new_outputs=2000,reused_outputs=7000,workers=workers)),flush=True)
    return manifest


def preflight(engine,private,tasks,manifest,controls,selected,update):
    gate=private/'preflight_checks.json'
    if gate.exists():
        result=read(gate);assert result['passed'] and result['hardware']==manifest['hardware'];return
    configs={c['name']:c for c in manifest['configurations']}
    known_case=manifest['selection_case_ids'][0]
    all_tasks={t['case_id']:t for t in read(SOURCE/'tasks.json')}
    jobs=[(arm,t,controls,TRAJECTORIES/'arms'/arm/'sampling_trace'/f"{t['case_id']}_0000.npz")
          for arm in ARMS[:-1] for t in tasks]
    row=selected[BEST,known_case,0]
    jobs.append((BEST,all_tasks[known_case],selected,Path(row['path']).parents[1]/'sampling_trace'/f'{known_case}_0000.npz'))
    checks=[]
    for arm,task,old_rows,expected_path in jobs:
        update('preflight_replay',active_arm=arm,active_case=task['case_id'],passed_replay_batches=len(checks),total_replay_batches=len(jobs))
        engine.hook.configure(configs[arm],private/'preflight'/arm)
        attempts,seeds,noise=batch(task,0)
        with torch.no_grad():outputs,xyz,numbers,metadata=engine.generate(task,attempts,seeds,noise)
        with np.load(expected_path) as expected:
            delta=float(np.max(np.abs(xyz-expected['xyz'])))
            assert np.array_equal(numbers,expected['atomic_numbers']),('replay elements',arm,task['case_id'])
            assert delta<.001,('replay coordinates',arm,task['case_id'],delta)
        replay=private/'preflight'/arm/task['case_id'];replay.mkdir(exist_ok=True)
        np.savez_compressed(replay/'sampling_trace.npz',xyz=xyz,atomic_numbers=numbers,seeds=seeds)
        scores=[]
        for i,(mol,error) in enumerate(outputs):
            old=old_rows[arm,task['case_id'],attempts[i]]
            assert metadata['noise_tape_sha256']==old['noise_tape_sha256']
            current=audit(mol,task);assert current==old['raw_audit'],('replay audit',arm,task['case_id'],attempts[i])
            save(replay/f'audit_{attempts[i]:04d}.json',current)
            if current['strict_success']:
                s=score_one(mol,task['protein_path'],replay/f'judge_{attempts[i]:04d}');previous=old['raw_score']
                assert s['status']==previous['status']=='scored'
                assert s['posebusters_checks']==previous['posebusters_checks']
                changes={m:s[m]-previous[m] for m in ('vina_kcal_mol','cnn_affinity_pK','cnn_pose_score')}
                assert all(abs(v)<.001 for v in changes.values()),('replay scores',changes)
                scores.append(dict(attempt=attempts[i],score_deltas=changes,all_geometry_checks_equal=True))
        checks.append(dict(arm=arm,case_id=task['case_id'],attempts=attempts,max_coordinate_delta_A=delta,
                           elements_and_audits_equal=True,scored_replays=scores))
        save(private/'preflight_progress.json',dict(passed_so_far=True,rows=checks))
    save(gate,dict(passed=True,rows=checks,hardware=manifest['hardware'],technical_replayed_outputs=2*len(checks),
                   coordinate_tolerance_A=.001,score_tolerance=.001,counts_as_new_efficacy_outputs=False))


def score_case(private,records,task,required,update):
    jobs=[r for r in records if r['case_id']==task['case_id'] and r['raw_audit']['strict_success'] and 'raw_score' not in r]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(score_one,Chem.Mol((Path(r['path'])/'raw.rdmol').read_bytes()),task['protein_path'],Path(r['path'])/'judge'):r for r in jobs}
        for done,future in enumerate(cf.as_completed(futures),1):
            row=futures[future];score=dict(future.result());missing=sorted(required-set(score.get('posebusters_checks',{})))
            score.update(legacy_posebusters_pass=score.get('posebusters_pass',False),posebusters_complete=not missing,missing_checks=missing)
            score['posebusters_pass']=score['legacy_posebusters_pass'] and not missing and score['status']=='scored'
            row['raw_score']=score;save(Path(row['path'])/'judge'/'verified_score.json',score)
            if done%10==0 or done==len(jobs):
                save(private/'records.json',records);update('scoring',active_case=task['case_id'],completed_scores=done,total_scores=len(jobs))
    save(private/'records.json',records)


def worker(run,gpu):
    manifest=read(run/'manifest.json');integrity(manifest)
    assert os.environ['CUDA_VISIBLE_DEVICES']==gpu and hardware()==manifest['hardware']
    shard=next(w for w in manifest['workers'] if w['gpu']==gpu)
    private=run/'workers'/('gpu'+gpu);private.mkdir(parents=True,exist_ok=True)
    records=read(private/'records.json') if (private/'records.json').exists() else []
    state=dict(pid=os.getpid(),gpu=gpu,started_utc=now(),assigned_cases=shard['case_ids'],total_outputs=shard['outputs'])
    timing=read(private/'timing.json') if (private/'timing.json').exists() else dict(batches=[],scoring_seconds=0)
    def update(stage,**values):
        state.update(stage=stage,updated_utc=now(),generated_outputs=len(records),**values)
        save(private/'status.json',state);print(json.dumps(state),flush=True)
    save(private/'records.json',records);save(private/'timing.json',timing)
    taskmap={t['case_id']:t for t in read(run/'tasks.json')};tasks=[taskmap[c] for c in shard['case_ids']]
    required=set(read(ROOT/'results/selection_rules_v3/geometry_schema.json')['required_checks']);assert len(required)==22
    controls={key(r):r for r in read(SOURCE/'records.json') if r['arm'] in ARMS[:-1]}
    selected={key(r):r for r in read(SCREEN/'records.json') if r['arm']==BEST}
    cfg=next(c for c in manifest['configurations'] if c['name']==BEST)
    assert len({key(r) for r in records})==len(records)
    committed={(r['case_id'],r['attempt']) for r in records}
    counts=Counter((r['case_id'],r['attempt']//2) for r in records)
    assert all(n==2 for n in counts.values())
    event_path=private/'arms'/BEST/'events.jsonl'
    events=Counter()
    if event_path.exists():
        with event_path.open() as handle:
            for line in handle:
                e=json.loads(line);events[e['case_id'],e['attempt']]+=1
    assert set(events)==committed and all(n==500 for n in events.values()),'Uncommitted events: archive interrupted batch before resuming.'
    for row in records:
        assert row['case_id'] in shard['case_ids'] and row['arm']==BEST and not row['reused']
        if row['reconstructed']:assert sha(Path(row['path'])/'raw.rdmol')==row['raw_state_sha256']
    engine=None
    try:
        update('loading_checkpoint',active_arm=None,active_case=None)
        engine=NativeFinalEngine();assert engine.edm.T==500
        preflight(engine,private,tasks,manifest,controls,selected,update)
        engine.hook.configure(cfg,private/'arms'/BEST)
        for task in tasks:
            for offset in range(0,ATTEMPTS,BATCH):
                present=[(task['case_id'],i) in committed for i in range(offset,offset+BATCH)]
                if any(present):assert all(present);continue
                update('generating',active_arm=BEST,active_case=task['case_id'],active_offset=offset)
                attempts,seeds,noise=batch(task,offset);began=time.perf_counter()
                with torch.no_grad():outputs,xyz,numbers,metadata=engine.generate(task,attempts,seeds,noise)
                seconds=time.perf_counter()-began
                trace=private/'arms'/BEST/'sampling_trace';trace.mkdir(exist_ok=True)
                np.savez_compressed(trace/f"{task['case_id']}_{offset:04d}.npz",xyz=xyz,atomic_numbers=numbers,seeds=seeds)
                with np.load(TRAJECTORIES/'arms/zero/sampling_trace'/f"{task['case_id']}_{offset:04d}.npz") as expected:
                    nc=task['fixed_heavy_atoms']
                    assert np.array_equal(numbers[:,:nc],expected['atomic_numbers'][:,:nc])
                    assert np.array_equal(xyz[:,:nc],expected['xyz'][:,:nc])
                batch_rows=[]
                for i,(mol,error) in enumerate(outputs):
                    base=controls['zero',task['case_id'],attempts[i]]
                    assert seeds[i]==base['seed'] and metadata['noise_tape_sha256']==base['noise_tape_sha256']
                    path=private/'arms'/BEST/task['case_id']/f'{attempts[i]:04d}';path.mkdir(parents=True,exist_ok=True)
                    row=dict(arm=BEST,case_id=task['case_id'],attempt=attempts[i],seed=seeds[i],reused=False,path=str(path),
                             reconstructed=mol is not None,reconstruction_error=error,raw_audit=audit(mol,task),**metadata)
                    if mol is not None:
                        options=Chem.PropertyPickleOptions.AllProps|Chem.PropertyPickleOptions.CoordsAsDouble
                        (path/'raw.rdmol').write_bytes(mol.ToBinary(options));row['raw_state_sha256']=sha(path/'raw.rdmol')
                        with Chem.SDWriter(str(path/'raw.sdf')) as writer:
                            writer.SetKekulize(False);writer.write(Chem.Mol(mol))
                    save(path/'attempt.json',row);batch_rows.append(row)
                records.extend(batch_rows);committed.update((r['case_id'],r['attempt']) for r in batch_rows)
                timing['batches'].append(dict(case_id=task['case_id'],offset=offset,seconds=seconds))
                save(private/'records.json',records);save(private/'timing.json',timing)
            began=time.perf_counter();update('scoring',active_case=task['case_id'],completed_scores=0)
            score_case(private,records,task,required,update)
            timing['scoring_seconds']+=time.perf_counter()-began;save(private/'timing.json',timing)
        assert len(records)==shard['outputs']==1000
        assert all(not r['raw_audit']['strict_success'] or 'raw_score' in r for r in records)
        integrity(manifest);update('complete',active_arm=None,active_case=None,completed_utc=now())
    except BaseException as error:
        save(private/'records.json',records);update('failed',error=repr(error));raise
    finally:
        if engine is not None:engine.hook.close()


def audit_and_account(run,manifest,records,update):
    integrity(manifest);tasks=read(run/'tasks.json');taskmap={t['case_id']:t for t in tasks}
    baseline={(r['case_id'],r['attempt']):r for r in records if r['arm']=='zero'}
    assert len(records)==len({key(r) for r in records})==9000
    assert Counter(r['arm'] for r in records)=={arm:3000 for arm in ARMS}
    assert sum(not r['reused'] for r in records)==2000
    assert validate_core_metadata(records,taskmap)==3000
    old_records={key(r):r for r in read(run/'reused_records.json')}
    for row in records:
        if row['reused']:assert row==old_records[key(row)],('reused record changed',key(row))
        base=baseline[row['case_id'],row['attempt']]
        assert row['seed']==base['seed'] and row['noise_tape_sha256']==base['noise_tape_sha256']
        assert row['noise_draws']==502
        assert not row['raw_audit']['strict_success'] or 'raw_score' in row
        mol=None
        if row['reconstructed']:
            path=Path(row['path'])/'raw.rdmol';assert sha(path)==row['raw_state_sha256']
            mol=Chem.Mol(path.read_bytes())
        assert audit(mol,taskmap[row['case_id']])==row['raw_audit'],('saved output audit',key(row))
    update('auditing_gradient_events',verified_records=len(records))
    origins={}
    for row in records:
        if row['arm']==BEST:origins.setdefault(Path(row['path']).parents[1],set()).add((row['case_id'],row['attempt']))
    mechanism=dict(event_count=0,guided_events=0,chemistry_skips=0,coordinate_score_calls=0,
                   joint_model_forward_calls=0,joint_backward_calls=0,scored_element_substitution_candidates=0,
                   coordinate_x0_max_A=0.,coordinate_reverse_mean_max_A=0.,feature_x0_max=0.,feature_reverse_mean_max=0.,
                   final_discrete_outputs_per_seed=1,terminal_edits=0,trajectory_branches=0)
    maxima=dict(coordinate_x0_max_A='actual_x0_coordinate_max_change_A',
                coordinate_reverse_mean_max_A='actual_reverse_mean_coordinate_max_change_A',
                feature_x0_max='actual_x0_feature_max_change',feature_reverse_mean_max='actual_reverse_mean_feature_max')
    origin_audits=[];all_keys=set()
    for origin,keys in origins.items():
        assert not (all_keys & keys);all_keys.update(keys)
        masks={};count=0
        with (origin/'events.jsonl').open() as handle:
            for line in handle:
                event=json.loads(line);k=event['case_id'],event['attempt'];assert k in keys
                assert event['status']=='guided' and event['joint_model_forward_calls']==event['joint_backward_calls']==1
                assert event['lambda_x']==.02 and event['lambda_h']==.6
                assert event['scored_element_substitution_candidates']==0
                assert event['diagnostics']['all_R_features_participate']
                assert event['seed']==baseline[k]['seed'] and 0<=event['step']<500
                assert all(i>=taskmap[k[0]]['fixed_heavy_atoms'] for i in event['actual_x0_feature_active_sites'])
                bit=1<<event['step'];old=masks.get(k,0);assert not (old&bit),('duplicate gradient event',origin,k,event['step'])
                masks[k]=old|bit;count+=1
                for name in ('coordinate_score_calls','joint_model_forward_calls','joint_backward_calls','scored_element_substitution_candidates'):
                    mechanism[name]+=event[name]
                for name,field in maxima.items():mechanism[name]=max(mechanism[name],event[field])
        assert set(masks)==keys and all(mask==(1<<500)-1 for mask in masks.values())
        assert count==len(keys)*500
        mechanism['event_count']+=count;mechanism['guided_events']+=count
        new_origin=origin.is_relative_to(run/'workers')
        trace_count=0
        if new_origin:
            update('auditing_new_traces',active_origin=str(origin),verified_events=mechanism['event_count'])
            for path in (origin/'guidance_traces').glob('*.npz'):
                case=path.name.rsplit('_',2)[0];nc=taskmap[case]['fixed_heavy_atoms']
                with np.load(path) as trace:
                    for name in ('delta_x0_coordinates','delta_x0_features'):
                        assert np.isfinite(trace[name]).all() and np.count_nonzero(trace[name][:,:nc])==0
                trace_count+=1
            assert trace_count==len(keys)//BATCH*500
        origin_audits.append(dict(origin=str(origin),outputs=len(keys),events=count,
                                 events_sha256=sha(origin/'events.jsonl'),new_guidance_traces_checked=trace_count,
                                 prior_trace_audit=str(SCREEN/'audit.json') if not new_origin else None))
    assert len(all_keys)==3000 and mechanism['guided_events']==1500000
    update('auditing_sampling_traces',active_origin=None)
    selected_rows={(r['case_id'],r['attempt']):r for r in records if r['arm']==BEST}
    for case in manifest['additional_case_ids']:
        nc=taskmap[case]['fixed_heavy_atoms']
        for offset in range(0,ATTEMPTS,BATCH):
            row=selected_rows[case,offset];origin=Path(row['path']).parents[1]
            with np.load(origin/'sampling_trace'/f'{case}_{offset:04d}.npz') as trace, np.load(TRAJECTORIES/'arms/zero/sampling_trace'/f'{case}_{offset:04d}.npz') as old:
                assert np.array_equal(trace['xyz'][:,:nc],old['xyz'][:,:nc])
                assert np.array_equal(trace['atomic_numbers'][:,:nc],old['atomic_numbers'][:,:nc])
                for i in range(BATCH):
                    actual=selected_rows[case,offset+i];assert trace['seeds'][i]==actual['seed']
                    if actual['reconstructed']:
                        mol=Chem.Mol((Path(actual['path'])/'raw.rdmol').read_bytes())
                        assert [a.GetAtomicNum() for a in mol.GetAtoms()]==trace['atomic_numbers'][i].tolist()
                        assert np.max(np.abs(mol.GetConformer().GetPositions()-trace['xyz'][i]))<.000051
    update('accounting',active_origin=None)
    references=read(run/'references.json')
    comparison=summarize(run,manifest['configurations'],records,tasks,references)
    assert comparison['complete']
    controls=read(SOURCE/'comparison.json')
    for arm in ARMS[:-1]:
        comparison['arms'][arm]['mechanism']=dict(controls['arms'][arm]['mechanism'],reused=True,source_run=str(SOURCE),new_sampling_outputs=0)
    mechanism.update(event_sources=origin_audits,reused_outputs=1000,new_sampling_outputs=2000)
    comparison['arms'][BEST]['mechanism']=mechanism
    comparison['scope']='Frozen winning setting on all 30 existing cases; selection10 and additional20 separately accounted.'
    save(run/'comparison.json',comparison)
    accounting=write_accounting(run,records,tasks,references,comparison)
    old_comparison=read(SCREEN/'comparison.json')
    for arm in ARMS:
        for metric in ('valid_new','dual_vs_reference'):
            assert accounting['scopes']['selection_10']['arms'][arm][metric]==old_comparison['arms'][arm]['metrics'][metric]
    integrity(manifest)
    save(run/'audit.json',dict(passed=True,cases=30,target_groups=7,attempts_per_case=100,seed_roots=1,
                              generated_outputs=2000,reused_outputs=7000,total_outputs=9000,
                              original_outputs_and_scores_unchanged=True,paired_seeds_and_noise=True,
                              native_final_reconstruction=True,fixed_core_verified=True,all_attempts_retained=True,
                              new_sampling_batches_checked=1000,legacy_control_sampling_batches_checked=3000,gradient_events_verified=1500000,
                              new_guidance_traces_checked=500000,event_sources=origin_audits,
                              accounting_totals_cross_checked=True,selection_subset_reproduces_previous_results=True,
                              records_sha256=sha(run/'records.json')))
    update('complete',completed_utc=now(),result_file=str(run/'accounting.json'),active_origin=None)


def coordinator(run,manifest,finalize_only=False):
    lock=(run/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(run/'status.json')
    if state['stage']=='complete':print('Already complete.',flush=True);return
    integrity(manifest);assert hardware()==manifest['hardware']
    reused=read(run/'reused_records.json')
    def update(stage,**values):
        state.update(stage=stage,pid=os.getpid(),updated_utc=now(),**values)
        save(run/'status.json',state);print(json.dumps(state),flush=True)
    if finalize_only:
        audit_and_account(run,manifest,read(run/'records.json'),update);return
    # Detached workers from an interrupted coordinator must not be launched twice.
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name)==os.getpid():continue
        try:argv=(path/'cmdline').read_bytes().split(b'\0')
        except (OSError,ProcessLookupError):continue
        if str(SCRIPT).encode() in argv and str(run).encode() in argv:
            raise RuntimeError(f'Existing full-cohort process {path.name}; refusing duplicate launch')
    launch=dict(pid=os.getpid(),launched_utc=now(),hardware=manifest['hardware'],workers=[],run=str(run))
    processes=[]
    for shard in manifest['workers']:
        private=run/'workers'/('gpu'+shard['gpu']);private.mkdir(parents=True,exist_ok=True)
        if not (private/'records.json').exists():save(private/'records.json',[])
        if not (private/'status.json').exists():save(private/'status.json',dict(stage='starting',generated_outputs=0))
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=shard['gpu'],OMP_NUM_THREADS='4',WANDB_MODE='disabled')
        command=[str(PYTHON),'-u',str(SCRIPT),'--run',str(run),'--worker',shard['gpu']]
        with (private/'worker.log').open('a') as output:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=output,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process)
        launch['workers'].append(dict(shard,pid=process.pid,log=str(private/'worker.log'),command=command))
    save(run/'launch.json',launch)
    cached={};merged=reused;last_print=0
    while True:
        snapshots=[];states=[];changed=False
        for shard,process in zip(manifest['workers'],processes):
            private=run/'workers'/('gpu'+shard['gpu']);path=private/'records.json';stamp=path.stat().st_mtime_ns
            if shard['gpu'] not in cached or cached[shard['gpu']][0]!=stamp:
                cached[shard['gpu']]=(stamp,read(path));changed=True
            rows=cached[shard['gpu']][1]
            assert all(r['arm']==BEST and r['case_id'] in shard['case_ids'] and not r['reused'] for r in rows)
            snapshots.extend(rows)
            status=read(private/'status.json')
            states.append(dict(gpu=shard['gpu'],pid=process.pid,returncode=process.poll(),
                               generated_outputs=len(rows),total_outputs=1000,stage=status['stage'],
                               active_arm=status.get('active_arm'),active_case=status.get('active_case'),
                               active_offset=status.get('active_offset'),passed_replay_batches=status.get('passed_replay_batches'),
                               error=status.get('error'),log=str(private/'worker.log')))
        if changed:
            merged=reused+snapshots
            assert len({key(r) for r in merged})==len(merged)<=9000
            save(run/'records.json',merged)
        state.update(stage='parallel_running' if all(p.poll() in (None,0) for p in processes) else 'parallel_worker_failed',
                     pid=os.getpid(),updated_utc=now(),generated_outputs=len(merged)-7000,workers=states)
        save(run/'status.json',state)
        if time.monotonic()-last_print>=60:print(json.dumps(state),flush=True);last_print=time.monotonic()
        if all(p.poll() is not None for p in processes):break
        time.sleep(15)
    if any(p.returncode!=0 for p in processes):
        update('failed',error='Worker failed; all committed records retained. Inspect worker logs before resuming.')
        raise RuntimeError(state['error'])
    # Read once more after both children exit so their final score commits cannot
    # race the last polling snapshot.
    merged=reused+[row for shard in manifest['workers']
                   for row in read(run/'workers'/('gpu'+shard['gpu'])/'records.json')]
    assert len({key(row) for row in merged})==len(merged)==9000
    save(run/'records.json',merged)
    gates={}
    for shard in manifest['workers']:
        private=run/'workers'/('gpu'+shard['gpu'])
        gate=read(private/'preflight_checks.json');assert gate['passed'] and gate['technical_replayed_outputs']==42
        gates[shard['gpu']]=dict(path=str(private/'preflight_checks.json'),sha256=sha(private/'preflight_checks.json'))
        assert read(private/'status.json')['stage']=='complete'
    save(run/'preflight_checks.json',dict(passed=True,per_gpu=gates,technical_replayed_outputs=84,
                                        counts_as_new_efficacy_outputs=False))
    try:
        update('auditing',generated_outputs=2000)
        audit_and_account(run,manifest,merged,update)
    except BaseException as error:
        update('failed',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='direct_guidance_selected_x0p02_h0p6_scale30_v1')
    parser.add_argument('--gpus',nargs='+',default=['0','1'])
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--finalize-only',action='store_true')
    args=parser.parse_args();RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4)
    run=(ROOT/'results'/args.run).resolve()
    if args.worker is not None:
        worker(run,args.worker);return
    manifest=read(run/'manifest.json') if run.exists() else prepare(run,args.gpus)
    if args.prepare_only:
        integrity(manifest);print('Prepared and verified; no new sampling outputs generated.',flush=True);return
    coordinator(run,manifest,args.finalize_only)


if __name__=='__main__':main()
