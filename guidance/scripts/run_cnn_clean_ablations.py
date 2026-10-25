"""Two CNN-only ablations, 3000 attempts each, transactional generation and clean selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('WANDB_MODE','disabled')
import argparse
from collections import Counter
import concurrent.futures as cf
import copy
import fcntl
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from rdkit import Chem,RDLogger
from tasks import ROOT,audit
from resume_selected_guidance_full_case_shards import (read,save,sha,now,hardware,integrity,batch,
    TRAJECTORIES,SEED_ROOT,BATCH,validate_core_metadata)
from autofragdiff_cnn_clean_hook import CleanEngine
from cnn_clean_settings import SOURCE,CHECKS,RUN,BASELINE,SETTINGS,ARMS,NEW_SOURCES,element_strength
from score_cnn_clean import score_one,project_score,assert_clean
from summarize_cnn_clean_full import write_accounting

SCRIPT=ROOT/'scripts/run_cnn_clean_ablations.py'
PYTHON=ROOT/'envs/generators/bin/python'
ATTEMPTS=100


def key(row):return row['arm'],row['case_id'],row['attempt']


def prepare(run):
    assert not run.exists(),run
    oldmanifest=read(SOURCE/'execution_manifest.finalize_v1.json');integrity(oldmanifest)
    assert read(SOURCE/'audit.json')['passed'] and read(SOURCE/'status.json')['stage']=='complete'
    assert read(CHECKS/'report_checks.json')['passed']
    assert read(CHECKS/'geometry_coverage.json')['passed']
    plan=read(CHECKS/'execution_plan.json')
    assert plan['configurations']==list(SETTINGS.values()) and plan['new_outputs']==6000 and plan['attempts_per_case']==100
    gpu_paths=[CHECKS/'gpu0/checks.json',CHECKS/'gpu1_recovery/checks.json']
    gates=[read(CHECKS/'cpu_checks.json')]+[read(p) for p in gpu_paths]
    recovery_cpu=read(CHECKS/'cpu_checks.recovery.json')
    for gate in gates+[recovery_cpu]:
        assert gate['passed']
        for name,digest in gate['source_sha256'].items():assert sha(ROOT/'scripts'/name)==digest
        for path,digest in gate.get('output_sha256',{}).items():assert sha(path)==digest
    for gpu,gate in enumerate(gates[1:]):
        assert gate['configuration']==list(SETTINGS.values())[gpu] and gate['hardware']==hardware()
        assert gate['exact_native_vs_new_equation_steps']==3500
    with np.load(CHECKS/'gpu0/baseline_trajectory.npz') as a,np.load(CHECKS/'gpu1_recovery/baseline_trajectory.npz') as b:
        assert np.array_equal(a['atomic_numbers'],b['atomic_numbers'])
        cross=float(np.max(np.abs(a['xyz']-b['xyz'])));assert cross<.001
    tasks=read(SOURCE/'tasks.json');assert len(tasks)==30
    references={case:{k:r[k] for k in ('status','cnn_affinity_pK','cnn_pose_score')} for case,r in read(SOURCE/'references.json').items()}
    reused=[]
    for row in read(SOURCE/'records.json'):
        if row['arm'] not in ARMS[:2]:continue
        r=copy.deepcopy(row);r.update(reused=True,reused_control_source=str(SOURCE))
        if 'raw_score' in r:r['raw_score']=project_score(r['raw_score'])
        assert_clean(r);reused.append(r)
    assert len(reused)==6000 and len({key(r) for r in reused})==6000
    assert_clean(references)
    sources=copy.deepcopy(oldmanifest['source_sha256'])
    sources.update({name:sha(ROOT/'scripts'/name) for name in NEW_SOURCES})
    for gate in gates+[recovery_cpu]:sources.update(gate['source_sha256'])
    inputs=copy.deepcopy(oldmanifest['input_sha256'])
    for name in ('records.json','references.json','audit.json','execution_manifest.finalize_v1.json'):
        path=SOURCE/name;inputs[str(path)]=sha(path)
    for path in [CHECKS/'cpu_checks.json',CHECKS/'cpu_checks.recovery.json']+gpu_paths:inputs[str(path)]=sha(path)
    for name in ('report_checks.json','gate_amendment.json','geometry_coverage.json','execution_plan.json',
                 'native_repeat_diagnostic/diagnostic.json','native_repeat_diagnostic/driver.py'):
        path=CHECKS/name;inputs[str(path)]=sha(path)
    for i in range(3):
        path=CHECKS/f'native_repeat_diagnostic/{i}/trajectory.npz';inputs[str(path)]=sha(path)
    for gate in gates:inputs.update(gate.get('output_sha256',{}))
    run.mkdir(parents=True)
    configs=[dict(BASELINE,name='zero',lambda_x=0.,lambda_h=0.,taper_end=0.),BASELINE,*SETTINGS.values()]
    for name,value in [('tasks',tasks),('references',references),('reused_records',reused),('configurations',configs),('records',reused)]:save(run/(name+'.json'),value)
    for name in ('tasks.json','references.json','reused_records.json','configurations.json'):inputs[str(run/name)]=sha(run/name)
    workers=[dict(gpu=str(gpu),arm=arm,case_ids=[t['case_id'] for t in tasks],outputs=3000) for gpu,arm in enumerate(SETTINGS)]
    manifest=dict(created_utc=now(),cases=30,target_groups=7,attempts_per_case=100,outputs_per_arm=3000,
        total_outputs=12000,new_sampling_outputs=6000,reused_outputs=6000,seed_root=SEED_ROOT,seed_roots=1,batch_size=BATCH,
        timesteps=500,configurations=configs,workers=workers,case_ids=[t['case_id'] for t in tasks],
        previous_selection_case_ids=oldmanifest['selection_case_ids'],source_run=str(SOURCE),hardware=hardware(),
        source_sha256=sources,input_sha256=inputs,
        technical_checks=dict(paths=[str(CHECKS/'cpu_checks.json'),str(CHECKS/'cpu_checks.recovery.json')]+[str(p) for p in gpu_paths],
            cross_gpu_baseline_coordinate_error_A=cross,technical_outputs=sum(g['technical_outputs'] for g in gates[1:]),counts_as_efficacy=False,
            previous_failed_technical_outputs=read(CHECKS/'gate_amendment.json')['historical_failed_outputs'],
            additional_original_engine_repeat_outputs=6,
            gate_amendment=str(CHECKS/'gate_amendment.json'),
            historical_coordinate_tolerance_all_passed=[g['historical_coordinate_tolerance_all_passed'] for g in gates[1:]],
            exact_native_vs_new_equation_steps=sum(g['exact_native_vs_new_equation_steps'] for g in gates[1:])),
        authorization='User requested both proposed ablations at full 30 cases x 100 attempts, with at least one clean evaluation metric.',
        protocol=dict(objective='CNN affinity only in both channels; no empirical binding-energy gradient or reward.',
            ablation_A='At reverse step s<100, feature strength=0.2+0.4*s/100; otherwise 0.6. Coordinate cap 0.02 A.',
            ablation_B='Final 100 steps: subtract 0.1 times separately clipped local geometry gradient from clipped CNN coordinate direction; cap final update at 0.02 A. Feature strength 0.6.',
            geometry='UFF equilibrium lengths +/-10%, angles +/-15 degrees; squared flat-bottom penalties averaged separately over bonds and angles touching R. No nonbonded or receptor terms. Detached intermediate graph; only R derivatives.',
            context='If intermediate reconstruction is unavailable, retain CNN guidance with zero geometry contribution and record the condition.',
            sampling='Frozen native 500-step trajectory and final reconstruction, 502 paired random draws, exact fixed-core mask; no resampling or output selection.',
            scoring='Identical fixed-heavy-atom preparation, pinned GNINA 1.1 score_only CNN outputs, complete 22-check PoseBusters. Unused empirical score output is discarded and not parsed or retained.',
            selector='Require valid-new count no lower than CNN baseline, then select strictly higher count of valid-new with CNN affinity above reference; lexicographic setting name breaks ties.',
            external_metric='Vina excluded from guidance, development records, selection and reporting; no automatic unblinding.',
            numerical_repeatability='Unmodified CNN engine showed a 0.0033 A and element/audit difference in one of three identical-noise replays on one case. All seven target-group baseline trajectories use bitwise old/new step-equation parity as the implementation gate; historical replay failures remain disclosed. Same seeds/noise do not imply bitwise identical final molecules.',
            scope='Previously inspected 30-case development cohort; no independent test or >18% joint claim from this run.'))
    save(run/'manifest.json',manifest)
    save(run/'status.json',dict(stage='prepared',created_utc=now(),generated_outputs=0,new_total_outputs=6000,
                               reused_outputs=6000,total_outputs=12000,execution_manifest=str(run/'manifest.json')))
    integrity(manifest);print(json.dumps(dict(prepared=str(run),workers=workers)),flush=True)


def score_case(private,records,task,required,update):
    jobs=[r for r in records if r['case_id']==task['case_id'] and r['raw_audit']['strict_success'] and 'raw_score' not in r]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(score_one,Chem.Mol((Path(r['path'])/'raw.rdmol').read_bytes()),task['protein_path'],Path(r['path'])/'judge'):r for r in jobs}
        for done,future in enumerate(cf.as_completed(futures),1):
            row=futures[future];score=dict(future.result());missing=sorted(required-set(score.get('posebusters_checks',{})))
            score.update(legacy_posebusters_pass=score.get('posebusters_pass',False),posebusters_complete=not missing,missing_checks=missing)
            score['posebusters_pass']=score['legacy_posebusters_pass'] and not missing and score['status']=='scored'
            assert_clean(score);row['raw_score']=score;save(Path(row['path'])/'judge'/'verified_score.json',score)
            if done%10==0 or done==len(jobs):
                save(private/'records.json',records);update('scoring',active_case=task['case_id'],completed_scores=done,total_scores=len(jobs))
    save(private/'records.json',records)


def load_committed(private):
    rows=read(private/'records.json') if (private/'records.json').exists() else []
    existing={key(r):r for r in rows}
    for path in sorted((private/'batches').glob('*/*/records.json')):
        for r in read(path):
            if key(r) not in existing:rows.append(r);existing[key(r)]=r
    assert len(rows)==len(existing)
    return rows


def worker(run,gpu):
    manifest=read(run/'manifest.json');integrity(manifest)
    assert os.environ['CUDA_VISIBLE_DEVICES']==gpu and hardware()==manifest['hardware']
    shard=next(w for w in manifest['workers'] if w['gpu']==gpu)
    private=run/'workers'/f'gpu{gpu}';private.mkdir(parents=True,exist_ok=True)
    lock=(private/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    BEST=shard['arm'];CONFIG=SETTINGS[BEST]
    rows=load_committed(private);save(private/'records.json',rows)
    timing=read(private/'timing.json') if (private/'timing.json').exists() else dict(batches=[],scoring_seconds=0.)
    state=dict(stage='starting',gpu=gpu,pid=os.getpid(),started_utc=now(),total_outputs=shard['outputs'])
    def update(stage,**values):
        state.update(stage=stage,generated_outputs=len(rows),scored_outputs=sum('raw_score' in r for r in rows),updated_utc=now(),**values)
        save(private/'status.json',state);print(json.dumps(state),flush=True)
    taskmap={t['case_id']:t for t in read(run/'tasks.json')}
    baseline={(r['case_id'],r['attempt']):r for r in read(run/'reused_records.json') if r['arm']=='zero'}
    required=set(read(ROOT/'results/selection_rules_v3/geometry_schema.json')['required_checks']);assert len(required)==22
    committed={(r['case_id'],r['attempt']) for r in rows}
    assert all(r['case_id'] in shard['case_ids'] and r['arm']==BEST and not r['reused'] for r in rows)
    for r in rows:
        if r['reconstructed']:assert sha(Path(r['path'])/'raw.rdmol')==r['raw_state_sha256']
    engine=None
    try:
        update('loading_checkpoint');engine=CleanEngine();assert engine.edm.T==500
        engine.hook.configure(CONFIG,private/'guidance')
        for case in shard['case_ids']:
            task=taskmap[case];nc=task['fixed_heavy_atoms']
            for offset in range(0,ATTEMPTS,BATCH):
                present=[(case,j) in committed for j in range(offset,offset+BATCH)]
                if any(present):assert all(present);continue
                update('generating',active_case=case,active_offset=offset)
                final=private/'batches'/case/f'{offset:04d}';assert not final.exists()
                temporary=private/'pending'/f'{case}_{offset:04d}_{os.getpid()}'
                temporary.mkdir(parents=True,exist_ok=False)
                attempts,seeds,noise=batch(task,offset);began=time.perf_counter()
                with torch.no_grad():outputs,xyz,numbers,metadata=engine.generate(task,attempts,seeds,noise)
                seconds=time.perf_counter()-began
                with np.load(TRAJECTORIES/'arms/zero/sampling_trace'/f'{case}_{offset:04d}.npz') as old:
                    assert np.array_equal(xyz[:,:nc],old['xyz'][:,:nc])
                    assert np.array_equal(numbers[:,:nc],old['atomic_numbers'][:,:nc])
                assert metadata['noise_draws']==502 and metadata['core_max_normalized_error']==0
                traces=engine.hook.pending_traces;events=engine.hook.pending_events
                assert len(traces)==500 and len(events)==1000
                np.savez_compressed(temporary/'sampling_trace.npz',xyz=xyz,atomic_numbers=numbers,seeds=seeds)
                arrays={k:np.stack([t[k] for t in traces]) for k in ('predicted_xyz','predicted_features','delta_x0_coordinates','delta_x0_features','posterior_weight','coordinates_before_geometry','effective_h')}
                np.savez_compressed(temporary/'guidance_trace.npz',steps=[t['step'] for t in traces],seeds=seeds,**arrays)
                with (temporary/'events.jsonl').open('w') as f:
                    for e in events:f.write(json.dumps(e)+'\n')
                batch_rows=[]
                for i,(mol,error) in enumerate(outputs):
                    base=baseline[case,attempts[i]]
                    assert seeds[i]==base['seed'] and metadata['noise_tape_sha256']==base['noise_tape_sha256']
                    saved=temporary/f'{attempts[i]:04d}';saved.mkdir()
                    row=dict(arm=BEST,case_id=case,attempt=attempts[i],seed=seeds[i],reused=False,
                             path=str(final/f'{attempts[i]:04d}'),batch_path=str(final),
                             reconstructed=mol is not None,reconstruction_error=error,raw_audit=audit(mol,task),**metadata)
                    if mol is not None:
                        opts=Chem.PropertyPickleOptions.AllProps|Chem.PropertyPickleOptions.CoordsAsDouble
                        (saved/'raw.rdmol').write_bytes(mol.ToBinary(opts));row['raw_state_sha256']=sha(saved/'raw.rdmol')
                        with Chem.SDWriter(str(saved/'raw.sdf')) as f:f.SetKekulize(False);f.write(Chem.Mol(mol))
                    save(saved/'attempt.json',row);batch_rows.append(row)
                save(temporary/'records.json',batch_rows)
                save(temporary/'commit.json',dict(case_id=case,offset=offset,seeds=seeds,seconds=seconds,
                     files={str(p.relative_to(temporary)):sha(p) for p in temporary.rglob('*') if p.is_file()}))
                final.parent.mkdir(parents=True,exist_ok=True);temporary.rename(final)
                rows.extend(batch_rows);committed.update((r['case_id'],r['attempt']) for r in batch_rows)
                timing['batches'].append(dict(case_id=case,offset=offset,seconds=seconds))
                save(private/'records.json',rows);save(private/'timing.json',timing)
            update('scoring',active_case=case);began=time.perf_counter()
            score_case(private,rows,task,required,update)
            timing['scoring_seconds']+=time.perf_counter()-began;save(private/'timing.json',timing)
        assert len(rows)==3000 and all(not r['raw_audit']['strict_success'] or 'raw_score' in r for r in rows)
        integrity(manifest);update('complete',active_case=None,active_offset=None,completed_utc=now())
    except BaseException as error:
        save(private/'records.json',rows);update('failed',error=repr(error));raise
    finally:
        if engine is not None:engine.hook.close()



def finalize(run,manifest,records,update):
    integrity(manifest);assert_clean(records)
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
                   coordinate_x0_max_A=0.,feature_x0_max=0.) for a in SETTINGS}
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
            assert np.all(np.linalg.norm(dh,axis=-1).max(axis=(1,2))<=expected+1e-7)
            assert np.linalg.norm(dx,axis=-1).max()<=.02+1e-7
            inactive=np.array([s>=100 or not config['geometry_weight'] for s in steps])
            assert np.array_equal(dx[inactive],before[inactive])
            count['coordinate_x0_max_A']=max(count['coordinate_x0_max_A'],float(np.linalg.norm(dx,axis=-1).max()))
            count['feature_x0_max']=max(count['feature_x0_max'],float(np.linalg.norm(dh,axis=-1).max()))
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
    assert counts['cnn_element_late_taper']['geometry_applied']==0
    assert counts['cnn_local_geometry_guard']['geometry_applied']>0
    update('accounting');accounting=write_accounting(run,records,tasks,read(run/'references.json'))
    save(run/'mechanism.json',counts);integrity(manifest)
    save(run/'audit.json',dict(passed=True,cases=30,target_groups=7,attempts_per_case=100,new_outputs=6000,reused_outputs=6000,
        total_outputs=12000,core_fixed=True,paired_seeds_and_all_noise=True,native_final_reconstruction=True,
        gradient_events_verified=event_count,guidance_batch_steps_verified=1500000,committed_batches_verified=3000,
        all_failures_retained=True,controls_unchanged_after_clean_projection=True,accounting_cross_checked=True,
        development_records_and_selector_clean=True,external_evaluation_unblinded=False,
        records_sha256=sha(run/'records.json'),accounting_sha256=sha(run/'accounting.json')))
    save(run/'selection_freeze.json',dict(created_utc=now(),selected_arm=accounting['selection']['selected_arm'],
         manifest_sha256=sha(run/'manifest.json'),selection_sha256=sha(run/'selection.json'),external_evaluation_unblinded=False))
    update('complete',completed_utc=now(),result_file=str(run/'accounting.txt'))


def coordinate(run,finalize_only=False):
    lock=(run/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(run/'manifest.json');integrity(manifest);assert hardware()==manifest['hardware']
    state=read(run/'status.json')
    if state['stage']=='complete':print('Already complete.',flush=True);return
    state.pop('error',None)
    def update(stage,**values):
        state.update(stage=stage,pid=os.getpid(),updated_utc=now(),**values);save(run/'status.json',state);print(json.dumps(state),flush=True)
    try:
        if finalize_only:
            finalize(run,manifest,read(run/'records.json'),update);return
        # Refuse to duplicate a worker left alive by a previous coordinator.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
            try:args=(proc/'cmdline').read_bytes().split(b'\0')
            except OSError:continue
            if str(SCRIPT).encode() in args and b'--worker' in args and str(run).encode() in args:
                raise RuntimeError(f'Existing CNN ablation worker PID {proc.name}; coordinator will not duplicate it')
        processes=[];launch=[]
        for w in manifest['workers']:
            private=run/'workers'/f"gpu{w['gpu']}";private.mkdir(parents=True,exist_ok=True)
            done=(private/'status.json').exists() and read(private/'status.json')['stage']=='complete'
            command=[str(PYTHON),'-u',str(SCRIPT),'--run',str(run),'--worker',w['gpu']]
            process=None
            if not done:
                with (private/'worker.log').open('a') as f:
                    process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=w['gpu']),
                                             stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(process);launch.append(dict(gpu=w['gpu'],pid=process.pid if process else None,command=command))
        save(run/'launch.json',dict(pid=os.getpid(),launched_utc=now(),workers=launch))
        reused=read(run/'reused_records.json');cached={}
        while True:
            snapshots=[];states=[];changed=False
            for w,p in zip(manifest['workers'],processes):
                private=run/'workers'/f"gpu{w['gpu']}";path=private/'records.json'
                if path.exists():
                    stamp=path.stat().st_mtime_ns
                    if w['gpu'] not in cached or cached[w['gpu']][0]!=stamp:
                        cached[w['gpu']]=(stamp,read(path));changed=True
                    snapshots.extend(cached[w['gpu']][1])
                status=read(private/'status.json') if (private/'status.json').exists() else dict(stage='starting')
                states.append(dict(status,returncode=p.poll() if p else 0))
            if changed:
                merged=reused+snapshots;assert len(merged)==len({key(r) for r in merged})<=12000;save(run/'records.json',merged)
            update('running',generated_outputs=len(snapshots),workers=states)
            if all(p is None or p.poll() is not None for p in processes):break
            time.sleep(15)
        assert all(p is None or p.returncode==0 for p in processes),('worker failed',states)
        records=reused+[r for w in manifest['workers'] for r in read(run/'workers'/f"gpu{w['gpu']}"/'records.json')]
        assert len(records)==12000;save(run/'records.json',records);update('auditing',generated_outputs=6000)
        finalize(run,manifest,records,update)
    except BaseException as error:update('failed',error=repr(error));raise



def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(RUN));p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--worker',choices=('0','1'));p.add_argument('--finalize-only',action='store_true')
    a=p.parse_args();run=Path(a.run).resolve();RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4)
    if a.prepare_only:prepare(run)
    elif a.worker is not None:worker(run,a.worker)
    else:coordinate(run,a.finalize_only)


if __name__=='__main__':main()
