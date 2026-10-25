"""Frozen direct continuous-element experiment with at least 100 outputs per arm."""
import os
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from tasks import ROOT,audit,write_sdf
from autofragdiff_direct_hook import StrengthEngine,make_noise
from evaluate_direct_guidance import score_arm,summarize
from local_regeneration_graph import check_graph_classifier

SOURCES=['run_direct_guidance_experiment.py','autofragdiff_direct_hook.py','check_direct_guidance.py',
         'gnina_soft_element_guidance.py','gnina_soft_element_client.py','gnina_soft_element_worker.py',
         'check_soft_element_encoding_v2.py','evaluate_direct_guidance.py','autofragdiff_strength_hook.py',
         'autofragdiff_local_regeneration.py','local_regeneration_graph.py','element_guidance_objective.py',
         'element_physics_guard.py','fixed_core_reconstruction.py','gnina_gradient_client.py',
         'gnina_gradient_worker.py','gnina_element_client.py','gnina_element_worker.py',
         'gnina_guidance.py','score_extra_results.py','restore_scoring_serialization.py','tasks.py','reanalyze_geometry_stereo.py']


def defaults():
    timing=dict(last_fraction=1.,interval=1,schedule='flat',temperature=.2)
    return [dict(name='zero',lambda_x=0.,lambda_h=0.,**timing),
            dict(name='coord_only',lambda_x=.005,lambda_h=0.,**timing)] + [
            dict(name='joint_'+str(h).replace('.','p'),lambda_x=.005,lambda_h=h,**timing)
            for h in [.1,.4,1.6]]


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)


def audit_direct_events(run,configurations,tasks,attempts_per_case):
    from collections import Counter
    taskmap={t['case_id']:t for t in tasks}
    result={}
    for config in configurations:
        arm=run/'arms'/config['name']
        path=arm/'events.jsonl'
        events=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        expected_events=500*attempts_per_case*len(tasks) if config['lambda_x'] or config['lambda_h'] else 0
        assert len(events)==expected_events
        trajectory_counts=Counter((r['case_id'],r['attempt']) for r in events)
        assert all(n==500 for n in trajectory_counts.values())
        assert all(r['status']=='guided' and r['joint_model_forward_calls']==r['joint_backward_calls']==1 for r in events)
        assert all(r['scored_element_substitution_candidates']==0 and r['diagnostics']['all_R_features_participate'] for r in events)
        assert all(all(i>=taskmap[r['case_id']]['fixed_heavy_atoms'] for i in r['actual_x0_feature_active_sites']) for r in events)
        sites=Counter(len(r['actual_x0_feature_active_sites']) for r in events)
        for trace_path in sorted((arm/'guidance_traces').glob('*.npz')):
            with np.load(trace_path) as trace:
                case=trace_path.name.rsplit('_',2)[0]
                nc=taskmap[case]['fixed_heavy_atoms']
                for name in ['delta_x0_coordinates','delta_x0_features']:
                    assert np.isfinite(trace[name]).all() and np.count_nonzero(trace[name][:,:nc])==0
        result[config['name']]=dict(score_gradient_events=len(events),
                                   feature_active_site_counts=dict(sites),
                                   graph_free_role_fallback_events=sum(r['diagnostics']['encoding_context']=='graph_free_role_mixture' for r in events))
    return dict(passed=True,arms=result,positive_gain_gates=0,substitution_score_candidates=0,
                note='Every step scores all R inputs; a zero derivative from the score is allowed, not replaced with an artificial update.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--checks',required=True)
    p.add_argument('--config');p.add_argument('--seed',type=int,default=2026101021)
    p.add_argument('--tasks');p.add_argument('--references')
    p.add_argument('--attempts-per-case',type=int,default=20);p.add_argument('--batch-size',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(4);wallstart=time.perf_counter()
    assert args.attempts_per_case>=20, 'Direct-guidance comparisons require at least 20 attempts per case; use check_direct_guidance.py for preflight checks.'
    assert args.batch_size>0 and args.attempts_per_case%args.batch_size==0
    source_tasks=Path(args.tasks).resolve() if args.tasks else ROOT/'results/guidance_dev_v1/tasks.json'
    tasks=json.loads(source_tasks.read_text())
    assert tasks and len({t['case_id'] for t in tasks})==len(tasks)
    expected=args.attempts_per_case*len(tasks)
    assert expected>=100, 'Direct-guidance comparisons require at least 100 outputs per arm; smaller runs belong in preflight checks.'
    target_groups=len({t.get('target_group',t['pocket_id']) for t in tasks})
    run=ROOT/'results'/args.run;run.mkdir(parents=True,exist_ok=False)
    checks_path=Path(args.checks);checks=json.loads(checks_path.read_text());assert checks['passed']
    for n,h in checks['source_sha256'].items():assert sha(ROOT/'scripts'/n)==h,n
    configurations=json.loads(Path(args.config).read_text()) if args.config else defaults()
    assert configurations[0]['name']=='zero' and configurations[0]['lambda_x']==configurations[0]['lambda_h']==0
    assert len({c['name'] for c in configurations})==len(configurations)
    for c in configurations:
        assert c['lambda_x']>=0 and c['lambda_h']>=0 and 0<c['last_fraction']<=1 and c['interval']>0
        assert c['schedule']=='flat' and c['temperature']>0
        assert c['interval']==1 and c['last_fraction']==1.
    shutil.copy2(source_tasks,run/'tasks.json');shutil.copy2(checks_path,run/'preflight_checks.json')
    save(run/'configurations.json',configurations)
    reference_path=Path(args.references).resolve() if args.references else ROOT/'results/persistent_element_replication_v1/judge_results_zero.json'
    references={r['case_id']:r for r in json.loads(reference_path.read_text())['rows'] if r['model']=='reference'}
    assert all(t['case_id'] in references and references[t['case_id']]['status']=='scored' for t in tasks)
    schema_path=ROOT/'results/selection_rules_v3/geometry_schema.json'
    required=set(json.loads(schema_path.read_text())['required_checks']);assert len(required)==22
    checkpoint=ROOT/'checkpoints/autofragdiff_pocket-gvp.ckpt'
    weights=sorted((ROOT/'vendor/gnina-torch-guidance/gninatorch/weights').glob('*.pt'))
    inputs=[*weights,source_tasks,checks_path,reference_path,schema_path,checkpoint,
            ROOT/'vendor/AutoFragDiff/src/edm.py',ROOT/'vendor/AutoFragDiff/src/dynamics_gvp.py']
    if args.config:inputs.append(Path(args.config))
    for task in tasks:inputs.extend(Path(task[k]) for k in ['core_path','protein_path','reference_path'])
    manifest=dict(created_utc=now(),seed_root=args.seed,cases=len(tasks),target_groups=target_groups,attempts_per_case=args.attempts_per_case,
                  outputs_per_arm=expected,configurations=configurations,batch_size=args.batch_size,timesteps=500,
                  source_sha256={n:sha(ROOT/'scripts'/n) for n in SOURCES},input_sha256={str(f):sha(f) for f in inputs},
                  checkpoint_sha256=sha(checkpoint),vendor_commit=subprocess.check_output(['git','-C',str(ROOT/'vendor/AutoFragDiff'),'rev-parse','HEAD'],text=True).strip(),
                  protocol=dict(sampling='one native-prior diffusion trajectory and one decoded output per seed per configuration',
                                pairing='independent CPU generator per trajectory; replay exact same 502 Gaussian draws across configurations; batch width independent',
                                strength='delta_x0 = lambda * clip_per_atom(gradient / fixed_gradient_scale, unit_norm); gradient clipped BEFORE strength',
                                coordinate_gradient_scale=.2,feature_gradient_scale=.05,
                                coordinate_strength_units='maximum per-atom x0 displacement A per guidance event before posterior weighting',
                                feature_strength_units='maximum per-atom unnormalized x0 type-feature L2 per event before posterior weighting; every R feature participates',
                                posterior='native EDM posterior; modify clean prediction through guided epsilon; log actual posterior mean correction',
                                coordinate_objective='GNINA continuous element-density affinity score, fixed supplied-core-centroid frame',
                                element_objective='same single GNINA score differentiated through h[*,10] -> softmax -> element-specific densities -> CNN; simultaneous gradient over all R atoms',
                                derivative='direct derivatives of predicted clean x0/h0; native posterior receives the correction; no denoiser Jacobian or gradient through discrete bond reconstruction',
                                chemical_roles='detached local current-state roles; graph-free role mixture on failed reconstruction; no gradient gate',
                                feature_constraints='fixed core mask and continuous update clipping; no legal-substitution, positive-gain or strain gate; final chemical and geometry failures all count',
                                finalization='original decoder; no terminal edits or forced types',
                                computational_serialization='RDKit AllProps plus CoordsAsDouble, saved before SDF export; SDF writer receives a copy',
                                molecule_reconstruction='same supplied-core-only graph restoration for all arms; no R/crossing repairs',
                                reference_R_information='atom count only, no original R element/coordinates/bonds in generator',
                                evaluation='raw generated heavy-atom pose; identical H-only preparation; complete 22-check PoseBusters; no post-generation GNINA optimization',
                                comparisons='all predeclared attempts per arm, paired to zero; no per-seed best-of-K, no fallback to zero for failed guided outputs',
                                reporting='raw quality, scored-pair proxy deltas, true chemical graph changes and chemically changed paired dual wins separately'),
                  limitations=['a frozen cohort; neither a training-independent nor a homology-held-out claim is established',
                               'continuous-density and detached-role relaxation is not a derivative through discrete chemical graphs, exact noisy-state classifier guidance, or CFG',
                               'scores are proxies, not measured DDG','changing strength can reduce validity; every failure counts'])
    save(run/'manifest.json',manifest);snap=run/'source';snap.mkdir()
    for n in SOURCES:shutil.copy2(ROOT/'scripts'/n,snap/n)
    state=dict(pid=os.getpid(),started_utc=now(),stage='preparing_noise',total_outputs=expected*len(configurations),
               generated_outputs=0,completed_arms=[])
    records=[];timing={};noise_records=[]
    def update(stage,**kwargs):
        state.update(stage=stage,updated_utc=now(),**kwargs);save(run/'status.json',state)
        print(json.dumps(state),flush=True)
    def integrity():
        for n,h in manifest['source_sha256'].items():assert sha(ROOT/'scripts'/n)==h,n
        for path,h in manifest['input_sha256'].items():assert sha(path)==h,path
    engine=None
    try:
        update('preparing_noise');integrity();noise_root=run/'noise';noise_root.mkdir()
        for task in tasks:
            nc=task['fixed_heavy_atoms'];nr=task['rgroup_heavy_atoms']
            for offset in range(0,args.attempts_per_case,args.batch_size):
                attempts=list(range(offset,offset+args.batch_size))
                seeds=[(args.seed+int(hashlib.sha256(f"{task['case_id']}/{i}".encode()).hexdigest()[:8],16))%(2**32) for i in attempts]
                noise=make_noise(seeds,nc+nr);path=noise_root/f"{task['case_id']}_{offset:04d}.npz"
                np.savez_compressed(path,noise=noise.numpy(),seeds=seeds,attempts=attempts)
                noise_records.append(dict(case_id=task['case_id'],offset=offset,path=str(path),file_sha256=sha(path),
                                          tape_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest(),seeds=seeds))
        save(run/'noise_manifest.json',noise_records)
        assert len({s for r in noise_records for s in r['seeds']})==expected
        update('loading_checkpoint');start=time.perf_counter();engine=StrengthEngine()
        assert engine.edm.T==500;timing['checkpoint_load_seconds']=time.perf_counter()-start
        for configuration in configurations:
            integrity();arm=configuration['name'];out=run/'arms'/arm;engine.hook.configure(configuration,out)
            batches=[];armstart=time.perf_counter()
            for task in tasks:
                for offset in range(0,args.attempts_per_case,args.batch_size):
                    source=next(n for n in noise_records if n['case_id']==task['case_id'] and n['offset']==offset)
                    assert sha(source['path'])==source['file_sha256'];data=np.load(source['path'])
                    noise=torch.from_numpy(data['noise'].copy());seeds=list(map(int,data['seeds']));attempts=list(map(int,data['attempts']))
                    update('generating',active_arm=arm,active_case=task['case_id'],active_offset=offset)
                    torch.cuda.synchronize();start=time.perf_counter()
                    with torch.no_grad():outputs,xyz,numbers,metadata=engine.generate(task,attempts,seeds,noise)
                    torch.cuda.synchronize();seconds=time.perf_counter()-start
                    assert metadata['noise_tape_sha256']==source['tape_sha256'] and metadata['noise_draws']==502
                    trace=out/'sampling_trace';trace.mkdir(exist_ok=True)
                    np.savez_compressed(trace/f"{task['case_id']}_{offset:04d}.npz",xyz=xyz,atomic_numbers=numbers,seeds=seeds)
                    batches.append(dict(case_id=task['case_id'],offset=offset,seconds=seconds,**metadata))
                    for i,(mol,error) in enumerate(outputs):
                        path=out/task['case_id']/f'{attempts[i]:04d}';path.mkdir(parents=True)
                        row=dict(arm=arm,case_id=task['case_id'],attempt=attempts[i],seed=seeds[i],path=str(path),
                                 noise_tape_sha256=metadata['noise_tape_sha256'],noise_draws=502,
                                 reconstructed=mol is not None,reconstruction_error=error,raw_audit=audit(mol,task))
                        if mol is not None:
                            binary=path/'raw.rdmol'
                            options=Chem.PropertyPickleOptions.AllProps|Chem.PropertyPickleOptions.CoordsAsDouble
                            binary.write_bytes(mol.ToBinary(options));row['raw_state_sha256']=sha(binary)
                            write_sdf(path/'raw.sdf',[Chem.Mol(mol)])
                        save(path/'attempt.json',row);records.append(row)
                    state['generated_outputs']+=args.batch_size;save(run/'records.json',records)
                    update('generating',active_arm_generated=sum(r['arm']==arm for r in records),
                           active_arm_strict=sum(r['arm']==arm and r['raw_audit']['strict_success'] for r in records))
            timing[arm]=dict(generation_seconds=sum(b['seconds'] for b in batches),generation_stage_wall_seconds=time.perf_counter()-armstart,batches=batches)
            update('scoring',active_arm=arm,completed_scores=0);start=time.perf_counter()
            score_arm(run,arm,records,tasks,required,update)
            timing[arm]['score_wall_seconds']=time.perf_counter()-start
            state['completed_arms'].append(arm);save(run/'timing.json',timing)
            result=summarize(run,configurations,records,tasks,references)
            update('arm_complete',active_arm=arm)
            print(json.dumps(dict(arm=arm,metrics=result['arms'][arm]['metrics'],mechanism=result['arms'][arm]['mechanism'])),flush=True)
        integrity();update('auditing')
        assert len(records)==expected*len(configurations)
        base={(r['case_id'],r['attempt']):r for r in records if r['arm']=='zero'}
        taskmap={t['case_id']:t for t in tasks}
        for r in records:
            b=base[r['case_id'],r['attempt']]
            assert r['seed']==b['seed'] and r['noise_tape_sha256']==b['noise_tape_sha256'] and r['noise_draws']==b['noise_draws']==502
            if r['reconstructed']:
                path=Path(r['path'])/'raw.rdmol';assert sha(path)==r['raw_state_sha256']
                check=audit(Chem.Mol(path.read_bytes()),taskmap[r['case_id']]);assert check==r['raw_audit']
        for cfg in configurations:
            for source in noise_records:
                name=f"{source['case_id']}_{source['offset']:04d}.npz"
                a=np.load(run/'arms'/'zero'/'sampling_trace'/name);b=np.load(run/'arms'/cfg['name']/'sampling_trace'/name)
                nc=taskmap[source['case_id']]['fixed_heavy_atoms']
                assert np.array_equal(a['atomic_numbers'][:,:nc],b['atomic_numbers'][:,:nc])
                assert np.array_equal(a['xyz'][:,:nc],b['xyz'][:,:nc])
        for source in noise_records:assert sha(source['path'])==source['file_sha256']
        site_audit=audit_direct_events(run,configurations,tasks,args.attempts_per_case)
        save(run/'direct_update_audit.json',site_audit)
        save(run/'audit.json',dict(passed=True,direct_joint_updates_verified=True,all_attempts_retained=len(records),outputs_per_seed_per_configuration=1,
                                 all_arms_identical_seed_and_502_noise_draws=True,core_coordinates_elements_unchanged=True,
                                 checkpoint_frozen=True,input_source_noise_integrity=True,trajectory_branches=0,
                                 terminal_edits=0,post_generation_pose_optimization=0,best_of_K_selection=0,
                                 graph_classifier=check_graph_classifier()))
        timing['total_wall_seconds']=time.perf_counter()-wallstart;save(run/'timing.json',timing)
        update('complete',active_arm=None,completed_utc=now(),result_file=str(run/'comparison.json'))
    except Exception as exc:
        save(run/'records.json',records);update('failed',error=repr(exc));raise
    finally:
        if engine is not None:engine.hook.close()


if __name__=='__main__':main()

