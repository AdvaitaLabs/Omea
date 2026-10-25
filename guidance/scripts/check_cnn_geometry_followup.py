"""Technical gates for C/D. All technical molecules excluded from efficacy counts."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse
from collections import Counter
import copy
import types
import numpy as np
import torch
from rdkit import Chem,RDLogger
from tasks import ROOT,read_mol,audit
from resume_selected_guidance_full_case_shards import read,save,sha,now,batch,TRAJECTORIES,hardware
from cnn_geometry_followup_settings import SOURCE,PARENT,CHECKS,BASELINE,SETTINGS,NEW_SOURCES
from cnn_geometry_projection import make_terms,components,project,values
from autofragdiff_cnn_geometry_hook import GeometryEngine,GeometryHook
from autofragdiff_cnn_clean_hook import CleanHook
from autofragdiff_strength_hook import scaled_gradient
from score_extra_results import score_one
from score_cnn_clean import assert_clean
from summarize_cnn_geometry_followup import choose

FILES=(*NEW_SOURCES,'cnn_clean_settings.py','cnn_local_geometry.py','autofragdiff_cnn_clean_hook.py',
       'autofragdiff_direct_hook.py','gnina_soft_element_guidance.py','gnina_soft_element_worker.py',
       'run_cnn_clean_ablations.py','score_extra_results.py')


def representatives():
    groups=set();result=[]
    for task in read(PARENT/'tasks.json'):
        if task['target_group'] not in groups:groups.add(task['target_group']);result.append(task)
    assert len(result)==7
    return result


def cpu_checks():
    rng=np.random.default_rng(20261024);rows=[];config=SETTINGS['cnn_geometry_projected']
    for task in representatives():
        mol=read_mol(task['reference_path'])
        mol=Chem.RenumberAtoms(mol,task['fixed_reference_indices']+task['removed_reference_indices'])
        nc=task['fixed_heavy_atoms'];xyz=mol.GetConformer().GetPositions().copy()
        xyz[nc:]+=rng.normal(size=xyz[nc:].shape)*.35
        terms=make_terms(mol,nc,config)
        x=torch.tensor(xyz,dtype=torch.float64,requires_grad=True);losses=components(x,nc,terms)
        direction=rng.normal(size=xyz.shape);direction[:nc]=0;direction/=np.linalg.norm(direction)
        eps=1e-5;errors=[]
        for k in range(3):
            g=torch.autograd.grad(losses[k],x,retain_graph=k<2)[0].detach().numpy()
            assert np.isfinite(g).all() and not np.count_nonzero(g[:nc])
            finite=(values(xyz+eps*direction,nc,terms)[k]-values(xyz-eps*direction,nc,terms)[k])/(2*eps)
            analytic=float((g*direction).sum());error=abs(finite-analytic)
            assert error<2e-5*(1+abs(finite)),(task['case_id'],k,error)
            errors.append(error)
        proposed,_=scaled_gradient(rng.normal(size=xyz.shape),.02,.2,nc)
        result,info=project(xyz,proposed,nc,mol,config)
        assert np.linalg.norm(result,axis=1).max()<=.02+1e-12 and not np.count_nonzero(result[:nc])
        assert np.all(values(xyz+result,nc,terms)<=values(xyz,nc,terms)+np.array(info['tolerance']))
        fallback,meta=project(xyz,proposed,nc,None,config)
        assert np.array_equal(fallback,proposed) and meta['status']=='no_valid_context'
        # Rigid rotation must not change penalty or geometric behavior.
        rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
        turned,turned_info=project(xyz@rotation,proposed@rotation,nc,mol,config)
        assert np.allclose(values(xyz@rotation,nc,terms),values(xyz,nc,terms),atol=1e-10)
        assert np.allclose(turned,result@rotation,atol=1e-8)
        rows.append(dict(case_id=task['case_id'],derivative_errors=errors,projection=info,
                         fixed_core_zero=True,rotation_equivariance=True))
    sample={BASELINE['name']:dict(attempts=3000,valid_new=1346,cnn_success=908),
            **{arm:dict(attempts=3000,valid_new=1346,cnn_success=909) for arm in SETTINGS}}
    assert choose(sample)['selected_arm']==sorted(SETTINGS)[0]
    contaminated=copy.deepcopy(sample);contaminated[BASELINE['name']]['vina_kcal_mol']=-9.
    try:choose(contaminated)
    except AssertionError:pass
    else:raise AssertionError('Vina accepted by selector')
    CHECKS.mkdir(parents=True,exist_ok=True)
    save(CHECKS/'cpu_checks.json',dict(passed=True,rows=rows,selection_rejects_external_score=True,
        source_sha256={name:sha(ROOT/'scripts'/name) for name in FILES},completed_utc=now()))
    print('CPU derivative, geometry, rotation, core and selector checks passed.',flush=True)


def paired_step(self,*args,**kwargs):
    if self.configuration['geometry_projection'] or not (self.configuration['lambda_x'] or self.configuration['lambda_h']):
        return GeometryHook.step(self,*args,**kwargs)
    phase=[0];cached={};positions={}
    forward=self.edm.dynamics.forward;noise=self.edm.sample_combined_position_features_noise;gradient=self.client.joint_gradient
    def shared(name,function):
        cached[name]=[];positions[name]=0
        def call(*a,**kw):
            if phase[0]==0:
                value=function(*a,**kw)
                signature=(a[1].copy(),a[2].copy()) if name=='gradient' else None
                cached[name].append((value,signature));return value
            i=positions[name];positions[name]+=1;value,signature=cached[name][i]
            if signature is not None:assert np.array_equal(a[1],signature[0]) and np.array_equal(a[2],signature[1])
            return value
        return call
    self.edm.dynamics.forward=shared('forward',forward)
    self.edm.sample_combined_position_features_noise=shared('noise',noise)
    self.client.joint_gradient=shared('gradient',gradient)
    ne,nt=len(self.pending_events),len(self.pending_traces)
    try:
        original=CleanHook.step(self,*args,**kwargs)
        del self.pending_events[ne:];del self.pending_traces[nt:]
        phase[0]=1
        current=GeometryHook.step(self,*args,**kwargs)
        assert all(positions[k]==len(v) for k,v in cached.items())
        assert torch.equal(original,current),('equation mismatch',float(torch.max(torch.abs(original-current))))
        self.paired_counts[self.configuration['name']]+=1
        return current
    finally:
        self.edm.dynamics.forward=forward;self.edm.sample_combined_position_features_noise=noise;self.client.joint_gradient=gradient


def gpu_checks(gpu):
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    out=CHECKS/f'gpu{gpu}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'checks.json').exists()
    hashes={name:sha(ROOT/'scripts'/name) for name in FILES}
    cpu=read(CHECKS/'cpu_checks.json');assert cpu['passed'] and cpu['source_sha256']==hashes
    config=list(SETTINGS.values())[gpu];tasks=representatives()
    controls={(r['case_id'],r['attempt']):r for r in read(SOURCE/'records.json') if r['arm']==BASELINE['name']}
    zero={(r['case_id'],r['attempt']):r for r in read(SOURCE/'records.json') if r['arm']=='zero'}
    engine=None;state=dict(gpu=gpu,pid=os.getpid(),started_utc=now());technical=0;replays=[];ablation=[]
    def update(stage,**extra):
        state.update(stage=stage,updated_utc=now(),**extra);save(out/'status.json',state);print(state,flush=True)
    def generate(task,cfg,label):
        nonlocal technical
        target=out/label;engine.hook.configure(cfg,target)
        attempts,seeds,noise=batch(task,0)
        with torch.no_grad():outputs,xyz,numbers,metadata=engine.generate(task,attempts,seeds,noise)
        technical+=2;nc=task['fixed_heavy_atoms']
        assert metadata['noise_draws']==502 and metadata['core_max_normalized_error']==0
        with np.load(TRAJECTORIES/'arms/zero/sampling_trace'/f"{task['case_id']}_0000.npz") as saved:
            assert np.array_equal(xyz[:,:nc],saved['xyz'][:,:nc]) and np.array_equal(numbers[:,:nc],saved['atomic_numbers'][:,:nc])
        for i,seed in enumerate(seeds):
            assert seed==zero[task['case_id'],i]['seed'] and metadata['noise_tape_sha256']==zero[task['case_id'],i]['noise_tape_sha256']
        np.savez_compressed(target/'trajectory.npz',xyz=xyz,atomic_numbers=numbers)
        if engine.hook.pending_traces:
            np.savez_compressed(target/'updates.npz',dx=np.stack([t['delta_x0_coordinates'] for t in engine.hook.pending_traces]),
                dh=np.stack([t['delta_x0_features'] for t in engine.hook.pending_traces]))
            save(target/'events.json',engine.hook.pending_events)
        return outputs,xyz,numbers,metadata
    try:
        update('loading');engine=GeometryEngine();engine.hook.paired_counts=Counter()
        engine.hook.step=types.MethodType(paired_step,engine.hook)
        outputs,xyz,numbers,meta=generate(tasks[0],dict(BASELINE,name='zero',lambda_x=0.,lambda_h=0.,taper_end=0.,geometry_weight=0.),'zero')
        with np.load(TRAJECTORIES/'arms/zero/sampling_trace'/f"{tasks[0]['case_id']}_0000.npz") as old:
            assert np.array_equal(numbers,old['atomic_numbers']) and np.max(np.abs(xyz-old['xyz']))<.001
        assert not engine.hook.pending_events
        for task in tasks:
            update('baseline_equation_replay',completed_cases=len(replays),active_case=task['case_id'])
            outputs,xyz,numbers,meta=generate(task,BASELINE,'baseline/'+task['case_id'])
            old=controls[task['case_id'],0]
            with np.load(__import__('pathlib').Path(old['batch_path'])/'sampling_trace.npz') as saved:
                error=float(np.max(np.abs(xyz-saved['xyz'])))
                elements=bool(np.array_equal(numbers,saved['atomic_numbers']))
            assert len(engine.hook.pending_traces)==500 and len(engine.hook.pending_events)==1000
            replays.append(dict(case_id=task['case_id'],historical_coordinate_error_A=error,historical_elements_equal=elements,
                historical_audits_equal=all(audit(m,task)==controls[task['case_id'],i]['raw_audit'] for i,(m,_) in enumerate(outputs)),
                exact_shared_oracle_step_equivalence=True))
        assert engine.hook.paired_counts[BASELINE['name']]==3500
        for i,task in enumerate((tasks[0],tasks[-1])):
            update('candidate_trajectory',configuration=config['name'],active_case=task['case_id'])
            outputs,xyz,numbers,meta=generate(task,config,f'candidate_{i}')
            events=engine.hook.pending_events;traces=engine.hook.pending_traces;nc=task['fixed_heavy_atoms']
            assert len(events)==1000 and len(traces)==500
            for trace in traces:
                dx,dh=trace['delta_x0_coordinates'],trace['delta_x0_features']
                assert np.isfinite(dx).all() and np.isfinite(dh).all()
                assert not np.count_nonzero(dx[:,:nc]) and not np.count_nonzero(dh[:,:nc])
                assert np.linalg.norm(dx.astype(np.float64),axis=-1).max()<=.02+1e-7
                assert np.linalg.norm(dh.astype(np.float64),axis=-1).max()<=.6+1e-7
                if trace['step']>=100:assert np.array_equal(dx,trace['coordinates_before_geometry'])
            counts=Counter()
            for event in events:
                assert_clean(event)
                assert event['joint_model_forward_calls']==event['joint_backward_calls']==1
                assert event['lambda_h_effective']==.6 and event['geometry_weight']==config['geometry_weight']
                p=event['projection'];counts[p['status']]+=1
                if config['geometry_projection'] and event['step']<100:
                    if p['status']!='no_valid_context':
                        assert np.all(np.array(p['stored_after'])<=np.array(p['before'])+np.array(p['tolerance']))
                else:assert p['status']=='inactive'
            if config['geometry_projection']:assert counts['accepted']>0 and counts['inactive']==800
            ablation.append(dict(case_id=task['case_id'],projection_counts=dict(counts),audits=[audit(m,task) for m,_ in outputs]))
            if i==0:
                judged=[]
                for j,(mol,_) in enumerate(outputs):
                    if audit(mol,task)['strict_success']:
                        s=score_one(mol,task['protein_path'],out/f'judge_candidate/{j}')
                        assert s['status']=='scored' and all(np.isfinite(s[k]) for k in ('vina_kcal_mol','cnn_affinity_pK','cnn_pose_score'))
                        judged.append({k:s[k] for k in ('status','vina_kcal_mol','cnn_affinity_pK')})
                assert judged
        assert technical==20 and hashes=={name:sha(ROOT/'scripts'/name) for name in FILES}
        save(out/'checks.json',dict(passed=True,gpu=gpu,configuration=config,hardware=hardware(),
            technical_outputs=technical,efficacy_outputs=0,zero_replay=True,baseline_replays=replays,
            baseline_equation_steps=engine.hook.paired_counts[BASELINE['name']],
            candidate_equation_steps=engine.hook.paired_counts[config['name']],candidate_checks=ablation,
            full_metric_scoring_smoke=judged,source_sha256=hashes,completed_utc=now(),
            repeatability_note='Historical GPU CNN replay can differ numerically; gate is exact equation parity under shared CNN gradients, model outputs and noise. Historical discrepancies are disclosed.',
            output_sha256={str(p):sha(p) for p in out.rglob('*') if p.is_file() and (p.suffix=='.npz' or p.name=='events.json')}))
        update('complete',passed=True)
    except BaseException as exc:update('failed',error=repr(exc));raise
    finally:
        if engine is not None:engine.hook.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu',action='store_true');p.add_argument('--gpu',type=int,choices=(0,1));a=p.parse_args()
    RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4)
    if a.cpu:cpu_checks()
    else:assert a.gpu is not None;gpu_checks(a.gpu)
