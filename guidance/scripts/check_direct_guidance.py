"""Full 500-step preflight for direct x0/h0 gradients and paired native noise."""
import os
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse
import hashlib
import json
import time
import types
from pathlib import Path
import numpy as np
import torch
from tasks import ROOT,audit
from autofragdiff_direct_hook import StrengthEngine,NoiseTape,make_noise,scaled_gradient
from run_direct_guidance_experiment import SOURCES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--encoding-checks',required=True)
    args=parser.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    began=time.perf_counter()
    checks=json.loads(Path(args.encoding_checks).read_text())
    assert checks['passed'] and checks['generator_10_element_bridge_implemented']
    for path,digest in checks['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    result=dict(source_sha256={n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in SOURCES},
                encoding_checks=str(Path(args.encoding_checks).resolve()),
                encoding_checks_sha256=hashlib.sha256(Path(args.encoding_checks).read_bytes()).hexdigest())
    gradient=np.array([[2.,3.,4.],[.01,-.02,.03],[1.,-2.,3.]])
    weak,_=scaled_gradient(gradient,.1,.05,1)
    strong,_=scaled_gradient(gradient,1.6,.05,1)
    assert np.allclose(strong,16*weak) and np.count_nonzero(weak[:1])==0
    result['strength_linearly_scales_post_clip_update']=True
    task=json.loads((ROOT/'results/guidance_dev_v1/tasks.json').read_text())[0]
    seeds=[119920,119921]
    nodes=task['fixed_heavy_atoms']+task['rgroup_heavy_atoms']
    noise=make_noise(seeds,nodes)
    assert torch.equal(noise[:,0],make_noise(seeds[:1],nodes)[:,0])
    assert not torch.equal(noise[:,0],noise[:,1])
    engine=StrengthEngine()
    try:
        inputs,center,core=engine.fresh_inputs(task,2)
        tape=NoiseTape(noise)
        def draw(_edm,n_samples,n_nodes,mask):
            return tape.draw(n_samples,n_nodes,mask)
        engine.edm.sample_combined_position_features_noise=types.MethodType(draw,engine.edm)
        with torch.no_grad():
            native_x,native_h,_=engine.edm.sample_chain_single_fragment(**inputs,keep_frames=1,lj_guidance=False)
        assert tape.position==502
        engine.edm.sample_combined_position_features_noise=engine.original_noise
        timing=dict(last_fraction=1.,interval=1,schedule='flat',temperature=.2)
        configs=[dict(name='zero',lambda_x=0.,lambda_h=0.,**timing),
                 dict(name='coord_only',lambda_x=.005,lambda_h=0.,**timing),
                 dict(name='joint_weak',lambda_x=.005,lambda_h=.1,**timing),
                 dict(name='joint_strong',lambda_x=.005,lambda_h=1.6,**timing)]
        result['arms']=[]
        for config in configs:
            path=out/config['name']
            engine.hook.configure(config,path)
            with torch.no_grad():
                outputs,xyz,numbers,metadata=engine.generate(task,[0,1],seeds,noise)
            assert metadata['noise_draws']==502
            if config['name']=='zero':
                difference=float(np.max(np.abs(xyz-(native_x.cpu().numpy()+center))))
                assert difference<1e-5
                expected=np.array([6,7,8,16,5,35,17,15,53,9])[native_h.argmax(-1).cpu().numpy()]
                assert np.array_equal(numbers,expected)
                zero_xyz,zero_numbers=xyz,numbers
                result.update(zero_native_max_coordinate_difference_A=difference,zero_native_elements_equal=True)
            nc=core.GetNumAtoms()
            assert np.array_equal(xyz[:,:nc],zero_xyz[:,:nc])
            assert np.array_equal(numbers[:,:nc],zero_numbers[:,:nc])
            np.savez_compressed(path/'sampling_trace.npz',xyz=xyz,atomic_numbers=numbers)
            events=[json.loads(s) for s in (path/'events.jsonl').read_text().splitlines()] if (path/'events.jsonl').exists() else []
            if config['name']!='zero':
                assert len(events)==1000
                for attempt in [0,1]:
                    assert sorted(e['step'] for e in events if e['attempt']==attempt)==list(range(500))
                assert all(e['status']=='guided' and e['diagnostics']['all_R_features_participate'] for e in events)
                assert all(e['joint_model_forward_calls']==e['joint_backward_calls']==1 for e in events)
                assert all(e['scored_element_substitution_candidates']==0 for e in events)
                assert any(e['context_reconstruction_error'] is not None for e in events)
                if config['lambda_h']:
                    assert any(len(e['actual_x0_feature_active_sites'])==task['rgroup_heavy_atoms'] for e in events)
                    assert max(e['actual_x0_feature_max_change'] for e in events)>0
                for trace_path in (path/'guidance_traces').glob('*.npz'):
                    with np.load(trace_path) as trace:
                        assert np.count_nonzero(trace['delta_x0_coordinates'][:,:nc])==0
                        assert np.count_nonzero(trace['delta_x0_features'][:,:nc])==0
            row=dict(configuration=config,metadata=metadata,score_gradient_events=len(events),
                     all_R_nonzero_feature_update_events=sum(len(e['actual_x0_feature_active_sites'])==task['rgroup_heavy_atoms'] for e in events),
                     graph_free_role_fallback_events=sum(e['diagnostics']['encoding_context']=='graph_free_role_mixture' for e in events),
                     max_feature_update=max((e['actual_x0_feature_max_change'] for e in events),default=0.),
                     attempts=[dict(error=error,audit=audit(mol,task)) for mol,error in outputs])
            result['arms'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='attempts'}),flush=True)
        assert len({a['metadata']['noise_tape_sha256'] for a in result['arms']})==1
        for name,digest in result['source_sha256'].items():
            assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==digest
        result.update(passed=True,same_502_noise_draws=True,all_500_steps_compute_all_R_input_gradients=True,
                      fixed_core_unchanged=True,scored_element_substitution_candidates=0,
                      wall_seconds=time.perf_counter()-began)
        (out/'checks.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(passed=True,wall_seconds=result['wall_seconds'])),flush=True)
    except Exception as exc:
        result.update(passed=False,error=repr(exc),wall_seconds=time.perf_counter()-began)
        (out/'checks.json').write_text(json.dumps(result,indent=2))
        raise
    finally:
        engine.hook.close()


if __name__=='__main__':
    main()
