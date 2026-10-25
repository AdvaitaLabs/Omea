"""Finalize existing outputs; repair only the reverse-feature event field name.

The original runner and execution manifest remain frozen. No sampling or
scoring is performed by this entry point. Audit logic otherwise matches the
recovery runner exactly; provenance is in execution_manifest.finalize_v1.json.
"""
import argparse
import fcntl
import json
import os
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, RDLogger
from resume_selected_guidance_full_case_shards import (
    ROOT, SCREEN, SOURCE, TRAJECTORIES, BATCH, ATTEMPTS, ARMS, BEST,
    read, key, sha, save, now, hardware, integrity, validate_core_metadata,
    audit, summarize, write_accounting)

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
                feature_x0_max='actual_x0_feature_max_change',feature_reverse_mean_max='actual_reverse_mean_feature_max_change')
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='direct_guidance_selected_x0p02_h0p6_scale30_v1')
    args=parser.parse_args()
    RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4)
    run=(ROOT/'results'/args.run).resolve()
    lock=(run/'coordinator.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest_path=run/'execution_manifest.finalize_v1.json'
    manifest=read(manifest_path)
    integrity(manifest);assert hardware()==manifest['hardware']
    state=read(run/'status.json')
    if state['stage']=='complete':
        assert read(run/'audit.json')['passed']
        print('Already complete.',flush=True);return
    state.pop('error',None)
    state['execution_manifest']=str(manifest_path)
    def update(stage,**values):
        state.update(stage=stage,pid=os.getpid(),updated_utc=now(),**values)
        save(run/'status.json',state);print(json.dumps(state),flush=True)
    try:
        update('auditing',generated_outputs=2000)
        audit_and_account(run,manifest,read(run/'records.json'),update)
    except BaseException as error:
        update('failed',error=repr(error));raise


if __name__=='__main__':main()
