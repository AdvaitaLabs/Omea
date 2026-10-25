"""Finish frozen CNN-only runs with float64 norm accumulation and unchanged 1e-7 tolerance.

Generation, saved vectors, scores, inclusion rules and selection remain frozen.
The original runner is not edited. This is an independently hashed audit amendment.
"""
import copy
import shutil
from run_cnn_clean_ablations import *

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
        audit_norm_accumulation_dtype='float64',unchanged_update_tolerance=1e-7,
        audit_execution_manifest=str(run/'execution_manifest.finalize_v1.json'),
        records_sha256=sha(run/'records.json'),accounting_sha256=sha(run/'accounting.json')))
    save(run/'selection_freeze.json',dict(created_utc=now(),selected_arm=accounting['selection']['selected_arm'],
         manifest_sha256=sha(run/'manifest.json'),selection_sha256=sha(run/'selection.json'),external_evaluation_unblinded=False,
         audit_execution_manifest_sha256=sha(run/'execution_manifest.finalize_v1.json')))
    update('complete',completed_utc=now(),result_file=str(run/'accounting.txt'))



def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(RUN))
    a=p.parse_args();run=Path(a.run).resolve();RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4)
    lock=(run/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original=read(run/'manifest.json');integrity(original)
    state=read(run/'status.json')
    if state['stage']=='complete':print('Already complete.',flush=True);return
    assert all(read(run/'workers'/f'gpu{i}'/'status.json')['stage']=='complete' for i in (0,1))
    records=read(run/'records.json');assert_clean(records)
    original_records_sha=sha(run/'records.json')
    diagnostic=run/'diagnostics/audit_norm_precision/diagnosis.json'
    proof=read(diagnostic)
    assert proof['checked_batches']==3000 and proof['float64_failing_vectors']==0
    assert proof['unchanged_absolute_tolerance']==1e-7 and proof['records_sha256']==original_records_sha
    backup=run/'status.before_finalize_v1.json'
    if not backup.exists():shutil.copy2(run/'status.json',backup)
    execution=run/'execution_manifest.finalize_v1.json'
    if execution.exists():
        manifest=read(execution);integrity(manifest)
    else:
        manifest=copy.deepcopy(original)
        manifest['source_sha256'][Path(__file__).name]=sha(Path(__file__))
        for path in [run/'manifest.json',run/'records.json',diagnostic,diagnostic.parent/'diagnose_counts.py',diagnostic.parent/'precision_checks.json',backup]:
            manifest['input_sha256'][str(path)]=sha(path)
        manifest['finalization_amendment']=dict(created_utc=now(),original_manifest_sha256=sha(run/'manifest.json'),
            reason='Float32 norm reduction rounded two saved feature-vector norms above the declared cap plus 1e-7, although the same vectors pass float64 and math.hypot with the unchanged tolerance.',
            change='Cast saved coordinate/feature deltas to float64 only when computing audit norms and reported maxima.',
            unchanged_tolerance=1e-7,new_sampling_outputs=0,new_score_calls=0,external_evaluation_unblinded=False)
        save(execution,manifest);integrity(manifest)
    state.pop('error',None)
    state.update(execution_manifest=str(execution),original_execution_manifest=str(run/'manifest.json'))
    def update(stage,**values):
        state.update(stage=stage,pid=os.getpid(),updated_utc=now(),**values)
        save(run/'status.json',state)
        print(json.dumps({k:state[k] for k in ('stage','updated_utc','verified_records','verified_batches','verified_events') if k in state}),flush=True)
    try:
        update('auditing')
        finalize(run,manifest,records,update)
        assert sha(run/'records.json')==original_records_sha
    except BaseException as error:
        update('failed',error=repr(error));raise


if __name__=='__main__':main()
