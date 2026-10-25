"""Retain full numerical replay evidence; never generate efficacy outputs or relax a gate."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('WANDB_MODE','disabled')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem,RDLogger
from run_selected_guidance_full import (NativeFinalEngine,SOURCE,SCREEN,TRAJECTORIES,ROOT,ARMS,BEST,
                                       read,save,sha,now,batch,hardware,integrity,audit,score_one)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();run=Path(args.run);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4)
    manifest=read(run/'manifest.json');integrity(manifest)
    tasks={t['case_id']:t for t in read(run/'tasks.json')}
    controls={(r['arm'],r['case_id'],r['attempt']):r for r in read(SOURCE/'records.json') if r['arm'] in ARMS[:-1]}
    selected={(r['arm'],r['case_id'],r['attempt']):r for r in read(SCREEN/'records.json') if r['arm']==BEST}
    configs={c['name']:c for c in manifest['configurations']}
    failed_case='3GT9-bio1_516_A_1__a002_r009'
    remaining=manifest['workers'][0]['case_ids'][5:]
    assert remaining[0]==failed_case
    jobs=[('joint_0p4',case,0) for case in remaining]+[(BEST,manifest['selection_case_ids'][0],0),('joint_0p4',failed_case,1)]
    report=dict(started_utc=now(),pid=os.getpid(),hardware=hardware(),original_coordinate_tolerance_A=.001,
                score_tolerance=.001,counts_as_new_efficacy_outputs=False,rows=[],stage='running')
    save(out/'report.json',report)
    engine=NativeFinalEngine()
    try:
        for arm,case,repeat in jobs:
            task=tasks[case];folder=out/f'{arm}_{case}_repeat{repeat}';folder.mkdir()
            engine.hook.configure(configs[arm],folder/'gradient')
            attempts,seeds,noise=batch(task,0);began=time.perf_counter()
            with torch.no_grad():outputs,xyz,numbers,metadata=engine.generate(task,attempts,seeds,noise)
            old_rows=selected if arm==BEST else controls
            old=old_rows[arm,case,0]
            expected_path=(Path(old['path']).parents[1]/'sampling_trace'/f'{case}_0000.npz') if arm==BEST else TRAJECTORIES/'arms'/arm/'sampling_trace'/f'{case}_0000.npz'
            with np.load(expected_path) as expected:
                difference=xyz-expected['xyz'];nc=task['fixed_heavy_atoms']
                row=dict(arm=arm,case_id=case,repeat=repeat,attempts=attempts,seeds=seeds,seconds=time.perf_counter()-began,
                         expected_trace=str(expected_path),expected_trace_sha256=sha(expected_path),
                         max_coordinate_delta_A=float(np.max(np.abs(difference))),
                         rms_coordinate_delta_A=float(np.sqrt(np.mean(difference**2))),
                         max_atom_displacement_A=float(np.max(np.linalg.norm(difference,axis=-1))),
                         core_exact=np.array_equal(xyz[:,:nc],expected['xyz'][:,:nc]),
                         elements_equal=np.array_equal(numbers,expected['atomic_numbers']),
                         noise_equal=metadata['noise_tape_sha256']==old['noise_tape_sha256'],outputs=[])
            np.savez_compressed(folder/'sampling_trace.npz',xyz=xyz,atomic_numbers=numbers,seeds=seeds)
            for i,(mol,error) in enumerate(outputs):
                previous=old_rows[arm,case,attempts[i]];actual_audit=audit(mol,task)
                changed={k:dict(old=previous['raw_audit'].get(k),replay=actual_audit.get(k))
                         for k in set(previous['raw_audit'])|set(actual_audit) if previous['raw_audit'].get(k)!=actual_audit.get(k)}
                detail=dict(attempt=attempts[i],reconstructed=mol is not None,reconstruction_error=error,
                            audit_equal=actual_audit==previous['raw_audit'],audit_differences=changed,
                            previous_strict=previous['raw_audit']['strict_success'],replay_strict=actual_audit['strict_success'])
                save(folder/f'audit_{i}.json',actual_audit)
                if mol is not None:
                    options=Chem.PropertyPickleOptions.AllProps|Chem.PropertyPickleOptions.CoordsAsDouble
                    (folder/f'output_{i}.rdmol').write_bytes(mol.ToBinary(options))
                if actual_audit['strict_success'] and 'raw_score' in previous:
                    score=score_one(mol,task['protein_path'],folder/f'judge_{i}')
                    old_score=previous['raw_score']
                    detail.update(score_status=score['status'],previous_score_status=old_score['status'],
                                  geometry_equal=score.get('posebusters_checks')==old_score.get('posebusters_checks'))
                    if score['status']==old_score['status']=='scored':
                        detail['score_deltas']={m:score[m]-old_score[m] for m in ('vina_kcal_mol','cnn_affinity_pK','cnn_pose_score')}
                        detail['score_deltas_below_original_tolerance']=all(abs(v)<.001 for v in detail['score_deltas'].values())
                row['outputs'].append(detail)
            row['passes_original_coordinate_guard']=row['max_coordinate_delta_A']<.001
            row['chemistry_geometry_and_scores_unchanged_within_original_rules']=bool(
                row['elements_equal'] and row['core_exact'] and row['noise_equal'] and
                all(o['audit_equal'] and (not o['previous_strict'] or (
                    o.get('geometry_equal') and o.get('score_deltas_below_original_tolerance'))) for o in row['outputs']))
            report['rows'].append(row);report['updated_utc']=now();save(out/'report.json',report)
            print(json.dumps(row),flush=True)
        repeated=[r for r in report['rows'] if r['case_id']==failed_case and r['arm']=='joint_0p4']
        arrays=[np.load(out/f'joint_0p4_{failed_case}_repeat{i}'/'sampling_trace.npz')['xyz'] for i in (0,1)]
        report.update(stage='complete',completed_utc=now(),same_gpu_repeat_max_coordinate_delta_A=float(np.max(np.abs(arrays[0]-arrays[1]))),
                      all_original_noncoordinate_rules_pass=all(r['chemistry_geometry_and_scores_unchanged_within_original_rules'] for r in report['rows']))
        save(out/'report.json',report)
    except BaseException as exc:
        report.update(stage='failed',error=repr(exc),updated_utc=now());save(out/'report.json',report);raise
    finally:engine.hook.close()


if __name__=='__main__':main()
