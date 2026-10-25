"""Shared fixed-pose judges. Score changes are proxies, never measured DDG."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tasks import ROOT,read_mol,write_sdf,audit

GNINA=Path('/root/autodl-tmp/rgroup-judge-benchmark/bin/gnina-1.1')

def prepare(mol):
    mol=Chem.RemoveHs(Chem.Mol(mol));Chem.SanitizeMol(mol)
    xyz=mol.GetConformer().GetPositions().copy();n=mol.GetNumAtoms()
    hmol=Chem.AddHs(mol,addCoords=True)
    info={'hydrogens':'RDKit AddHs','heavy_atoms_fixed':True}
    if AllChem.MMFFHasAllMoleculeParams(hmol):
        props=AllChem.MMFFGetMoleculeProperties(hmol)
        ff=AllChem.MMFFGetMoleculeForceField(hmol,props)
        for i in range(n):ff.AddFixedPoint(i)
        ff.Initialize();info['H_minimization_status']=int(ff.Minimize(maxIts=200))
        info['H_forcefield']='MMFF94'
    else:info['H_forcefield']='unavailable; geometric H only'
    assert np.max(np.abs(hmol.GetConformer().GetPositions()[:n]-xyz))<1e-6
    return hmol,info

def score_one(mol,protein,out):
    out.mkdir(parents=True,exist_ok=True)
    result_path=out/'score.json'
    if result_path.exists():return json.loads(result_path.read_text())
    began=time.perf_counter();result={'status':'pending','mode':'fixed_pose_score_only'}
    try:
        hmol,prep=prepare(mol);write_sdf(out/'prepared.sdf',[hmol]);result['preparation']=prep
        result['input_sha256']=hashlib.sha256((out/'prepared.sdf').read_bytes()).hexdigest()
        cmd=[str(GNINA),'-r',str(protein),'-l',str(out/'prepared.sdf'),'--scoring','vina','--seed','20260930','--cpu','2','--no_gpu','--cnn_scoring','rescore','--score_only']
        t=time.perf_counter();proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
        result['gnina_seconds']=time.perf_counter()-t;result['command']=cmd
        (out/'gnina.log').write_text(proc.stdout)
        if proc.returncode:raise RuntimeError(f'GNINA exit {proc.returncode}: {proc.stdout[-1200:]}')
        for name,key in [('Affinity','vina_kcal_mol'),('CNNaffinity','cnn_affinity_pK'),('CNNscore','cnn_pose_score')]:
            match=re.search(rf'^{name}:\s*([-+0-9.eE]+)',proc.stdout,re.M)
            if match is None:raise ValueError(f'GNINA metric missing: {name}')
            result[key]=float(match.group(1))
        result['status']='scored'
        try:
            from posebusters import PoseBusters
            t=time.perf_counter()
            # dock omits identity and RMSD to a reference, which is a different molecule.
            pb=PoseBusters(config='dock',max_workers=0)
            frame=pb.bust(mol_pred=hmol,mol_cond=str(protein),full_report=False)
            checks={k:bool(v) for k,v in frame.iloc[0].items() if isinstance(v,(bool,np.bool_))}
            if not checks:raise ValueError('no PoseBusters checks returned')
            result['posebusters_checks']=checks;result['posebusters_pass']=all(checks.values())
            result['posebusters_seconds']=time.perf_counter()-t
            frame.to_csv(out/'posebusters.csv')
        except Exception as e:result['posebusters_error']=repr(e)
    except Exception as e:result.update(status='failed',error=repr(e))
    result['wall_seconds']=time.perf_counter()-began
    result_path.write_text(json.dumps(result,indent=2))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='moad_ccd_preflight')
    parser.add_argument('--tasks',default='data/moad_preflight_tasks.json');parser.add_argument('--workers',type=int,default=8);parser.add_argument('--model')
    args=parser.parse_args();tasks=json.loads(Path(args.tasks).read_text());rundir=ROOT/'results'/args.run
    jobs=[]
    for task in tasks:
        case=task['case_id']
        # Reference is shared across every model for this site.
        if not any(rundir.glob(f'*/{case}/summary.json')):continue
        jobs.append(('reference',task,None,read_mol(task['reference_path']),rundir/'judges'/'reference'/case))
        for path in sorted(rundir.glob(f'*/{case}/attempts.json')):
            model=path.parent.parent.name
            if args.model and model!=args.model:continue
            rows=json.loads(path.read_text());attempts={r['attempt']:r for r in rows}
            for mol in Chem.SDMolSupplier(str(path.parent/'raw.sdf'),removeHs=True,sanitize=False):
                if mol is None:continue
                attempt=int(mol.GetProp('attempt'));row=attempts[attempt]
                if not row['strict_success']:continue
                checked=audit(mol,task)
                if not checked['strict_success']:raise ValueError(f'SDF roundtrip audit changed: {model}, {case}, {attempt}, {checked}')
                jobs.append((model,task,attempt,mol,rundir/'judges'/model/case/f'{attempt:04d}'))
    results=[]
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future={pool.submit(score_one,mol,task['protein_path'],out):(model,task,attempt) for model,task,attempt,mol,out in jobs}
        for f in cf.as_completed(future):
            model,task,attempt=future[f];score=f.result()
            row=dict(model=model,case_id=task['case_id'],attempt=attempt,**score);results.append(row)
            print(json.dumps({k:row[k] for k in ['model','case_id','attempt','status','wall_seconds']}),flush=True)
    refs={r['case_id']:r for r in results if r['model']=='reference'}
    for row in results:
        ref=refs.get(row['case_id'],{})
        if row['model']!='reference' and row['status']==ref.get('status')=='scored':
            row['delta_vina_kcal_mol']=row['vina_kcal_mol']-ref['vina_kcal_mol']
            row['delta_cnn_affinity_pK']=row['cnn_affinity_pK']-ref['cnn_affinity_pK']
    result=dict(protocol='Fixed heavy-atom pose. Same H preparation and CPU GNINA 1.1. Only strict successes scored; all raw attempts remain denominators in generator summaries.',interpretation='delta_vina < 0 and delta_cnn_affinity_pK > 0 are predicted improvements. Neither is physical DDG.',rows=results)
    result_name=f'judge_results_{args.model}.json' if args.model else 'judge_results.json'
    tmp=rundir/(result_name+'.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(rundir/result_name)
    print(json.dumps(dict(scored=sum(r['status']=='scored' for r in results),failed=sum(r['status']=='failed' for r in results),total=len(results))),flush=True)

if __name__=='__main__':main()
