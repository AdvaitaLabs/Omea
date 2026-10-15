"""Same H-only preparation, CPU GNINA 1.1 and PoseBusters as the experiments."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tasks import write_sdf
from benchmark_io import ROOT
GNINA=Path(os.environ.get("GNINA_BIN", str(ROOT/"tools/gnina")))

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
    if mol.HasProp('scoring_serialization'):
        result['scoring_serialization']=json.loads(mol.GetProp('scoring_serialization'))
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
