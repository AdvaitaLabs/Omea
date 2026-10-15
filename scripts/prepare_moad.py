"""Reconstruct the official test split, then derive the declared single-site task."""
import collections
import json
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser,PDBIO,Select
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tasks import ROOT,make_tasks,audit,write_sdf

class ProteinHeavy(Select):
    def accept_model(self,model):return model.id==0
    def accept_residue(self,residue):return is_aa(residue.get_resname(),standard=True)
    def accept_atom(self,atom):return atom.element!='H' and atom.get_altloc() in (' ','A')

def main():
    ids=json.loads((ROOT/'data'/'official_moad_test_ids.json').read_text())
    rows=[];tasks=[]
    for case in ids:
        row=dict(official_id=case,status='pending')
        try:
            base,lig=case.split('_');name,chain,resid=lig.split(':')
            raw=ROOT/'data'/'moad_raw'/(base.lower().replace('-bio','.bio'))
            if not raw.exists():raise FileNotFoundError(raw)
            structure=PDBParser(QUIET=True).get_structure(base,raw)
            residue=structure[0][chain][('H_'+name,int(resid),' ')]
            atoms=[a for a in residue.get_atoms() if a.element!='H']
            coords=np.array([a.coord for a in atoms]);symbols=[a.element.capitalize() for a in atoms]
            if any(s not in {'C','N','O','S','P','F','Cl','Br','I'} for s in symbols):raise ValueError('unsupported ligand element')
            from audit_ccd import from_ccd
            mol=from_ccd(case)
            if len(Chem.GetMolFrags(mol))!=1:raise ValueError('reference disconnected after reconstruction')
            mol.SetProp('_Name',case);Chem.AssignStereochemistryFrom3D(mol)
            pocket_id=case.replace(':','_')
            dest=ROOT/'data'/'moad_prepared'/pocket_id;dest.mkdir(parents=True,exist_ok=True)
            protein=dest/'protein.pdb';io=PDBIO();io.set_structure(structure);io.save(str(protein),ProteinHeavy())
            if (dest/'reference.sdf').exists() and not (dest/'reference_openbabel_initial.sdf').exists():
                import shutil
                shutil.copy2(dest/'reference.sdf',dest/'reference_openbabel_initial.sdf')
            write_sdf(dest/'reference.sdf',[mol])
            scaffold=MurckoScaffold.GetScaffoldForMol(mol)
            derived=make_tasks(mol,pocket_id,protein,'moad_single_site')
            for task in derived:
                check=audit(mol,task)
                if not check['strict_success']:raise ValueError('reference positive control failed: '+str(check))
                task['official_id']=case
                task['pdb_id']=base[:4]
            tasks.extend(derived)
            row.update(status='prepared',heavy_atoms=mol.GetNumAtoms(),scaffold_atoms=scaffold.GetNumAtoms(),single_site_tasks=len(derived),pdb_id=base[:4],smiles=Chem.MolToSmiles(mol),source=str(raw))
        except Exception as e:row.update(status='excluded',error=repr(e))
        rows.append(row)
    assert len({t['case_id'] for t in tasks})==len(tasks), 'duplicate task ids'
    (ROOT/'data'/'moad_tasks.json').write_text(json.dumps(tasks,indent=2))
    # Preflight picks one deterministic, nontrivial site per PDB, without scores.
    candidates=sorted(tasks,key=lambda t:(t['pdb_id'],t['case_id']))
    chosen=[];seen=set()
    for task in candidates:
        if task['pdb_id'] in seen or not (2<=task['rgroup_heavy_atoms']<=10):continue
        chosen.append(task);seen.add(task['pdb_id'])
        if len(chosen)==10:break
    (ROOT/'data'/'moad_preflight_tasks.json').write_text(json.dumps(chosen,indent=2))
    summary=dict(official_split_cases=len(ids),prepared=sum(r['status']=='prepared' for r in rows),
        single_site_tasks=len(tasks),eligible_pocket_cases=len({t['pocket_id'] for t in tasks}),
        eligible_pdb_ids=len({t['pdb_id'] for t in tasks}),preflight_cases=len(chosen),
        rgroup_size_histogram=dict(collections.Counter(t['rgroup_heavy_atoms'] for t in tasks)),
        exact_paper_55_membership_verified=False,
        cohort_description='single-site extension derived from the official 130-complex split; not asserted identical to paper 55-case inpainting subset',
        receptor_preparation='standard amino-acid heavy atoms; no waters or other ligands; both models extract pocket around fixed fragment',
        reference_reconstruction='PDB CCD atom-name-mapped elements, bond orders and formal charges; crystallographic coordinates retained; incomplete CCD atom sets excluded; no deleted-R coordinates passed into generation',cases=rows)
    (ROOT/'results'/'moad_preparation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'}),flush=True)

if __name__=='__main__':main()
