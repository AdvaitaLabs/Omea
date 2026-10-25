"""Deterministic single-site partitions and a model-independent output audit."""
import json
import hashlib
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]

def write_sdf(path,mols):
    with Chem.SDWriter(str(path)) as out:
        for mol in mols: out.write(mol)

def read_mol(path):
    mol=Chem.SDMolSupplier(str(path),removeHs=True)[0]
    if mol is None: raise ValueError(f'invalid input {path}')
    Chem.AssignStereochemistryFrom3D(mol)
    return mol

def strip_maps(mol):
    mol=Chem.Mol(mol)
    # Per-atom property lists from source force-field files become stale after
    # removing explicit H or an R group; none are model inputs or affinity labels.
    for key in list(mol.GetPropNames()):
        if key.startswith(('atom.dprop.','atom.iprop.','atom.prop.')):mol.ClearProp(key)
    for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
    return mol

def make_tasks(mol,pocket_id,protein_path,cohort):
    mol=strip_maps(Chem.RemoveHs(mol))
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    scaffold=MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms()==0:return []
    match=mol.GetSubstructMatch(scaffold)
    scaffold_set=set(match)
    remaining=set(range(mol.GetNumAtoms()))-scaffold_set
    components=[]
    while remaining:
        todo=[min(remaining)];comp=set()
        while todo:
            i=todo.pop()
            if i not in remaining:continue
            remaining.remove(i);comp.add(i)
            todo.extend(a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx() in remaining)
        components.append(comp)
    output=[]
    for comp in sorted(components,key=lambda x:min(x)):
        crossing=[b for b in mol.GetBonds() if (b.GetBeginAtomIdx() in comp)!=(b.GetEndAtomIdx() in comp)]
        if len(crossing)!=1:continue
        bond=crossing[0]
        if bond.IsInRing() or bond.GetBondType()!=Chem.BondType.SINGLE:continue
        kept=[i for i in range(mol.GetNumAtoms()) if i not in comp]
        anchor_old=bond.GetBeginAtomIdx() if bond.GetBeginAtomIdx() not in comp else bond.GetEndAtomIdx()
        edit=Chem.RWMol(mol)
        for idx in sorted(comp,reverse=True):edit.RemoveAtom(idx)
        core=edit.GetMol()
        # The cut atom receives an implicit H for a chemically valid input
        # fragment. This cap is not a fixed part of the generated heavy-atom graph.
        cap=core.GetAtomWithIdx(kept.index(anchor_old))
        cap.SetNoImplicit(False);cap.SetNumRadicalElectrons(0)
        core.ClearComputedProps();core.UpdatePropertyCache(strict=False)
        try:Chem.SanitizeMol(core)
        except Exception:continue
        if len(Chem.GetMolFrags(core))!=1:continue
        anchor=kept.index(anchor_old)
        case_id=f'{pocket_id}__a{anchor_old:03d}'
        if cohort.startswith('moad_'):case_id+=f'_r{min(comp):03d}'
        path=ROOT/'data'/'tasks'/cohort/case_id;path.mkdir(parents=True,exist_ok=True)
        core.SetProp('_Name',case_id+'_fixed')
        write_sdf(path/'fixed.sdf',[core]);write_sdf(path/'reference.sdf',[mol])
        # Dummy attachment is a graph label only, never a deleted R atom position.
        dummy=Chem.RWMol(core);star=dummy.AddAtom(Chem.Atom(0));dummy.AddBond(anchor,star,Chem.BondType.SINGLE)
        dummy.GetAtomWithIdx(star).SetAtomMapNum(1)
        (path/'anchor.smi').write_text(Chem.MolToSmiles(dummy)+'\n')
        task=dict(case_id=case_id,pocket_id=pocket_id,cohort=cohort,
            protein_path=str(Path(protein_path).resolve()),core_path=str(path/'fixed.sdf'),
            reference_path=str(path/'reference.sdf'),anchor_smiles_path=str(path/'anchor.smi'),
            fixed_reference_indices=kept,removed_reference_indices=sorted(comp),
            anchor_in_core=anchor,anchor_in_reference=anchor_old,
            fixed_heavy_atoms=len(kept),rgroup_heavy_atoms=len(comp),
            reference_heavy_atoms=mol.GetNumAtoms(),reference_charge=Chem.GetFormalCharge(mol),
            reference_smiles=Chem.MolToSmiles(mol),core_smiles=Chem.MolToSmiles(core))
        (path/'task.json').write_text(json.dumps(task,indent=2))
        output.append(task)
    return output

def atom_signature(atom):
    return atom.GetAtomicNum(),atom.GetFormalCharge(),atom.GetIsAromatic()

def audit(mol,task):
    result=dict(case_id=task['case_id'],valid=False,strict_success=False,reasons=[])
    if mol is None:
        result['reasons']=['missing_or_unreconstructable'];return result
    try:
        mol=strip_maps(Chem.RemoveHs(Chem.Mol(mol),sanitize=False))
        Chem.SanitizeMol(mol);Chem.AssignStereochemistryFrom3D(mol)
        result.update(valid=True,smiles=Chem.MolToSmiles(mol),heavy_atoms=mol.GetNumAtoms(),formal_charge=Chem.GetFormalCharge(mol))
        core=read_mol(task['core_path']);nc=core.GetNumAtoms();n=mol.GetNumAtoms()
        reference=read_mol(task['reference_path'])
        # Removing one of two identical substituents can make an achiral reference
        # atom spuriously chiral in the H-capped query. Enforce only reference
        # stereocentres, never stereochemistry created by the temporary cap.
        for atom in core.GetAtoms():
            ref_atom=reference.GetAtomWithIdx(task['fixed_reference_indices'][atom.GetIdx()])
            if ref_atom.GetChiralTag()==Chem.ChiralType.CHI_UNSPECIFIED:
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        result['size_match']=n==task['reference_heavy_atoms']
        result['connected']=len(Chem.GetMolFrags(mol))==1
        result['charge_match']=Chem.GetFormalCharge(mol)==task['reference_charge']
        result['new_design']=result['smiles']!=task['reference_smiles']
        if n<nc:result['reasons']=['fewer_atoms_than_fixed'];return result
        xyz=mol.GetConformer().GetPositions();fixed=core.GetConformer().GetPositions()
        costs=np.sum((fixed[:,None,:]-xyz[None,:,:])**2,axis=2)
        # Geometric assignment cannot hide atom-type changes: types are checked below.
        _,indices=linear_sum_assignment(costs)
        mapping=list(map(int,indices));kept=set(mapping);new=set(range(n))-kept
        displacement=np.sqrt(np.sum((fixed-xyz[indices])**2,axis=1))
        result['core_rmsd_A']=float(np.sqrt(np.mean(displacement**2)))
        result['core_max_displacement_A']=float(displacement.max())
        result['core_mapping']=mapping
        result['core_atoms_preserved']=all(atom_signature(a)==atom_signature(mol.GetAtomWithIdx(mapping[a.GetIdx()])) for a in core.GetAtoms())
        core_bonds={(b.GetBeginAtomIdx(),b.GetEndAtomIdx()):str(b.GetBondType()) for b in core.GetBonds()}
        inv={j:i for i,j in enumerate(mapping)}
        actual={tuple(sorted((inv[b.GetBeginAtomIdx()],inv[b.GetEndAtomIdx()]))):str(b.GetBondType()) for b in mol.GetBonds() if b.GetBeginAtomIdx() in kept and b.GetEndAtomIdx() in kept}
        expected={tuple(sorted(k)):v for k,v in core_bonds.items()}
        result['core_bonds_preserved']=actual==expected
        result['core_stereo_preserved']=tuple(mapping) in mol.GetSubstructMatches(core,useChirality=True,uniquify=False,maxMatches=10000)
        cross=[]
        for b in mol.GetBonds():
            i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
            if (i in kept)!=(j in kept):
                if i not in kept:i,j=j,i
                cross.append([inv[i],j,str(b.GetBondType())])
        result['cross_bonds']=cross
        result['single_site']=len(cross)==1 and cross[0][0]==task['anchor_in_core'] and cross[0][2]=='SINGLE'
        seen=set();todo=[min(new)] if new else []
        while todo:
            i=todo.pop()
            if i in seen:continue
            seen.add(i);todo.extend(a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx() in new and a.GetIdx() not in seen)
        result['rgroup_connected']=bool(new) and seen==new
        result['core_pose_preserved']=result['core_rmsd_A']<=0.1 and result['core_max_displacement_A']<=0.3
        required=['size_match','connected','core_atoms_preserved','core_bonds_preserved','core_stereo_preserved','single_site','rgroup_connected','core_pose_preserved']
        result['reasons']=[k for k in required if not result[k]]
        result['strict_success']=not result['reasons']
    except Exception as e:
        result['reasons'].append('chemical_or_audit_error');result['error']=str(e)
    return result

if __name__=='__main__':
    import csv
    plan=Path('/root/autodl-tmp/rgroup-judge-benchmark-plan/molecules.csv')
    rows=list(csv.DictReader(plan.open(encoding='utf-8-sig')))
    tasks=[]
    for row in rows:
        if row['is_reference']!='True':continue
        base=Path('/root/autodl-tmp/rgroup-judge-benchmark/judge_inputs')/row['target']
        tasks+=make_tasks(read_mol(base/(row['ligand_id']+'.sdf')),row['target']+'_'+row['ligand_id'],base/'receptor.pdb','calibration_controls')
    (ROOT/'data'/'calibration_tasks.json').write_text(json.dumps(tasks,indent=2))
    # Positive control plus deliberately incorrect site and pose. These tests guard
    # scientific validity; they are not a second implementation of the generator.
    checks=[]
    for task in tasks:
        mol=read_mol(task['reference_path']);good=audit(mol,task)
        assert good['strict_success'],good
        moved=Chem.Mol(mol);conf=moved.GetConformer()
        for i in task['fixed_reference_indices']:
            p=conf.GetAtomPosition(i);conf.SetAtomPosition(i,(p.x+1,p.y,p.z))
        bad=audit(moved,task);assert not bad['strict_success']
        wrong=dict(task,anchor_in_core=(task['anchor_in_core']+1)%task['fixed_heavy_atoms'])
        wrong_result=audit(mol,wrong);assert not wrong_result['single_site']
        checks.append(dict(case_id=task['case_id'],positive=True,wrong_site_rejected=True,pose_shift_rejected=True))
    (ROOT/'results'/'validator_tests.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(dict(calibration_tasks=len(tasks),validator_tests=len(checks)*3)))
