"""Compare reconstructed reference bonds and charges with the PDB CCD.

This never changes generated molecules. It validates ground-truth chemistry before
freezing the task manifest; cached CCD files and mismatches remain inspectable.
"""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import requests
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB import PDBParser
from rdkit import Chem
from tasks import ROOT,read_mol

def fetch(code):
    dest=ROOT/'data'/'ccd'/(code+'.cif');url=f'https://files.rcsb.org/ligands/download/{code}.cif'
    try:
        if not dest.exists():
            r=requests.get(url,timeout=45);r.raise_for_status();dest.write_bytes(r.content)
        return dict(code=code,status='downloaded',url=url,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    except Exception as e:return dict(code=code,status='failed',url=url,error=repr(e))

def from_ccd(case):
    base,lig=case.split('_');code,chain,resid=lig.split(':')
    raw=ROOT/'data'/'moad_raw'/base.lower().replace('-bio','.bio')
    pdb=PDBParser(QUIET=True).get_structure('',raw)
    residue=pdb[0][chain][('H_'+code,int(resid),' ')]
    atoms=[a for a in residue.get_atoms() if a.element!='H']
    ids=[a.name for a in atoms];idx={name:i for i,name in enumerate(ids)}
    ccd=MMCIF2Dict(str(ROOT/'data'/'ccd'/(code+'.cif')))
    specs={name:(symbol,charge) for name,symbol,charge in zip(ccd['_chem_comp_atom.atom_id'],ccd['_chem_comp_atom.type_symbol'],ccd['_chem_comp_atom.charge'])}
    heavy={name for name,(symbol,_) in specs.items() if symbol not in ('H','D')}
    if set(ids)!=heavy:raise ValueError(f'CCD heavy atom mismatch missing={sorted(heavy-set(ids))}, extra={sorted(set(ids)-heavy)}')
    mol=Chem.RWMol()
    for atom in atoms:
        symbol,charge=specs[atom.name];a=Chem.Atom(symbol.capitalize());a.SetFormalCharge(int(charge));mol.AddAtom(a)
    types={'SING':Chem.BondType.SINGLE,'DOUB':Chem.BondType.DOUBLE,'TRIP':Chem.BondType.TRIPLE,'AROM':Chem.BondType.AROMATIC}
    for a,b,order in zip(ccd['_chem_comp_bond.atom_id_1'],ccd['_chem_comp_bond.atom_id_2'],ccd['_chem_comp_bond.value_order']):
        if a in idx and b in idx:mol.AddBond(idx[a],idx[b],types[order])
    conf=Chem.Conformer(len(atoms))
    for i,a in enumerate(atoms):conf.SetAtomPosition(i,a.coord.astype(float))
    mol.AddConformer(conf);mol=mol.GetMol();Chem.SanitizeMol(mol);Chem.AssignStereochemistryFrom3D(mol)
    mol.SetProp('_Name',case)
    return mol

if __name__=='__main__':
    (ROOT/'data'/'ccd').mkdir(exist_ok=True)
    ids=json.loads((ROOT/'data'/'official_moad_test_ids.json').read_text())
    codes=sorted({x.split('_')[1].split(':')[0] for x in ids})
    with cf.ThreadPoolExecutor(max_workers=6) as ex:downloads=list(ex.map(fetch,codes))
    (ROOT/'results'/'ccd_download_manifest.json').write_text(json.dumps(downloads,indent=2))
    rows=[]
    for case in ids:
        row=dict(case_id=case)
        try:
            ccd=from_ccd(case)
            path=ROOT/'data'/'moad_prepared'/case.replace(':','_')/'reference.sdf'
            source=read_mol(path)
            row.update(status='checked',ccd_smiles=Chem.MolToSmiles(ccd),source_smiles=Chem.MolToSmiles(source),exact_match=Chem.MolToSmiles(ccd)==Chem.MolToSmiles(source))
        except Exception as e:row.update(status='failed',error=repr(e))
        rows.append(row)
    summary=dict(downloaded=sum(x['status']=='downloaded' for x in downloads),components=len(codes),checked=sum(x['status']=='checked' for x in rows),exact_matches=sum(x.get('exact_match',False) for x in rows),cases=rows)
    (ROOT/'results'/'ccd_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'}))
