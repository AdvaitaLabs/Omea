"""Infer stereo from unchanged coordinates; no graph or pose repair."""
import numpy as np
from rdkit import Chem

def normalize(mol):
    copied=Chem.Mol(mol);changed=[]
    for bond in copied.GetBonds():
        if bond.GetBondType()==Chem.BondType.DOUBLE and (bond.GetStereo()==Chem.BondStereo.STEREOANY or bond.GetBondDir()==Chem.BondDir.EITHERDOUBLE):
            changed.append(bond.GetIdx())
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            bond.SetBondDir(Chem.BondDir.NONE)
    assert np.array_equal(mol.GetConformer().GetPositions(),copied.GetConformer().GetPositions())
    return copied,changed
