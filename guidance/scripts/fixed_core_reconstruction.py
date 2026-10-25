"""Versioned input-graph constraint for AutoFragDiff's coordinate reconstruction.

Restore only the supplied fixed-core graph. Never alter generated R-R edges,
core-R edges, R atom properties, or any coordinates. This is an adapter change,
not a claim of improvement from guidance. Incompatible chemistry still fails.
"""
import numpy as np
from rdkit import Chem


def preserve_core_graph(generated, core, max_core_displacement=0.01):
    mol = Chem.Mol(generated)
    nc = core.GetNumAtoms()
    if mol.GetNumAtoms() <= nc:
        raise ValueError('Missing R atoms')
    xyz = mol.GetConformer().GetPositions().copy()
    fixed = core.GetConformer().GetPositions()
    if np.max(np.abs(xyz[:nc]-fixed)) > max_core_displacement:
        raise ValueError('Generated core moved or atom order changed')
    if [a.GetAtomicNum() for a in mol.GetAtoms()][:nc] != [a.GetAtomicNum() for a in core.GetAtoms()]:
        raise ValueError('Generated core elements changed')
    outside_edges = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()):str(b.GetBondType())
                     for b in mol.GetBonds() if max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) >= nc}
    edit = Chem.RWMol(mol)
    for bond in list(edit.GetBonds()):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i < nc and j < nc:
            edit.RemoveBond(i, j)
    for atom in core.GetAtoms():
        edit.ReplaceAtom(atom.GetIdx(), Chem.Atom(atom), preserveProps=False)
    for bond in core.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edit.AddBond(i, j, bond.GetBondType())
        copied = edit.GetBondBetweenAtoms(i, j)
        copied.SetIsAromatic(bond.GetIsAromatic())
        copied.SetBondDir(bond.GetBondDir())
    # The attachment cap is an implicit H in the supplied fragment; RDKit
    # recomputes it with the unchanged generated crossing edge present.
    result = edit.GetMol()
    result.ClearComputedProps()
    Chem.SanitizeMol(result)
    Chem.AssignStereochemistryFrom3D(result, replaceExistingTags=True)
    if not np.array_equal(xyz, result.GetConformer().GetPositions()):
        raise AssertionError('Core graph constraint changed coordinates')
    after_edges = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()):str(b.GetBondType())
                   for b in result.GetBonds() if max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) >= nc}
    if after_edges != outside_edges:
        raise ValueError('Sanitization changed generated R or crossing bond orders')
    for i in range(nc, result.GetNumAtoms()):
        a, b = mol.GetAtomWithIdx(i), result.GetAtomWithIdx(i)
        if (a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsAromatic()) != (b.GetAtomicNum(), b.GetFormalCharge(), b.GetIsAromatic()):
            raise ValueError('Sanitization changed generated R atom properties')
    return result
