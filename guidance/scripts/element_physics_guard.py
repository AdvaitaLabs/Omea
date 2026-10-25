"""Proposal constraints; no affinity/reference score is used in this module.

MMFF bond/angle energy is a local geometry heuristic, not a binding energy.
Reconstruction must preserve the proposed labeled graph before it is scored.
"""
from collections import Counter
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from element_guidance_objective import enumerate_substitutions, NUMBERS
from fixed_core_reconstruction import preserve_core_graph


def bond_angle_strain(mol):
    hmol = Chem.AddHs(mol, addCoords=True)
    if not AllChem.MMFFHasAllMoleculeParams(hmol):
        raise ValueError('missing_mmff_parameters')
    props = AllChem.MMFFGetMoleculeProperties(hmol)
    props.SetMMFFOopTerm(False)
    props.SetMMFFStretchBendTerm(False)
    props.SetMMFFTorsionTerm(False)
    props.SetMMFFVdWTerm(False)
    props.SetMMFFEleTerm(False)
    ff = AllChem.MMFFGetMoleculeForceField(hmol, props)
    for i in range(mol.GetNumAtoms()):
        ff.AddFixedPoint(i)
    ff.Initialize()
    ff.Minimize(maxIts=100)
    value = float(ff.CalcEnergy())
    if not np.isfinite(value):
        raise ValueError('nonfinite_mmff_strain')
    return value


def graph_signature(mol):
    return (
        tuple((a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsAromatic(), a.GetTotalNumHs())
              for a in mol.GetAtoms()),
        tuple(sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                      max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), str(b.GetBondType()))
                     for b in mol.GetBonds())),
    )


class PhysicsGuard:
    def __init__(self, protein):
        protein_mol = Chem.MolFromPDBFile(str(protein), sanitize=False,
                                        removeHs=False, proximityBonding=False)
        if protein_mol is None:
            raise ValueError('Cannot read receptor for proposal guard')
        ids = [a.GetIdx() for a in protein_mol.GetAtoms() if a.GetAtomicNum() > 1]
        self.xyz = protein_mol.GetConformer().GetPositions()[ids]
        pt = Chem.GetPeriodicTable()
        self.radii = np.array([pt.GetRvdw(protein_mol.GetAtomWithIdx(i).GetAtomicNum()) for i in ids])

    def contacts(self, mol, ncore):
        xyz = mol.GetConformer().GetPositions()[ncore:]
        radii = np.array([Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum())
                          for a in list(mol.GetAtoms())[ncore:]])
        distance = np.linalg.norm(xyz[:, None] - self.xyz[None, :], axis=-1)
        sums = radii[:, None] + self.radii[None, :]
        hard = distance < .75 * sums
        soft = float(np.square(np.maximum(.85 * sums - distance, 0.)).sum())
        return hard, soft

    def proposals(self, mol, core, reconstruct):
        ncore = core.GetNumAtoms()
        candidates, labels, rejected = enumerate_substitutions(mol, ncore)
        reasons = Counter(x['error'] for x in rejected)
        try:
            base_strain = bond_angle_strain(mol)
        except ValueError as exc:
            return [mol], [None], dict(reasons, baseline_guard=str(exc)), []
        base_hard, base_soft = self.contacts(mol, ncore)
        kept, kept_labels, diagnostics = [mol], [None], []
        for candidate, label in zip(candidates[1:], labels[1:]):
            try:
                with rdBase.BlockLogs():
                    strain = bond_angle_strain(candidate)
                if strain > base_strain + 1.0:
                    raise ValueError('bond_angle_strain_increase_gt_1_kcal')
                hard, soft = self.contacts(candidate, ncore)
                if np.any(hard & ~base_hard):
                    raise ValueError('new_receptor_clash')
                if soft > base_soft + .01:
                    raise ValueError('increased_receptor_overlap')
                xyz = candidate.GetConformer().GetPositions()
                with rdBase.BlockLogs():
                    rebuilt = reconstruct(xyz.tolist(), [a.GetAtomicNum() for a in candidate.GetAtoms()])
                    Chem.SanitizeMol(rebuilt)
                    rebuilt = preserve_core_graph(rebuilt, core)
                if graph_signature(rebuilt) != graph_signature(candidate):
                    raise ValueError('proposal_decoder_graph_mismatch')
                kept.append(candidate)
                kept_labels.append(label)
                diagnostics.append(dict(atom=label[0], element=NUMBERS[label[1]],
                                        strain_delta=strain-base_strain,
                                        overlap_delta=soft-base_soft))
            except (ValueError, RuntimeError) as exc:
                reasons[str(exc)[:160]] += 1
        return kept, kept_labels, dict(reasons), diagnostics
