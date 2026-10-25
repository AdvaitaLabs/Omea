"""Recover only aromatic explicit-H metadata lost by non-kekulized V2000.

The generator's audited canonical SMILES is an existing output record. It is
never used to repair topology or choose a different molecule. Recovery requires
an identical heavy-atom graph, an unambiguous aromatic-H assignment across all
graph isomorphisms, unchanged coordinates, and the original audit and SMILES.
"""
import numpy as np
import networkx as nx
from rdkit import Chem
from tasks import audit
from reanalyze_geometry_stereo import normalize


class AmbiguousSerializedHydrogen(ValueError):
    """The original aromatic-H position cannot be determined from the archive."""


def heavy_graph(molecule):
    graph = nx.Graph()
    for atom in molecule.GetAtoms():
        graph.add_node(atom.GetIdx(), label=(atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope()))
    for bond in molecule.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label=str(bond.GetBondType()))
    return graph


def recover_for_scoring(molecule, recorded, task, verified_metadata=None):
    molecule, _ = normalize(molecule)
    checked = audit(molecule, task)
    if checked['strict_success']:
        if checked['smiles'] != recorded['smiles']:
            raise ValueError('Serialized molecule identity differs from its recorded audited SMILES')
        return molecule, dict(recovered=False)
    expected = Chem.MolFromSmiles(recorded['smiles'])
    if expected is None:
        raise ValueError('Invalid recorded audited SMILES')
    raw_graph, expected_graph = heavy_graph(molecule), heavy_graph(expected)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        raw_graph, expected_graph,
        node_match=lambda a, b: a['label'] == b['label'],
        edge_match=lambda a, b: a['label'] == b['label'])
    assignments = set()
    mappings = 0
    verified_assignment = None
    if verified_metadata is not None:
        assert verified_metadata['canonical_smiles'] == recorded['smiles']
        assert verified_metadata['heavy_atom_labels_identical'] and verified_metadata['bonds_identical']
        assert verified_metadata['max_saved_coordinate_error_A'] <= .000051
        values=verified_metadata['explicit_H_by_atom']
        assert len(values)==molecule.GetNumAtoms()
        verified_assignment=tuple(values[index] if any(b.GetBondType()==Chem.BondType.AROMATIC for b in atom.GetBonds())
                                  else atom.GetNumExplicitHs() for index,atom in enumerate(molecule.GetAtoms()))
    for mapping in matcher.isomorphisms_iter():
        mappings += 1
        if mappings > 10000:
            raise ValueError('Too many graph symmetries to prove unique aromatic-H recovery')
        assignment = tuple(expected.GetAtomWithIdx(mapping[index]).GetNumExplicitHs()
                           if expected.GetAtomWithIdx(mapping[index]).GetIsAromatic()
                           else molecule.GetAtomWithIdx(index).GetNumExplicitHs()
                           for index in range(molecule.GetNumAtoms()))
        if verified_assignment is not None and assignment != verified_assignment:
            continue
        assignments.add(assignment)
        if len(assignments) > 1:
            raise AmbiguousSerializedHydrogen('Aromatic-H atom placement lost in SDF is ambiguous; no recovered structure or affinity score is assigned')
    if not assignments:
        raise ValueError('Recorded SMILES and serialized heavy-atom graphs differ')
    recovered = Chem.Mol(molecule)
    assignment = next(iter(assignments))
    changes = []
    for atom, explicit_h in zip(recovered.GetAtoms(), assignment):
        if atom.GetNumExplicitHs() != explicit_h:
            changes.append(dict(atom=atom.GetIdx(), before=atom.GetNumExplicitHs(), after=explicit_h))
            atom.SetNumExplicitHs(explicit_h)
    if not changes:
        raise ValueError('Audit failure is not caused by lost aromatic explicit-H metadata')
    assert nx.is_isomorphic(raw_graph, heavy_graph(recovered),
                           node_match=lambda a, b: a['label'] == b['label'],
                           edge_match=lambda a, b: a['label'] == b['label'])
    assert np.array_equal(molecule.GetConformer().GetPositions(), recovered.GetConformer().GetPositions())
    checked = audit(recovered, task)
    if not checked['strict_success'] or checked['smiles'] != recorded['smiles']:
        raise ValueError(f'Recovered metadata does not reproduce the original audited molecule: {checked}')
    return recovered, dict(recovered=True, method='unambiguous aromatic explicit-H metadata from existing audited SMILES',
                          heavy_graph_unchanged=True, coordinates_unchanged=True,
                          original_smiles_verified=True, aromatic_H_changes=changes, checked_isomorphisms=mappings)
