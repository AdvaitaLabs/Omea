"""Permutation-invariant graph changes; core labels fixed, R indices exchangeable."""
import networkx as nx
from rdkit import Chem


def graph(mol, nc, chemistry=False, bond_order=True):
    mol = Chem.Mol(mol); Chem.SanitizeMol(mol)
    g = nx.Graph()
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        label = (i if i < nc else -1,)
        if chemistry:
            label += (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsAromatic(), atom.GetTotalNumHs())
        g.add_node(i, label=label)
    for bond in mol.GetBonds():
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                   label=str(bond.GetBondType()) if bond_order else 'edge')
    return g


def equivalent(a, b, nc, chemistry=False, bond_order=True):
    return nx.is_isomorphic(graph(a, nc, chemistry, bond_order), graph(b, nc, chemistry, bond_order),
                            node_match=lambda x,y: x['label']==y['label'],
                            edge_match=lambda x,y: x['label']==y['label'])


def changes(parent, candidate, nc):
    chem = not equivalent(parent, candidate, nc, chemistry=True)
    bonds = not equivalent(parent, candidate, nc)
    adjacency = not equivalent(parent, candidate, nc, bond_order=False)
    kind = ('connectivity_changed' if adjacency else 'bond_order_changed' if bonds else
            'atom_chemistry_changed' if chem else 'same_chemical_graph')
    return dict(chemical_graph_changed=chem, bond_graph_changed=bonds,
                connectivity_changed=adjacency, change_category=kind,
                parent_smiles=Chem.MolToSmiles(parent, isomericSmiles=False),
                candidate_smiles=Chem.MolToSmiles(candidate, isomericSmiles=False))


def check_graph_classifier():
    a = Chem.MolFromSmiles('CCCC')
    assert changes(a, Chem.RenumberAtoms(a, [0, 3, 2, 1]), 1)['change_category']=='same_chemical_graph'
    assert changes(a, Chem.MolFromSmiles('CCCN'), 1)['change_category']=='atom_chemistry_changed'
    assert changes(a, Chem.MolFromSmiles('CCC=O'), 1)['change_category']=='bond_order_changed'
    assert changes(a, Chem.MolFromSmiles('CC(C)C'), 1)['change_category']=='connectivity_changed'
    return dict(permuted_R_not_a_change=True, element_change_detected=True,
                bond_order_change_detected=True, branching_change_detected=True)
