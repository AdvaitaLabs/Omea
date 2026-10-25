"""Frozen GNINA CNNs with libmolgrid coordinate gradients.

Atom/bond typing is discrete and held fixed during each gradient evaluation.
No gradient through reconstruction, atom identity, or a receptor coordinate.
"""
from pathlib import Path
import numpy as np
import torch
import molgrid
from openbabel import openbabel as ob
from rdkit import Chem
from gninatorch import gnina


def read_ob(path):
    conv = ob.OBConversion()
    conv.SetInFormat(Path(path).suffix.lstrip('.'))
    mol = ob.OBMol()
    if not conv.ReadFile(mol, str(path)):
        raise ValueError(f'OpenBabel failed to read {path}')
    mol.AddHydrogens()
    return mol


def ligand_ob(mol):
    # RDKit preserves heavy-atom order; OpenBabel appends hydrogens.
    mol = Chem.RemoveHs(Chem.Mol(mol))
    Chem.SanitizeMol(mol)
    conv = ob.OBConversion()
    conv.SetInFormat('sdf')
    result = ob.OBMol()
    if not conv.ReadString(result, Chem.MolToMolBlock(mol)):
        raise ValueError('Cannot type reconstructed ligand')
    result.AddHydrogens()
    return result


def typed(mol, typer, offset=0):
    rows = []
    for atom in ob.OBMolAtomIter(mol):
        typ, radius = typer.get_atom_type_index(atom)
        if typ < 0:
            continue
        rows.append((atom.GetIdx()-1, typ+offset, radius,
                     [atom.GetX(), atom.GetY(), atom.GetZ()]))
    if not rows:
        raise ValueError('No typed atoms')
    idx, types, radii, xyz = zip(*rows)
    return np.array(idx), np.array(types), np.array(radii), np.array(xyz)


class CoordinateScorer:
    def __init__(self, cnn='default', device='cuda', grid_device='cuda'):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device, self.grid_device = device, grid_device
        self.model, self.ensemble = gnina.setup_gnina_model(cnn=cnn)
        self.model = self.model.to(device).eval().requires_grad_(False)
        self.gmaker = molgrid.GridMaker(resolution=0.5, dimension=23.5)
        self.rec_typer = molgrid.defaultGninaReceptorTyper
        self.lig_typer = molgrid.defaultGninaLigandTyper
        self.nr = self.rec_typer.num_types()
        self.nt = self.nr + self.lig_typer.num_types()
        self.receptors = {}

    def receptor(self, protein):
        key = str(protein)
        if key not in self.receptors:
            _, ts, rs, xyz = typed(read_ob(protein), self.rec_typer)
            self.receptors[key] = (self.tensor(xyz), self.onehot(ts), self.tensor(rs))
        return self.receptors[key]

    def tensor(self, a):
        return torch.as_tensor(a, dtype=torch.float32, device=self.grid_device)

    def onehot(self, ts):
        return torch.nn.functional.one_hot(torch.as_tensor(ts, device=self.grid_device),
                                           self.nt).float()

    def ligand_types(self, mol):
        idx, ts, rs, _ = typed(ligand_ob(mol), self.lig_typer, self.nr)
        if not np.array_equal(idx, np.arange(mol.GetNumHeavyAtoms())):
            raise ValueError('Typed atoms do not match ligand heavy-atom order')
        return self.onehot(ts), self.tensor(rs)

    def score(self, xyz, ligand_types, protein, center):
        """Return affinity pK and pose probability; center is a fixed tuple."""
        rx, rt, rr = self.receptor(protein)
        lt, lr = ligand_types
        coords = torch.cat([rx, xyz.to(self.grid_device)], 0).contiguous()
        types = torch.cat([rt, lt], 0).contiguous()
        radii = torch.cat([rr, lr], 0).contiguous()
        grid = molgrid.Coords2Grid(self.gmaker, center=tuple(float(v) for v in center))(
            coords, types, radii)
        outputs = self.model(grid.unsqueeze(0).to(self.device))
        return outputs[1].reshape(()), outputs[0].exp()[0, 1]

    def affinity_gradient(self, mol, xyz, protein, center):
        types = self.ligand_types(mol)
        with torch.enable_grad():
            x = self.tensor(xyz).detach().requires_grad_(True)
            affinity, pose = self.score(x, types, protein, center)
            grad, = torch.autograd.grad(affinity, x)
        if not torch.isfinite(grad).all():
            raise ValueError('Nonfinite GNINA gradient')
        return float(affinity.detach()), float(pose.detach()), grad.detach().cpu().numpy()
