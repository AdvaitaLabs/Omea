"""Local, flat-bottom covalent geometry restraints derived only from a detached graph.

UFF equilibrium lengths and angles define broad allowed bands. No receptor,
reference R coordinates, affinity oracle, or nonbonded force-field term is used.
Only terms touching R atoms contribute, and core coordinates are detached.
"""
import itertools
import math
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers as FF


def restraints(mol, ncore, bond_relative_tolerance=.10, angle_tolerance_degrees=15.):
    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)
    Chem.SetHybridization(m)
    bonds = []; angles = []; unavailable = 0
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if max(i, j) < ncore: continue
        params = FF.GetUFFBondStretchParams(m, i, j)
        if params is None:
            unavailable += 1; continue
        length = float(params[1])
        if not math.isfinite(length) or length <= 0:
            unavailable += 1; continue
        bonds.append((i, j, length*(1-bond_relative_tolerance), length*(1+bond_relative_tolerance)))
    for a in m.GetAtoms():
        j = a.GetIdx()
        for i, k in itertools.combinations(sorted(n.GetIdx() for n in a.GetNeighbors()), 2):
            if max(i, j, k) < ncore: continue
            params = FF.GetUFFAngleBendParams(m, i, j, k)
            if params is None:
                unavailable += 1; continue
            angle = float(params[1])
            if not math.isfinite(angle) or not 0 < angle <= 180:
                unavailable += 1; continue
            # Cosine bounds avoid unstable acos derivatives near 0 or 180 degrees.
            lower = math.cos(math.radians(min(180., angle+angle_tolerance_degrees)))
            upper = math.cos(math.radians(max(0., angle-angle_tolerance_degrees)))
            angles.append((i, j, k, lower, upper))
    return dict(bonds=bonds, angles=angles, unavailable_parameters=unavailable)


def energy(x, ncore, terms):
    # Concatenation makes fixed-core gradients identically zero, including bonds crossing the anchor.
    xyz = torch.cat((x[:ncore].detach(), x[ncore:]), dim=0)
    loss = x.sum()*0
    if terms['bonds']:
        t = torch.as_tensor(terms['bonds'], device=x.device, dtype=x.dtype)
        i, j = t[:, 0].long(), t[:, 1].long()
        distance = torch.sqrt((xyz[i]-xyz[j]).square().sum(-1)+1e-16)
        lo, hi = t[:, 2], t[:, 3]
        width = (hi-lo)/2
        loss = loss + ((torch.relu(lo-distance)+torch.relu(distance-hi))/width).square().mean()
    if terms['angles']:
        t = torch.as_tensor(terms['angles'], device=x.device, dtype=x.dtype)
        i, j, k = t[:, 0].long(), t[:, 1].long(), t[:, 2].long()
        a, b = xyz[i]-xyz[j], xyz[k]-xyz[j]
        cosine = (a*b).sum(-1)/(torch.sqrt(a.square().sum(-1)+1e-16)*torch.sqrt(b.square().sum(-1)+1e-16))
        lo, hi = t[:, 3], t[:, 4]
        width = torch.clamp((hi-lo)/2, min=1e-3)
        loss = loss + ((torch.relu(lo-cosine)+torch.relu(cosine-hi))/width).square().mean()
    return loss


def gradient(xyz, ncore, context, config):
    if context is None:
        return np.zeros_like(xyz), dict(status='no_valid_context', energy=None, bonds=0, angles=0,
                                       unavailable_parameters=0)
    # Parameter typing is discrete context. No graph is differentiated or rewritten.
    terms = restraints(context, ncore, config['bond_relative_tolerance'], config['angle_tolerance_degrees'])
    with torch.enable_grad():
        x = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        loss = energy(x, ncore, terms)
        g, = torch.autograd.grad(loss, x)
    grad = g.detach().numpy()
    assert np.isfinite(grad).all() and not np.count_nonzero(grad[:ncore])
    return grad, dict(status='applied' if terms['bonds'] or terms['angles'] else 'no_available_parameters',
                     energy=float(loss.detach()), bonds=len(terms['bonds']), angles=len(terms['angles']),
                     unavailable_parameters=terms['unavailable_parameters'])
