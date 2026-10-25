"""Constrain a CNN coordinate correction using only covalent/local geometry.

Three frozen-graph penalties: existing UFF bond/angle bands and severe
intramolecular nonbonded overlaps. No receptor, affinity score, reference R
coordinates, candidate ranking, final repair or new random draw is used.
The guarantee applies to this x0 correction, not to later diffusion/noise.
"""
import numpy as np
import torch
from rdkit import Chem
from cnn_local_geometry import restraints, energy


def make_terms(mol, ncore, config):
    local = restraints(mol, ncore, config['bond_relative_tolerance'], config['angle_tolerance_degrees'])
    graph_distance = Chem.GetDistanceMatrix(mol)
    table = Chem.GetPeriodicTable()
    radii = [table.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()]
    clashes = [(i, j, config['clash_radius_fraction'] * (radii[i] + radii[j]))
               for i in range(mol.GetNumAtoms()) for j in range(i+1, mol.GetNumAtoms())
               if j >= ncore and graph_distance[i, j] >= 3]
    return dict(**local, clashes=clashes)


def components(x, ncore, terms):
    xyz = torch.cat((x[:ncore].detach(), x[ncore:]))
    bond = energy(x, ncore, dict(bonds=terms['bonds'], angles=[]))
    angle = energy(x, ncore, dict(bonds=[], angles=terms['angles']))
    clash = x.sum()*0
    if terms['clashes']:
        t = torch.as_tensor(terms['clashes'], device=x.device, dtype=x.dtype)
        i, j = t[:, 0].long(), t[:, 1].long()
        distance = torch.sqrt((xyz[i]-xyz[j]).square().sum(-1)+1e-16)
        clash = (torch.relu(t[:, 2]-distance)/t[:, 2]).square().mean()
    return torch.stack((bond, angle, clash))


def values(xyz, ncore, terms):
    return components(torch.tensor(xyz, dtype=torch.float64), ncore, terms).detach().numpy()


def project(xyz, proposed, ncore, context, config):
    if context is None:
        return proposed, dict(status='no_valid_context')
    terms = make_terms(context, ncore, config)
    with torch.enable_grad():
        x = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        penalties = components(x, ncore, terms)
        grads = [torch.autograd.grad(penalties[i], x, retain_graph=i<2)[0].detach().numpy()
                 for i in range(3)]
    before = penalties.detach().numpy()
    d = np.array(proposed, dtype=np.float64, copy=True)
    d[:ncore] = 0
    assert np.isfinite(before).all() and all(np.isfinite(g).all() and not np.count_nonzero(g[:ncore]) for g in grads)
    iterations = 0
    # Alternating projections onto g_k dot d <= 0. Nonlinear checks below are
    # authoritative even if the finite projection budget has not converged.
    for iterations in range(config['projection_iterations']):
        changed = False
        for g in grads:
            squared = float(np.square(g).sum())
            slope = float((g*d).sum())
            if squared > 1e-24 and slope > 1e-14:
                d -= slope/squared*g
                changed = True
        if not changed:
            break
    # Uniform scaling preserves all linear inequalities (per-atom clipping may not).
    max_norm = float(np.linalg.norm(d, axis=1).max())
    d *= min(1., config['lambda_x']/max(max_norm, 1e-30))
    tolerance = 1e-10*np.maximum(1., before)
    factor = 1.
    accepted = False
    after = before.copy()
    for backtracks in range(config['backtracking_steps']+1):
        after = values(np.asarray(xyz, dtype=np.float64)+factor*d, ncore, terms)
        if np.isfinite(after).all() and np.all(after <= before+tolerance):
            accepted = True
            break
        factor *= .5
    if not accepted:
        d[:] = 0
        factor = 0.
        after = before.copy()
    else:
        d *= factor
    assert not np.count_nonzero(d[:ncore])
    assert np.linalg.norm(d, axis=1).max() <= config['lambda_x']+1e-12
    assert np.all(after <= before+tolerance)
    return d, dict(status='accepted' if accepted else 'zero_fallback',
        penalty_names=['bonds','angles','internal_clashes'], before=before.tolist(), after=after.tolist(),
        tolerance=tolerance.tolist(), gradient_dot_update=[float((g*d).sum()) for g in grads],
        projection_iterations=iterations+1, backtracks=backtracks, accepted_fraction=factor,
        bonds=len(terms['bonds']), angles=len(terms['angles']), clashes=len(terms['clashes']),
        proposed_max_A=float(np.linalg.norm(proposed, axis=1).max()),
        applied_max_A=float(np.linalg.norm(d, axis=1).max()))
