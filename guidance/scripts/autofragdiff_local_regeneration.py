"""R-only forward noising and reverse sampling with the frozen AutoFragDiff EDM.

No reference R coordinates, affinity labels, scoring force or learned parameters
enter sampling. Full sampling is a regression-tested copy of the native path.
"""
import inspect
import sys
import numpy as np
import torch
from rdkit import Chem
from tasks import ROOT, read_mol
from fixed_core_reconstruction import preserve_core_graph
from reanalyze_geometry_stereo import normalize

NUMBERS = np.array([6, 7, 8, 16, 5, 35, 17, 15, 53, 9])


def initial_state(edm, inputs, mode, restart_steps):
    x, h = edm.normalize(inputs['x'], inputs['h'])
    xh = torch.cat([x, h], dim=2)
    fixed = inputs['scaffold_mask'].unsqueeze(-1)
    extension = inputs['extension_mask'].unsqueeze(-1)
    noise = edm.sample_combined_position_features_noise(x.size(0), x.size(1), extension)
    if mode == 'full':
        assert restart_steps == edm.T
        z = xh * fixed + noise * extension
        alpha, sigma = 0., 1.
    else:
        assert mode == 'local' and 0 < restart_steps < edm.T
        t = torch.full((x.size(0), 1), restart_steps / edm.T, device=x.device)
        gamma = edm.gamma(t)
        a, s = edm.alpha(gamma, xh), edm.sigma(gamma, xh)
        z = xh * fixed + (a * xh + s * noise) * extension
        alpha, sigma = float(a.flatten()[0]), float(s.flatten()[0])
    return z, xh, dict(alpha=alpha, sigma=sigma, restart_steps=restart_steps)


@torch.no_grad()
def sample_restart(edm, inputs, mode, restart_steps):
    z, clean, metadata = initial_state(edm, inputs, mode, restart_steps)
    px, ph = edm.normalize(inputs['pocket_x'], inputs['pocket_h'])
    context = {k: inputs[k] for k in ['scaffold_mask', 'extension_mask', 'anchors',
                                     'pocket_anchors', 'pocket_mask']}
    context['pocket_xh'] = torch.cat([px, ph], dim=2)
    fixed = inputs['scaffold_mask'].unsqueeze(-1)
    core_error = torch.max(torch.abs((z - clean) * fixed))
    for step in reversed(range(restart_steps)):
        s = torch.full((z.size(0), 1), step, device=z.device) / edm.T
        t = torch.full((z.size(0), 1), step + 1, device=z.device) / edm.T
        z = edm.sample_p_zs_given_zt_only_extension(s=s, t=t, z_t=z, **context)
        core_error = torch.maximum(core_error, torch.max(torch.abs((z - clean) * fixed)))
    assert torch.isfinite(z).all(), 'Nonfinite sampling state'
    assert float(core_error) == 0., 'Fixed core changed in reverse sampling'
    x, h = edm.sample_p_xh_given_z0_only_extension(z_0=z, **context)
    assert torch.equal(x * fixed, inputs['x'] * fixed), 'Decoder moved core coordinates'
    assert torch.equal(h * fixed, inputs['h'] * fixed), 'Decoder changed core elements'
    metadata.update(core_max_normalized_error=float(core_error), denoiser_calls=restart_steps + 1)
    return x, h, metadata


class LocalRegenerator:
    def __init__(self):
        sys.path.insert(0, str(ROOT / 'vendor/AutoFragDiff'))
        import Bio.PDB.Polypeptide as pp
        from Bio.SeqUtils import seq1
        if not hasattr(pp, 'three_to_one'):
            pp.three_to_one = seq1
        from src.lightning import AR_DDPM
        from utils.templates import get_pocket, atom_dict, pocket_atom_dict
        from analysis.reconstruct_mol import reconstruct_from_generated
        data = torch.load(ROOT / 'checkpoints/autofragdiff_pocket-gvp.ckpt',
                          map_location='cpu', weights_only=False)
        accepted = inspect.signature(AR_DDPM.__init__).parameters
        self.model = AR_DDPM(**{k: v for k, v in data['hyper_parameters'].items() if k in accepted})
        self.model.load_state_dict(data['state_dict'], strict=True)
        self.model = self.model.to('cuda').eval().requires_grad_(False)
        self.edm = self.model.edm
        self.get_pocket, self.atom_dict, self.pocket_atom_dict = get_pocket, atom_dict, pocket_atom_dict
        self.reconstruct = reconstruct_from_generated
        self.cache = {}

    def inputs(self, task, parent, count):
        key = task['case_id']
        if key not in self.cache:
            ph, px, cx, ch = self.get_pocket(task['protein_path'], task['core_path'],
                                            self.atom_dict, self.pocket_atom_dict, dist_cutoff=7)
            core = read_mol(task['core_path'])
            assert np.allclose(cx, core.GetConformer().GetPositions(), atol=1e-6) and len(px)
            self.cache[key] = ph, px, cx, ch, core
        ph, px, cx, ch, core = self.cache[key]
        nc, nr = len(cx), task['rgroup_heavy_atoms']
        center = cx[task['anchor_in_core']]
        assert parent.GetNumAtoms() == nc + nr
        pos = parent.GetConformer().GetPositions()
        assert np.max(np.abs(pos[:nc] - cx)) <= 1e-4
        kinds = [int(np.where(NUMBERS == a.GetAtomicNum())[0][0]) for a in parent.GetAtoms()]
        parent_h = np.eye(len(NUMBERS))[kinds]
        def tensor(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.float32, device='cuda')
        x = tensor(pos - center)[None].repeat(count, 1, 1)
        h = tensor(parent_h)[None].repeat(count, 1, 1)
        x[:, :nc] = tensor(cx - center)
        h[:, :nc] = tensor(ch)
        fixed = torch.zeros((count, nc + nr), device='cuda'); fixed[:, :nc] = 1
        anchors = torch.zeros_like(fixed); anchors[:, task['anchor_in_core']] = 1
        pm = torch.ones((count, len(px)), device='cuda')
        return dict(x=x, h=h, scaffold_mask=fixed, extension_mask=1-fixed, anchors=anchors,
                    pocket_x=tensor(px-center)[None].repeat(count, 1, 1),
                    pocket_h=tensor(ph)[None].repeat(count, 1, 1), pocket_mask=pm,
                    pocket_anchors=torch.zeros_like(pm)), center, core

    def generate(self, task, parent, count, mode, restart_steps):
        inputs, center, core = self.inputs(task, parent, count)
        x, h, metadata = sample_restart(self.edm, inputs, mode, restart_steps)
        xyz = x.cpu().numpy() + center
        numbers = NUMBERS[h.argmax(-1).cpu().numpy()]
        assert np.isfinite(xyz).all()
        output = []
        for pos, atoms in zip(xyz, numbers):
            try:
                mol = self.reconstruct(pos.tolist(), atoms.tolist())
                Chem.SanitizeMol(mol)
                mol = preserve_core_graph(mol, core)
                mol = normalize(mol)[0]
                # All candidates use the existing benchmark's coordinate precision.
                rounded = np.round(mol.GetConformer().GetPositions(), 4)
                for i, point in enumerate(rounded):
                    mol.GetConformer().SetAtomPosition(i, point)
                output.append((mol, None))
            except AssertionError:
                raise
            except Exception as exc:
                output.append((None, repr(exc)))
        return output, xyz, numbers, metadata
