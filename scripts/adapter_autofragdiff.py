# SPDX-License-Identifier: MIT
# Upstream source: https://github.com/keiserlab/autofragdiff/blob/84f0885cb12e6ac4abc7558870f8d304c78c8a38/LICENSE
# Adapter changes: retained-core inputs, fixed raw budgets; see docs/methods.md.
"""Official AutoFragDiff fragment sampler with a supplied anchor and R size.

The released high-level example predicts its own anchor/size (and ignores the
custom-anchor flag). We call its native single-fragment denoiser directly using
the benchmark-supplied anchor and size, with frozen weights and no score search.
"""
import hashlib
import inspect
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from tasks import ROOT, read_mol


def make_autofragdiff(args):
    repo = ROOT / 'vendor/AutoFragDiff'
    sys.path.insert(0, str(repo))
    import Bio.PDB.Polypeptide as pp
    from Bio.SeqUtils import seq1
    if not hasattr(pp, 'three_to_one'):
        pp.three_to_one = seq1
    from src.lightning import AR_DDPM
    from utils.templates import get_pocket, atom_dict, pocket_atom_dict
    from analysis.reconstruct_mol import reconstruct_from_generated
    checkpoint = ROOT / 'checkpoints/autofragdiff_pocket-gvp.ckpt'
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    params = dict(data['hyper_parameters'])
    accepted = inspect.signature(AR_DDPM.__init__).parameters
    ignored = {k: v for k, v in params.items() if k not in accepted}
    model = AR_DDPM(**{k: v for k, v in params.items() if k in accepted})
    model.load_state_dict(data['state_dict'], strict=True)
    model = model.to('cuda').eval().requires_grad_(False)
    numbers = [6, 7, 8, 16, 5, 35, 17, 15, 53, 9]
    cache = {}

    def generate(task, count):
        if task['case_id'] not in cache:
            ph, px, x, h = get_pocket(task['protein_path'], task['core_path'],
                                    atom_dict, pocket_atom_dict, dist_cutoff=7)
            if not len(px):
                raise ValueError('AutoFragDiff pocket has no supported atoms')
            cache[task['case_id']] = (ph, px, x, h)
        ph, px, core_x, core_h = cache[task['case_id']]
        nc, nr = len(core_x), task['rgroup_heavy_atoms']
        center = core_x[task['anchor_in_core']]
        def tensor(a):
            return torch.as_tensor(a, dtype=torch.float32, device='cuda')
        x = torch.zeros((count, nc + nr, 3), device='cuda')
        h = torch.zeros((count, nc + nr, 10), device='cuda')
        x[:, :nc] = tensor(core_x - center)
        h[:, :nc] = tensor(core_h)
        fixed = torch.zeros((count, nc + nr), device='cuda')
        fixed[:, :nc] = 1
        extension = 1 - fixed
        anchors = torch.zeros_like(fixed)
        anchors[:, task['anchor_in_core']] = 1
        pocket_x = tensor(px - center)[None].repeat(count, 1, 1)
        pocket_h = tensor(ph)[None].repeat(count, 1, 1)
        pm = torch.ones((count, len(px)), device='cuda')
        x, h, _ = model.edm.sample_chain_single_fragment(
            x=x, h=h, extension_mask=extension, scaffold_mask=fixed,
            anchors=anchors, pocket_x=pocket_x, pocket_h=pocket_h,
            pocket_mask=pm, pocket_anchors=torch.zeros_like(pm),
            keep_frames=1, lj_guidance=False)
        xyz = x.detach().cpu().numpy() + center
        kinds = h.argmax(-1).detach().cpu().numpy()
        # Verify the denoiser really holds the supplied core fixed before
        # reconstruction; do not overwrite or repair the generated core.
        if np.max(np.abs(xyz[:, :nc] - core_x)) > 0.01:
            raise ValueError('AutoFragDiff changed fixed-core coordinates')
        result = []
        for pos, at in zip(xyz, kinds):
            try:
                result.append(reconstruct_from_generated(pos.tolist(), [numbers[i] for i in at]))
            except Exception:
                result.append(None)
        return result

    return generate, dict(checkpoint=str(checkpoint),
        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timesteps=model.edm.T, pocket_radius_A=7,
        input='fixed core, retained anchor, deleted R heavy-atom count only',
        sampler='native sample_chain_single_fragment; one fragment, supplied anchor and size',
        native_anchor_condition=True, native_exact_size_condition=True,
        anchor_predictor_used=False, lj_guidance=False, rejection_sampling=False,
        post_filters=False, original_R_coordinates_used=False,
        checkpoint_unused_hparams=ignored)
