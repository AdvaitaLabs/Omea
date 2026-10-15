# SPDX-License-Identifier: MIT
# Upstream source: https://github.com/LPDI-EPFL/lddm/blob/8fac7133c29ccdce576eb38bf7d43c09df9cb498/LICENSE.md
# Adapter changes: retained-core inputs, fixed raw budgets; see docs/methods.md.
"""Frozen LDDM fragment sampler; condition only on core, anchor, protein and size."""
import argparse
import hashlib
import sys
from pathlib import Path

import torch

from tasks import ROOT, read_mol


def make_lddm(args):
    repository = ROOT / 'vendor/LDDM'
    sys.path.insert(0, str(repository))
    from lddm.inference.utils import get_model_params, prepare_input
    from lddm.model.lightning import LDDM
    from lddm.data.dataset import ProcessedDataset
    from lddm.constants import bond_encoder

    checkpoint = ROOT / 'checkpoints/lddm.ckpt'
    if hashlib.md5(checkpoint.read_bytes()).hexdigest() != '7f11f7a15bbdf84b1a08a60fd3ad9f2f':
        raise ValueError('LDDM official checkpoint checksum mismatch')
    native_args = argparse.Namespace(checkpoint=str(checkpoint), device='cuda',
        batch_size=args.batch_size, n_steps=100, sampler='ForwardEuler', sampling_noise=5.0)
    # Instantiate explicitly: Lightning 2.5 forwards the release's
    # weights_only=False keyword into LDDMConfig instead of consuming it.
    model = LDDM(**get_model_params(checkpoint, native_args))
    released = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.on_load_checkpoint(released)
    model.load_state_dict(released['state_dict'], strict=True)
    del released
    model.setup(stage='generation')
    model.batch_size = model.eval_batch_size = args.batch_size
    model.eval().to('cuda')
    model.all_masked = True
    # Retain the checkpoint's native extra virtual slots for fragment design.
    # Native docking disables these, but docking fixes all atom labels, unlike
    # this task. NOATOM decoding and any wrong final sizes remain raw failures.
    resize = model.change_ligand_size_and_keep_masked_values
    current = {}

    def resize_with_exit_mask(ligand, num_nodes):
        ligand = resize(ligand, num_nodes)
        nc, anchor = current['nc'], current['anchor']
        edges = ligand['bonds']
        graph = ligand['bond_mask']
        offsets = torch.cat([torch.zeros(1, device=num_nodes.device, dtype=num_nodes.dtype),
                             num_nodes.cumsum(0)[:-1]])
        left = edges[0] - offsets[graph]
        right = edges[1] - offsets[graph]
        forbidden = (((left < nc) & (right >= nc) & (left != anchor)) |
                     ((right < nc) & (left >= nc) & (right != anchor)))
        ligand['known_e'][forbidden] = True
        ligand['bond_one_hot'][forbidden] = 0
        ligand['bond_one_hot'][forbidden, bond_encoder['NOBOND']] = 1
        return ligand

    model.change_ligand_size_and_keep_masked_values = resize_with_exit_mask

    def generate(task, count):
        core = read_mol(task['core_path'])
        current.update(nc=core.GetNumAtoms(), anchor=task['anchor_in_core'])
        native_args.protein = task['protein_path']
        native_args.ref_ligand = native_args.ligand = task['core_path']
        native_args.mode = 'design'
        ligand, pocket = prepare_input(native_args, model)
        data = ProcessedDataset.collate_fn([{'ligand': ligand, 'pocket': pocket}])
        data = {key: value.to('cuda') for key, value in data.items()}
        molecules, _, _ = model.sample(data, n_samples=count,
            num_nodes=current['nc'] + task['rgroup_heavy_atoms'], timesteps=100)
        return molecules

    config = dict(checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_commit='8fac7133c29ccdce576eb38bf7d43c09df9cb498',
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timesteps=100, sampler='ForwardEuler', sampling_noise=5.0,
        input='protein_retained_core_anchor_R_heavy_atom_count_only',
        core_mask='native known_x / known_h / known_e',
        exit_mask='native known_e=NOBOND for all non-anchor core-to-new edges',
        anchor_bond='model predicted; exact single-bond attachment audited',
        virtual_nodes=model.virtual_nodes, additional_virtual_slots=model.add_virtual_max,
        native_exact_size_condition=False, candidate_filter=False, regeneration=False)
    return generate, config
