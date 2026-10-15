# SPDX-License-Identifier: MIT
# Upstream source: https://github.com/QiaoyuHu89/DiffGui/blob/f2f90b560ad32db9d584a3b7b153420dc1e8d258/LICENSE
# Adapter changes: retained-core inputs, fixed raw budgets; see docs/methods.md.
"""Frozen official DiffGui frag_cond inference; no success replenishment."""
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from tasks import ROOT, read_mol


def make_diffgui(args):
    # Compatibility aliases used by the official NumPy 1.x featurizer.
    import types
    if 'int' not in np.__dict__: np.int = int
    if 'long' not in np.__dict__: np.long = np.int64
    if 'compat' not in np.__dict__: np.compat = types.SimpleNamespace(long=np.int64)
    repo = ROOT / 'vendor/DiffGui'; sys.path.insert(0, str(repo))
    from models.model import DiffGui
    from models.bond_predictor import BondPredictor
    from utils.data import PDBProtein, parse_lig_file
    from utils.dataset import ProteinLigandData, to_torch_dict
    from utils.transforms import FeatureComplexWithFrag, make_data_placeholder
    from utils.reconstruct import reconstruct_from_generated_with_edges
    from torch_geometric.data import Batch
    from easydict import EasyDict
    from Bio.PDB.Polypeptide import is_aa
    checkpoint = ROOT / 'checkpoints/diffgui_trained.pt'
    bond_checkpoint = ROOT / 'checkpoints/diffgui_bond_trained.pt'
    with torch.serialization.safe_globals([EasyDict]):
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    featurizer = FeatureComplexWithFrag(ckpt['config'].data.transform.ligand_atom_mode, sample=True)
    dims = dict(protein_node_types=featurizer.protein_feat_dim,
                ligand_node_types=featurizer.atom_feat_dim,
                num_edge_types=featurizer.bond_feat_dim)
    model = DiffGui(config=ckpt['config'].model, **dims)
    model.load_state_dict(ckpt['model'], strict=True)
    model = model.to('cuda').eval().requires_grad_(False)
    patched_sampler_sha256 = None
    if args.model == 'DiffGui-maskfix':
        import inspect, textwrap, types
        import models.model as model_module
        original = textwrap.dedent(inspect.getsource(DiffGui.sample_frag))
        old = 'frag_halfedge_mask = get_fragment_mask(halfedge_batch, frag_halfedge_batch)'
        new = 'frag_halfedge_mask = frag_node_mask[halfedge_index[0]] & frag_node_mask[halfedge_index[1]]'
        assert original.count(old) == 1
        fixed = original.replace(old, new)
        old = 'pred_ligand_pos = pred_ligand_pos + offset[ligand_batch]'
        new = '''pred_ligand_pos = pred_ligand_pos + offset[ligand_batch]
    # Restore the conditioned variables at the final denoising boundary.
    pred_ligand_pos[frag_node_mask] = frag_pos + offset[frag_batch]
    pred_ligand_node[frag_node_mask] = index_to_log_onehot(frag_node, self.ligand_node_types)
    pred_ligand_halfedge[frag_halfedge_mask] = index_to_log_onehot(frag_halfedge_type, self.num_edge_types)'''
        assert fixed.count(old) == 1
        fixed = fixed.replace(old, new)
        namespace = dict(vars(model_module))
        exec(compile(fixed, 'DiffGui_maskfix_sampler', 'exec'), namespace)
        model.sample_frag = types.MethodType(namespace['sample_frag'], model)
        patched_sampler_sha256 = hashlib.sha256(fixed.encode()).hexdigest()
        dest = ROOT / 'results' / args.run / args.model / 'source'
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'sample_frag_maskfix.py').write_text(fixed)
    with torch.serialization.safe_globals([EasyDict]):
        bc = torch.load(bond_checkpoint, map_location='cpu', weights_only=True)
    bond_model = BondPredictor(config=bc['config']['model'], **dims)
    bond_model.load_state_dict(bc['model'], strict=True)
    bond_model = bond_model.to('cuda').eval().requires_grad_(False)
    cache = {}

    def generate(task, count):
        if task['case_id'] not in cache:
            core = read_mol(task['core_path'])
            xyz = core.GetConformer().GetPositions()
            lines = Path(task['protein_path']).read_text().splitlines()
            atomlines = [l for l in lines if l.startswith('ATOM') and is_aa(l[17:20], standard=True)]
            keep = set()
            for line in atomlines:
                pos = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                if np.linalg.norm(xyz-pos, axis=1).min() < 10:
                    keep.add((line[21], line[22:27]))
            pocket_block = '\n'.join(l for l in atomlines if (l[21], l[22:27]) in keep) + '\nEND\n'
            protein = PDBProtein(pocket_block, mode='block').to_dict_atom()
            if not len(protein['pos']): raise ValueError('DiffGui pocket has no atoms')
            fragment = parse_lig_file(task['core_path'])
            # The retained core also fills the unused ligand bookkeeping fields.
            data = ProteinLigandData.protein_ligand_dicts(
                protein_dict=to_torch_dict(protein), ligand_dict=to_torch_dict(fragment),
                frag_dict=to_torch_dict(fragment))
            cache[task['case_id']] = featurizer(data)
        batch = Batch.from_data_list([cache[task['case_id']].clone() for _ in range(count)],
                                     follow_batch=featurizer.follow_batch).to('cuda')
        n = task['fixed_heavy_atoms'] + task['rgroup_heavy_atoms']
        holder = make_data_placeholder([n] * count, device='cuda')
        bn, hi, bh = holder['batch_node'], holder['halfedge_index'], holder['batch_halfedge']
        labels = torch.tensor([[2., 100., 1., .8, 12.]], device='cuda').repeat(count, 1)
        out = model.sample_frag(n_graphs=count,
            protein_node=batch.protein_atom_feat.float(), protein_pos=batch.protein_pos,
            protein_batch=batch.protein_element_batch,
            frag_node=batch.frag_atom_feat_full, frag_pos=batch.frag_pos,
            frag_batch=batch.frag_element_batch,
            frag_halfedge_type=batch.frag_halfedge_type,
            frag_halfedge_index=batch.frag_halfedge_index,
            frag_halfedge_batch=batch.frag_halfedge_type_batch,
            ligand_batch=bn, halfedge_index=hi, halfedge_batch=bh,
            batch_lab=labels, gui_strength=3., bond_predictor=bond_model,
            guidance=['uncertainty', 1.e-4], gen_mode='frag_cond')
        nodes, positions, bonds = [v.detach().cpu().numpy() for v in out['pred']]
        hi, bn, bh = [v.cpu().numpy() for v in (hi, bn, bh)]
        result = []
        for i in range(count):
            ni, bi = bn == i, bh == i
            info = featurizer.decode_output(nodes[ni], positions[ni], bonds[bi], hi[:, bi] - i*n)
            try: result.append(reconstruct_from_generated_with_edges(info, add_edge=None))
            except Exception: result.append(None)
        return result

    return generate, dict(checkpoint=str(checkpoint),
        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        bond_checkpoint=str(bond_checkpoint), bond_checkpoint_sha256=hashlib.sha256(bond_checkpoint.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        sampler='native sample_frag, frag_cond', timesteps=model.num_timesteps,
        pocket_radius_A=10, native_anchor_condition=False, native_exact_size_condition=True,
        input='retained core chemistry and pose; R heavy-atom count only',
        guidance='official default uncertainty 1e-4, classifier-free scale 3',
        property_targets=dict(logp=2., tpsa=100., sa=1., qed=.8, aff=12.),
        reference_property_labels_used=False, score_search=False, success_replenishment=False,
        diagnostic_variant=args.model == 'DiffGui-maskfix', patched_sampler_sha256=patched_sampler_sha256,
        sampler_changes=['correct fixed-core halfedge mask', 'restore conditioned core in final output'] if patched_sampler_sha256 else [])
