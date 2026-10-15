# SPDX-License-Identifier: MIT
# Upstream source: https://github.com/OdinZhang/Delete/blob/ba57693822ac82950bf0aa081b788772039373de/LICENSE
# Adapter changes: retained-core inputs, fixed raw budgets; see docs/methods.md.
"""DELETE with exactly one autoregressive path per raw attempt, no branch pool."""
import hashlib
import inspect
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace, MethodType

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

from tasks import ROOT, read_mol


def make_delete(args):
    # DELETE's NumPy 1.x parser uses only this removed integer-type alias.
    if not hasattr(np, 'compat'):
        np.compat = SimpleNamespace(long=np.int64)
    device = getattr(args,'inference_device','cuda')
    repository = ROOT/'vendor/DELETE'
    sys.path.insert(0,str(repository))
    from models.delete import Delete
    from models.sample import add_ligand_atom_to_data
    from utils.protein_ligand import parse_rdmol, read_ply
    from utils.data import ProteinLigandData, torchify_dict
    from utils.transforms import (RefineData,LigandCountNeighbors,FeaturizeLigandAtom,
                                  FeaturizeProteinAtom,AtomComposer,LigandMaskZero)
    from utils.datasets import transform_data

    checkpoint = ROOT/'checkpoints/delete.pt'
    assert hashlib.md5(checkpoint.read_bytes()).hexdigest() == 'fecf2f778bd77efe83574dd461a757b1'
    release = torch.load(checkpoint,map_location='cpu',weights_only=False)
    ligand_features,protein_features = FeaturizeLigandAtom(),FeaturizeProteinAtom()
    composer = AtomComposer(protein_features.feature_dim,ligand_features.feature_dim,release['config'].model.encoder.knn)
    initial_transform = Compose([RefineData(),LigandCountNeighbors(),ligand_features,protein_features])
    native_mask_zero = LigandMaskZero()

    def ordered_core_context(data):
        # Native mask-zero still shuffles context atoms. Restore the supplied
        # core order before assigning the fixed attachment-site index.
        data = native_mask_zero(data)
        permutation = data.context_idx
        order = permutation.argsort()
        assert data.masked_idx.numel() == 0
        for field in ('ligand_context_element', 'ligand_context_feature_full',
                      'ligand_context_pos', 'ligand_context_num_neighbors',
                      'ligand_context_valence', 'ligand_context_num_bonds',
                      'ligand_frontier'):
            setattr(data, field, getattr(data, field)[order])
        data.ligand_context_bond_index = permutation[data.ligand_context_bond_index]
        data.context_idx = torch.arange(len(permutation), device=permutation.device)
        return composer(data)

    context_transform = ordered_core_context
    model = Delete(release['config'].model,num_classes=7,num_bond_types=3,
        protein_atom_feature_dim=protein_features.feature_dim,ligand_atom_feature_dim=ligand_features.feature_dim).to(device)
    model.load_state_dict(release['model'],strict=True);model.eval();model.requires_grad_(False)
    # The released triangle helper hard-codes CUDA for three temporaries.
    # Use the input tensor's device; GPU arithmetic remains identical.
    triangle_source = textwrap.dedent(inspect.getsource(Delete.get_tri_edges))
    assert triangle_source.count(".to('cuda')") == 3
    triangle_source = triangle_source.replace(".to('cuda')", '.to(edge_index_query.device)')
    triangle_namespace = dict(Delete.get_tri_edges.__globals__)
    exec(triangle_source, triangle_namespace)
    model.get_tri_edges = MethodType(triangle_namespace['get_tri_edges'], model)
    current,cache = {},{}
    native_focal,native_query = model.sample_focal,model.query_position

    def single_focal(*values,**keywords):
        keywords['n_samples'] = 1
        return native_focal(*values,**keywords)

    def single_position(h_compose,compose_pos,idx_focal_in_compose,n_samples=-1):
        assert len(idx_focal_in_compose) == 1
        _,mu,sigma,pi = model.pos_predictor(h_compose,idx_focal_in_compose,compose_pos)
        component = pi.multinomial(1).squeeze(-1)
        row = torch.arange(len(mu),device=mu.device)
        xyz = mu[row,component] + torch.randn_like(mu[row,component])*sigma[row,component]
        pdf = model.pos_predictor.get_mdn_probability(mu=mu,sigma=sigma,pi=pi,pos_target=xyz)
        parent = torch.zeros(1,dtype=torch.long,device=compose_pos.device)
        return xyz,pdf,parent,mu,sigma,pi

    def query_with_exit_mask(*values,**keywords):
        element_logits,edge_logits = native_query(*values,**keywords)
        nc,anchor,step = current['nc'],current['anchor'],current['step']
        # Condition bond logits before native categorical sampling. The first
        # atom may attach only at the specified core anchor; subsequent atoms
        # may form bonds only with already generated R atoms.
        forbidden = torch.arange(edge_logits.shape[1],device=edge_logits.device) < nc
        if step == 0: forbidden[anchor] = False
        edge_logits = edge_logits.clone()
        edge_logits[:,forbidden,:] = -50.
        edge_logits[:,forbidden,0] = 0.
        return element_logits,edge_logits

    model.sample_focal = single_focal
    model.sample_position = single_position
    model.query_position = query_with_exit_mask

    def generate(task,count):
        if task['case_id'] not in cache:
            core = read_mol(task['core_path'])
            ply = ROOT/'data/delete_surfaces'/task['case_id']/'pocket.ply'
            if not ply.exists(): raise FileNotFoundError(f'DELETE native protein surface missing: {ply}')
            surface = read_ply(str(ply))
            assert surface['feature'].shape[1] == 4
            data = ProteinLigandData.from_protein_ligand_dicts(
                protein_dict=torchify_dict(surface),ligand_dict=torchify_dict(parse_rdmol(core)))
            data = transform_data(initial_transform(data),context_transform)
            cache[task['case_id']] = (core,data)
        core,initial = cache[task['case_id']]
        nc = core.GetNumAtoms(); result = []
        for _ in range(count):
            data = initial.clone().to(device)
            for step in range(task['rgroup_heavy_atoms']):
                current.update(nc=nc,anchor=task['anchor_in_core'],step=step)
                frozen = torch.arange(nc,device=device)
                anchor = None
                if step == 0:
                    frozen = frozen[frozen != task['anchor_in_core']]
                    anchor = torch.tensor([task['anchor_in_core']],device=device)
                batch = Batch.from_data_list([data])
                proposed = model.sample(compose_feature=batch.compose_feature.float(),compose_pos=batch.compose_pos,
                    idx_ligand=batch.idx_ligand_ctx_in_compose,idx_protein=batch.idx_protein_in_compose,
                    compose_knn_edge_index=batch.compose_knn_edge_index,compose_knn_edge_feature=batch.compose_knn_edge_feature,
                    ligand_context_bond_index=batch.ligand_context_bond_index,
                    ligand_context_bond_type=batch.ligand_context_bond_type,
                    n_samples_pos=1,n_samples_atom=1,frontier_threshold=0,freeze=frozen,anchor=anchor)
                if not proposed[0]: break
                values = [value.cpu() for value in proposed]
                (_,_,_,_,_,position,_,_,_,_,element,_,_,bond_index,bond_type,_) = values
                assert len(position) == len(element) == 1
                data = add_ligand_atom_to_data(data.cpu(),position[0],element[0],bond_index,bond_type)
                data = composer(data).to(device)
            # Construct the sampled graph directly, retaining even early stops.
            # No validity-driven retry, force connection, geometry minimization
            # or likelihood-ranked branch selection occurs.
            data = data.cpu(); total = len(data.ligand_context_element)
            molecule = Chem.RWMol(core)
            for z in data.ligand_context_element[nc:]: molecule.AddAtom(Chem.Atom(int(z)))
            for (left,right),kind in zip(data.ligand_context_bond_index.T.tolist(),data.ligand_context_bond_type.tolist()):
                if left >= right or (left < nc and right < nc): continue
                molecule.AddBond(left,right,[None,Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE][kind])
            coordinates = data.ligand_context_pos.numpy()
            assert np.max(np.abs(coordinates[:nc]-core.GetConformer().GetPositions())) < 1e-4
            conformer = Chem.Conformer(total)
            for index,xyz in enumerate(coordinates): conformer.SetAtomPosition(index,list(map(float,xyz)))
            molecule.RemoveAllConformers();molecule.AddConformer(conformer)
            result.append(molecule.GetMol())
        return result

    return generate,dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_commit='ba57693822ac82950bf0aa081b788772039373de',
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input='native protein surface from protein+retained core; core+anchor+requested R count',
        numpy_compatibility='np.compat.long mapped to np.int64; no arithmetic changes',
        variant='DELETE-singlepath; explicit single-proposal sampler replacing native branching search',
        proposals_per_step=1,focal_samples=1,position_samples=1,element_samples=1,
        position_sampler='one draw from native Gaussian mixture',bond_sampler='native categorical draw',
        exit_mask='non-anchor core bonds forbidden before prediction decoding; core frozen',
        steps='requested R heavy atom count; native early stop retained as failed size attempt',
        candidate_filter=False,branch_pool=False,regeneration=False,guidance=None)
