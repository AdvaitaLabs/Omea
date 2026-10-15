# SPDX-License-Identifier: Apache-2.0
# Upstream source: https://github.com/pfizer-opensource/pilot-sbdd/blob/5b7e668ea88baac175536d804599e59236b60afd/LICENSE
# Adapter changes: retained-core inputs, fixed raw budgets; see docs/methods.md.
"""Official kinase cutable coupled checkpoints, with reference-free R inpainting."""
import hashlib
import importlib
import inspect
import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace, MethodType

import torch

from tasks import ROOT,read_mol


def make_pfizer(args):
    sys.path.insert(0,str(ROOT/'vendor/PfizerCoupled'))
    from e3mol.experiments.data.datainfo import ATOM_ENCODER,ATOM_DECODER
    from e3mol.experiments.inference.utils import create_batch_from_pl_files
    kind = 'flow' if args.model.endswith('-flow') else 'diffusion'
    device = getattr(args,'inference_device','cuda')
    path = ROOT/f'checkpoints/pfizer/ckpts/kinodata3d/{kind}/cutable/best_valid.ckpt'
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    hp = dict(checkpoint['hyper_parameters'])
    state = {key.removeprefix('model.'):value for key,value in checkpoint['state_dict'].items() if key.startswith('model.')}
    assert len(state)>0 and 'atoms_prior' in state and 'bonds_prior' in state and 'charges_prior' in state
    # Use the released checkpoint's exact learned-dataset marginal priors,
    # including the categorical kernel buffers. No training dataset or fitted
    # benchmark statistics enter generation. Size is supplied by this task.
    info=SimpleNamespace(name='kinodata3d',atom_encoder=ATOM_ENCODER,atom_decoder=ATOM_DECODER,
        atom_types=state['atoms_prior'].clone(),edge_types=state['bonds_prior'].clone(),
        charges_marginals=state['charges_prior'].clone(),charge_offset=2,
        collapse_charges=torch.arange(-2,4),num_charge_classes=len(state['charges_prior']))
    for name in ('numHs','hybridization'):
        if name+'_prior' in state:setattr(info,name,state[name+'_prior'].clone())
    assert hp['num_atom_types']==len(ATOM_ENCODER)
    module = importlib.import_module('e3mol.experiments.flowmodel' if kind=='flow' else 'e3mol.experiments.diffmodel')
    model = module.EQGATDiffPL(hp,info,ckpt_path=None)
    model.load_state_dict(state,strict=True)
    model=model.to(device).eval();model.requires_grad_(False)
    diffusion_source = None
    if kind == 'diffusion':
        # The released node-time inpainting path clamps core atom identities,
        # positions and bonds, but omits their formal charges. Apply the same
        # known-core condition at initialization and after every native step.
        # This never changes a generated graph after sampling.
        diffusion_source = textwrap.dedent(inspect.getsource(module.EQGATDiffPL.reverse_sampling_node_level_t))
        initial = ('    atom_types_ligand_initial = F.one_hot(\n'
                   '        atom_types_ligand_initial, self.num_atom_types\n'
                   '    ).float()\n')
        assert diffusion_source.count(initial) == 1
        charge_initial = ('    charge_types_ligand_initial = F.one_hot(\n'
                          '        pocket_data.charges.to(device) + 2, self.num_charge_classes\n'
                          '    ).float()\n')
        diffusion_source = diffusion_source.replace(initial, initial + charge_initial)
        atom_clamp = (r'(?m)^(?P<indent> +)atom_types_ligand = \(\n'
                      r'(?P=indent)    atom_types_ligand \* \(1\.0 - lig_inpaint_mask_f\)\n'
                      r'(?P=indent)    \+ atom_types_ligand_initial \* lig_inpaint_mask_f\n'
                      r'(?P=indent)\)')
        def add_charge_clamp(match):
            indent = match.group('indent')
            return match.group(0) + '\n' + indent + 'charge_types_ligand = (\n' + indent + '    charge_types_ligand * (1.0 - lig_inpaint_mask_f)\n' + indent + '    + charge_types_ligand_initial * lig_inpaint_mask_f\n' + indent + ')'
        diffusion_source, clamp_count = re.subn(atom_clamp, add_charge_clamp, diffusion_source)
        assert clamp_count == 2
        namespace = dict(module.EQGATDiffPL.reverse_sampling_node_level_t.__globals__)
        exec(diffusion_source, namespace)
        model.reverse_sampling_node_level_t = MethodType(namespace['reverse_sampling_node_level_t'], model)
    current={}
    native_edge_mask=module.get_edge_mask_inpainting

    def mask_with_exit(edge_index,edge_attr,fixed_nodes_indices):
        _,mask=native_edge_mask(edge_index,edge_attr,fixed_nodes_indices)
        nc,n,anchor=current['nc'],current['n'],current['anchor']
        left,right=edge_index[0]%n,edge_index[1]%n
        forbidden=(((left < nc)&(right >= nc)&(left != anchor)) |
                   ((right < nc)&(left >= nc)&(right != anchor)))
        mask=mask | forbidden
        return torch.where(mask)[0],mask
    module.get_edge_mask_inpainting=mask_with_exit
    if kind == 'diffusion':
        # The charge-clamped sampler above was exec'd with a COPY of the
        # module globals. Bind the declared exit mask to that copy as well.
        # This fixes inference conditioning; it does not repair final graphs.
        sampler_globals = model.reverse_sampling_node_level_t.__func__.__globals__
        sampler_globals['get_edge_mask_inpainting'] = mask_with_exit
        assert sampler_globals['get_edge_mask_inpainting'] is module.get_edge_mask_inpainting

    def generate(task,count):
        core=read_mol(task['core_path']);nc=core.GetNumAtoms();nr=task['rgroup_heavy_atoms']
        current.update(nc=nc,n=nc+nr,anchor=task['anchor_in_core'])
        batch=create_batch_from_pl_files(pdb_file=task['protein_path'],sdf_file=task['core_path'],
            batch_size=count,pocket_cutoff=5.,removeHs_ligand_post=True,removeHs_pocket_post=True,
            inpainting=True,keep_ids=list(range(nc)),anchor_idx=[task['anchor_in_core']],
            min_nodes_bias=nr,max_nodes_bias=nr,atom_encoder=ATOM_ENCODER).to(device)
        assert torch.equal(batch.batch.bincount(),torch.full((count,),nc+nr,device=device))
        batch.n_variable=torch.full((count,),nr,dtype=torch.long,device=device)
        if kind=='flow':
            sampled=model.reverse_sampling_node_level_t(pocket_data=batch,device=device,N=getattr(args,'smoke_steps',100),
                cat_noise=True,pos_noise=False,sanitize=False,save_traj=False,
                clash_guidance=False,score_dynamics=True,score_scale=.5,dt_pow=.5,pos_context_noise=0.)
        else:
            sampled=model.reverse_sampling_node_level_t(pocket_data=batch,device=device,
                sanitize=False,build_obabel_mol=False,relax_mol=False,clash_guidance=False,
                save_traj=False,every_k_step=(max(1,hp['timesteps']//args.smoke_steps) if hasattr(args,'smoke_steps') else 1))
        molecules=sampled[0]
        assert len(molecules)==count
        return [m.rdkit_mol for m in molecules]

    return generate,dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_commit='5b7e668ea88baac175536d804599e59236b60afd',
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset='Kinodata3D; kinase-domain checkpoint, not a general-domain claim',
        checkpoint_selection='official cutable example selected before any benchmark scores',
        prior_source='exact checkpoint buffers, strict model state_dict load',
        input='protein_retained_core_anchor_R_heavy_atom_count_only',pocket_radius_A=5.,
        unknown_R_initialization='native freshly sampled priors; core-only SDF plus nr new empty slots',
        exit_mask='native edge inpainting mask extended with non-anchor NOBOND edges',
        timesteps=100 if kind=='flow' else hp['timesteps'],
        diffusion_charge_inpainting=(kind=='diffusion'),
        diffusion_exit_mask_global_binding_verified=(kind=='diffusion'),
        release_adapter_version='github_release_v1',
        diffusion_sampler_sha256=(hashlib.sha256(diffusion_source.encode()).hexdigest() if diffusion_source else None),
        native_score_dynamics=(kind=='flow'),external_property_guidance=False,
        candidate_filter=False,regeneration=False)
