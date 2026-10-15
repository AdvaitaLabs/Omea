# SPDX-License-Identifier: MIT
# Upstream source: https://github.com/pengxingang/PocketXMol/blob/65488cf635c856101dbe703ac97e2f10f58e005c/LICENSE
# Adapter changes: retained-core inputs, fixed raw budgets; see docs/methods.md.
"""PocketXMol generic maskfill preset: one fixed fragment and one allowed anchor."""
import copy
import hashlib
import json
import sys
import tarfile
from pathlib import Path
import torch
from tasks import ROOT

def make_pocketxmol(args):
    repo=ROOT/'vendor'/'PocketXMol';sys.path.insert(0,str(repo))
    from models.maskfill import PMAsymDenoiser
    from models.sample import sample_loop3,seperate_outputs2
    from process.utils_process import get_input_from_file,extract_pocket
    from utils.dataset import UseDataset
    from utils.misc import make_config
    from utils.transforms import FeaturizeMol,FeaturizePocket,get_transforms,Compose
    from utils.sample_noise import get_sample_noiser
    from utils.reconstruct import reconstruct_from_generated_with_edges
    from torch_geometric.loader import DataLoader
    directory=ROOT/'checkpoints'/'pocketxmol'
    # The release also includes pxm_use; select the checkpoint named by the
    # official sampling preset, rather than relying on an ambiguous glob.
    preset=make_config(str(repo/'configs/sample/pxm.yml'))
    checkpoint=directory/preset.model.checkpoint
    if not checkpoint.is_file():
        archive=ROOT/'checkpoints'/'model_weights.tar.gz'
        digest=hashlib.md5(archive.read_bytes()).hexdigest()
        if digest!='d92f279750f4e4a2ee63fc5b9dbed3ae':raise ValueError('Unverified PocketXMol archive')
        directory.mkdir(exist_ok=True)
        with tarfile.open(archive) as f:f.extractall(directory,filter='data')
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    cfgdir=checkpoint.parent.parent/'train_config';cfgpaths=list(cfgdir.glob('*'))
    if len(cfgpaths)!=1:raise ValueError(f'Ambiguous training configs: {cfgpaths}')
    train=make_config(str(cfgpaths[0]))
    # Same atom/bond encodings as training; centre on the retained fragment.
    train.transforms.featurizer.mol_as_pocket_center=True
    featurizer=FeaturizeMol(train.transforms.featurizer)
    pocket_featurizer=FeaturizePocket(train.transforms.featurizer_pocket)
    dims=dict(num_node_types=featurizer.num_node_types,num_edge_types=featurizer.num_edge_types,pocket_in_dim=pocket_featurizer.feature_dim)
    model=PMAsymDenoiser(config=train.model,**dims).to('cuda')
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict({k[6:]:v for k,v in ckpt['state_dict'].items() if k.startswith('model.')},strict=True)
    model.eval();model.requires_grad_(False)
    template=make_config(str(repo/'configs/sample/examples/linking_fixed_frags_connecting.yml'))
    cache={}
    def generate(task,count):
        if task['case_id'] not in cache:
            config=copy.deepcopy(template)
            n=task['fixed_heavy_atoms'];fixed=list(range(n))
            config.transforms.variable_mol_size.num_atoms_distri.mean.bias=n+task['rgroup_heavy_atoms']
            config.transforms.variable_mol_size.num_atoms_distri.std.bias=0
            config.transforms.variable_mol_size.num_atoms_distri.min=n+task['rgroup_heavy_atoms']
            config.transforms.variable_mol_size.not_remove=fixed
            config.task.transform.preset_partition.grouped_node_p1=[fixed]
            config.task.transform.preset_partition.grouped_anchor_p1=[[task['anchor_in_core']]]
            pocket=extract_pocket(task['protein_path'],task['core_path'],radius=6,criterion='min')
            data,mol=get_input_from_file(task['core_path'],pocket,data_id=task['case_id'],return_mol=True)
            transforms=Compose([pocket_featurizer,featurizer,get_transforms(config.transforms.variable_mol_size),get_transforms(config.task.transform,mode='use')])
            follow=sum([getattr(t,'follow_batch',[]) for t in transforms.transforms],[])
            exclude=sum([getattr(t,'exclude_keys',[]) for t in transforms.transforms],[])
            noiser=get_sample_noiser(config.noise,dims['num_node_types'],dims['num_edge_types'],mode='sample',device='cuda',ref_config=train.noise)
            cache[task['case_id']]=(data,mol,transforms,follow,exclude,noiser,config)
        data,mol,transforms,follow,exclude,noiser,config=cache[task['case_id']]
        dataset=UseDataset(data,n=count,task='maskfill',transforms=transforms)
        loader=DataLoader(dataset,batch_size=count,shuffle=False,num_workers=0,follow_batch=follow,exclude_keys=exclude)
        batch=next(iter(loader)).to('cuda')
        batch,outputs,trajs=sample_loop3(batch,model,noiser,'cuda',is_ar='maskfill')
        generated,_,_=seperate_outputs2(batch,outputs,trajs)
        result=[]
        for output in generated:
            info=featurizer.decode_output(**output);info.update(task='maskfill',data_id=task['case_id'],db='use',key='')
            try:built=reconstruct_from_generated_with_edges(info,in_mol=mol)
            except Exception:built=None
            result.append(built)
        return result
    return generate,dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        steps=100,task='maskfill',part1='fixed',known_anchor='all',native_anchor_condition=True,
        native_exact_size_condition=True,pocket_radius_A=6,pocket_criterion='min',input='fixed_fragment_only',
        training_config_sha256=hashlib.sha256(cfgpaths[0].read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
