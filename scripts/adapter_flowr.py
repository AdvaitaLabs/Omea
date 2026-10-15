# SPDX-License-Identifier: MIT
# Upstream source: https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/LICENSE
# Adapter changes: retained-core inputs, fixed raw budgets; see docs/methods.md.
"""FLOWR.root fragment growing through native APIs, with a fixed raw budget.

The official raw sampler is used without candidate resampling or post-filters.
"""
import hashlib
import sys
from pathlib import Path
import torch
import numpy as np
from tasks import ROOT,read_mol

def make_flowr(args):
    repo=ROOT/'vendor'/'flowr_root';sys.path.insert(0,str(repo))
    from flowr.gen.generate_from_pdb import get_args,get_dataset
    from flowr.scriptutil import load_model
    from flowr.gen.generate import generate_ligands_per_target
    import flowr.gen.utils as util
    checkpoint=ROOT/'checkpoints/flowr_root_v2.2.ckpt'
    old=sys.argv
    try:
        sys.argv=['benchmark','--arch','pocket','--pocket_type','holo',
            '--ckpt_path',str(checkpoint),'--gpus','1','--num_workers','0',
            '--batch_cost',str(args.batch_size),'--fragment_growing','--cut_pocket',
            '--pocket_cutoff','6','--max_sample_iter','0','--final_inpaint']
        fargs=get_args()
    finally:sys.argv=old
    loaded=load_model(fargs)
    model,hp,vocab,vcharges,vhybrid,varom,_,_=loaded
    model=model.to('cuda').eval();model.requires_grad_(False)
    cache={}
    def generate(task,count):
        if task['case_id'] not in cache:
            fargs.pdb_file=task['protein_path'];fargs.ligand_file=task['core_path']
            fargs.grow_size=task['rgroup_heavy_atoms'];fargs.sample_n_molecules_per_target=count
            out=ROOT/'results'/args.run/args.model/task['case_id'];out.mkdir(parents=True,exist_ok=True)
            fargs.save_dir=str(out);fargs.save_file='native_unused.sdf'
            core=read_mol(task['core_path']);point=core.GetConformer().GetPositions()[task['anchor_in_core']]
            center=out/'prior_center.xyz'
            center.write_text('1\nRetained anchor only; no removed R coordinates\nC '+' '.join(map(str,point))+'\n')
            fargs.prior_center_file=str(center)
            transform,interpolant=util.load_util(fargs,hp,vocab,vcharges,vhybrid,varom)
            system=util.load_data_from_pdb(fargs,remove_hs=hp['remove_hs'],remove_aromaticity=hp['remove_aromaticity'],canonicalize_conformer=False)
            cache[task['case_id']]=(system,transform,interpolant,int(torch.initial_seed()%(2**32)),0)
        system,transform,interpolant,seed,index=cache[task['case_id']]
        fargs.sample_n_molecules_per_target=count;fargs.seed=seed
        dataset=get_dataset(system,transform,vocab,interpolant,fargs,hp)
        loader=util.get_dataloader(fargs,dataset,interpolant,iter=index)
        output=[]
        for prior,data,_,_ in loader:
            # Native model._generate already reverses centering via system.com.
            # A reference-only round trip verifies the protein coordinate frame.
            refs=model._generate_ligs(data,lig_mask=data['lig_mask'].bool(),scale=model.coord_scale)
            ref=refs[0];core=read_mol(task['core_path'])
            if ref.GetNumAtoms()!=core.GetNumAtoms():raise ValueError('Unexpected reference ligand size after FLOWR preprocessing')
            if np.max(np.abs(ref.GetConformer().GetPositions()-core.GetConformer().GetPositions()))>0.01:
                raise ValueError('FLOWR preprocessing did not restore the input coordinate frame / atom order')
            generated=generate_ligands_per_target(fargs,model,prior=prior,posterior=data,pocket_noise='fix',device='cuda',save_traj=False,iter=str(index))
            output.extend(generated)
        cache[task['case_id']]=(system,transform,interpolant,seed,index+1)
        if len(output)!=count:raise ValueError(f'Native generation changed raw budget: {len(output)} versus {count}')
        return output
    return generate,dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        steps=fargs.integration_steps,task='fragment_growing',pocket_radius_A=6,
        input='fixed fragment, removed heavy-atom count, retained anchor coordinate as prior center',
        native_anchor_condition='prior centre only; attachment assessed by common audit',
        native_exact_size_condition=True,final_inpaint=True,resampling=False,post_filters=False,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
