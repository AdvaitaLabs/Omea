# SPDX-License-Identifier: MIT
# Upstream source: https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/LICENSE
# Adapter changes: retained-core inputs, fixed raw budgets; see docs/methods.md.
"""Existing DiffSBDD and DiffDec adapters, relocated for the public benchmark."""
import os
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from tasks import ROOT,read_mol
from checkpoint_io import load_checkpoint
def make_diffsbdd(args):
    repo=ROOT/'vendor/DiffSBDD'
    sys.path.insert(0,str(repo))
    import Bio.PDB.Polypeptide as polypeptide
    from Bio.SeqUtils import seq1
    if not hasattr(polypeptide,'three_to_one'):polypeptide.three_to_one=seq1
    import inpaint
    from lightning_modules import LigandPocketDDPM
    checkpoint=ROOT/'checkpoints'/'moad_fullatom_cond.ckpt'
    data=load_checkpoint(checkpoint)
    model=LigandPocketDDPM(**data['hyper_parameters'])
    model.load_state_dict(data['state_dict'],strict=True)
    model=model.to('cuda').eval();inpaint.model=model
    def generate(task,count):
        # Only the retained SDF enters inference. The deleted reference R is never
        # loaded by this adapter. Official pocket selection uses the fixed fragment.
        return inpaint.inpaint_ligand(model,task['protein_path'],count,
            task['core_path'],[task['core_path']],add_n_nodes=task['rgroup_heavy_atoms'],
            center='ligand',sanitize=False,largest_frag=False,relax_iter=0,
            timesteps=50,resamplings=20)
    return generate,dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),timesteps=50,resamplings=20,input='fixed_fragment_only',native_anchor_condition=False)
def make_diffdec(args):
    repo=ROOT/'vendor'/'DiffDec';sys.path.insert(0,str(repo))
    from src.model_single import DDPM
    from src.datasets import collate
    from src import const
    from openbabel import openbabel as ob
    ob.obErrorLog.StopLogging()
    checkpoint=ROOT/'checkpoints'/'diffdec_single.ckpt'
    data=load_checkpoint(checkpoint);params=dict(data['hyper_parameters']);params['torch_device']='cuda'
    model=DDPM(**params);model.load_state_dict(data['state_dict'],strict=True);model=model.to('cuda').eval()
    def generate(task,count):
        core=read_mol(task['core_path']);coords=core.GetConformer().GetPositions()
        protein=PDBParser(QUIET=True).get_structure('',task['protein_path'])[0]
        atoms=[]
        for residue in protein.get_residues():
            if not is_aa(residue.get_resname(),standard=True):continue
            heavy=[a for a in residue.get_atoms() if a.element!='H']
            pos=np.array([a.coord for a in heavy])
            if len(pos) and np.sqrt(((pos[:,None]-coords[None,:])**2).sum(-1)).min()<6:
                atoms+=heavy
        pocket=np.array([a.coord for a in atoms]);nc=len(coords);npoc=len(pocket);nr=10;n=nc+npoc+nr
        symbols=[a.GetSymbol() for a in core.GetAtoms()]+[a.element.capitalize() for a in atoms]+['C']*nr
        h=np.eye(len(const.ATOM2IDX),dtype=np.float32)[[const.ATOM2IDX[s] for s in symbols]]
        anchor=np.zeros(n);anchor[task['anchor_in_core']]=1
        row=dict(uuid=task['case_id'],name=task['case_id'],num_atoms=n)
        arrays=dict(positions=np.concatenate([coords,pocket,np.zeros((nr,3))]),one_hot=h,
            charges=np.array([const.CHARGES[s] for s in symbols]),anchors=anchor,
            scaffold_only_mask=np.r_[np.ones(nc),np.zeros(npoc+nr)],
            pocket_mask=np.r_[np.zeros(nc),np.ones(npoc),np.zeros(nr)],
            scaffold_mask=np.r_[np.ones(nc+npoc),np.zeros(nr)],
            rgroup_mask=np.r_[np.zeros(nc+npoc),np.ones(nr)])
        row.update({k:torch.tensor(v,dtype=torch.float32,device='cuda') for k,v in arrays.items()})
        batch=collate([row]*count)
        chain,mask=model.sample_chain(batch,keep_frames=1)
        center=coords.mean(0) if model.center_of_mass=='scaffold' else coords[task['anchor_in_core']]
        pos=chain[-1,:,:,:3].detach().cpu().numpy()+center
        kinds=chain[-1,:,:,3:].argmax(-1).detach().cpu().numpy()
        mask=mask.squeeze(-1).detach().cpu().numpy().astype(bool)
        result=[]
        for j in range(count):
            indices=[i for i in np.flatnonzero(mask[j]) if const.IDX2ATOM[int(kinds[j,i])]!='#']
            xyz=str(len(indices))+'\nDiffDec\n'+'\n'.join(f'{const.IDX2ATOM[int(kinds[j,i])]} {pos[j,i,0]:.8f} {pos[j,i,1]:.8f} {pos[j,i,2]:.8f}' for i in indices)+'\n'
            converter=ob.OBConversion();converter.SetInAndOutFormats('xyz','sdf');mol=ob.OBMol()
            converter.ReadString(mol,xyz)
            built=Chem.MolFromMolBlock(converter.WriteString(mol),sanitize=False,removeHs=False)
            if built is not None:
                for atom in built.GetAtoms():
                    atom.SetNumRadicalElectrons(0);atom.SetNoImplicit(False)
            result.append(built)
        return result
    return generate,dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),timesteps=model.edm.T,input='fixed_fragment_anchor_and_size_only',native_anchor_condition=True,native_exact_size_condition=False,native_atom_slots=10)
