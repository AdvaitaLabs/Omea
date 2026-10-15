# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Upstream source: https://github.com/GenSI-THUAIR/MolCRAFT/blob/743dfb5198c44ca0cec180822bd54258f9d386f1/MolJO/LICENSE
# Adapter changes: retained-core inputs, fixed raw budgets; see docs/methods.md.
"""Official MolJO inpainting sampler with reference-free placeholder atoms."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa
from io import StringIO
from tasks import ROOT,read_mol

def make_moljo(args):
    sys.path.insert(0,str(ROOT/'vendor'/'MolCRAFT'/'MolJO'))
    from core.models.bfn4sbdd import BFN4SBDDScoreModel,ClassifierScoreModel
    from core.datasets.utils import PDBProtein
    from core.datasets.pl_data import ProteinLigandData,torchify_dict
    import core.utils.transforms as trans
    from core.utils.reconstruct import reconstruct_from_generated
    from checkpoint_io import load_checkpoint
    path=ROOT/'checkpoints'/'moljo_backbone.ckpt';ckpt=load_checkpoint(path)
    hp=ckpt['hyper_parameters'];dynamics=dict(hp['dynamics'])
    # Official sample_guided.py defaults time_emb_dim=0. Released backbone
    # tensor has 13 inputs; checkpoint metadata incorrectly retains 1.
    assert ckpt['state_dict']['dynamics.ligand_atom_emb.weight'].shape[1]==13
    dynamics['time_emb_dim']=0
    model=BFN4SBDDScoreModel(**dynamics)
    # Native EMACallback swaps the backbone to EMA at test start. Classifier
    # modules are loaded separately with state_dict in the official entry point.
    weights=ckpt['ema_state_dict'] if ckpt.get('_ema_state_dict_ready') else ckpt['state_dict']
    model.load_state_dict({k.removeprefix('dynamics.'):v for k,v in weights.items()},strict=True)
    model=model.to('cuda').eval();model.requires_grad_(False)
    model.preserve_unmasked_state=args.model=='MolJO-maskfix'
    classifiers=[];hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest()}
    for name,objective in [('moljo_affinity9.ckpt','vina'),('moljo_sa9.ckpt','sa')]:
        cp=ROOT/'checkpoints'/name;data=load_checkpoint(cp)
        classifier=ClassifierScoreModel(**data['hyper_parameters']['dynamics'])
        classifier.load_state_dict({k.removeprefix('dynamics.'):v for k,v in data['state_dict'].items()},strict=True)
        classifier=classifier.to('cuda').eval();classifier.requires_grad_(False);classifier.prop_name=objective
        classifiers.append(classifier);hashes[name]=hashlib.sha256(cp.read_bytes()).hexdigest()
    scale=float(hp['data']['normalizer_dict']['pos'])
    cache={};batch_index={}
    def generate(task,count):
        if task['case_id'] not in cache:
            core=read_mol(task['core_path']);xyz=core.GetConformer().GetPositions()
            structure=PDBParser(QUIET=True).get_structure('',task['protein_path'])
            selected=set()
            for residue in structure[0].get_residues():
                if not is_aa(residue.get_resname(),standard=True):continue
                positions=np.array([a.coord for a in residue.get_atoms() if a.element!='H'])
                if len(positions) and np.linalg.norm(positions[:,None]-xyz[None,:],axis=-1).min()<6:selected.add(residue.full_id)
            class Pocket(Select):
                def accept_model(self,m):return m.id==0
                def accept_residue(self,r):return r.full_id in selected
                def accept_atom(self,a):return a.element!='H'
            stream=StringIO();writer=PDBIO();writer.set_structure(structure);writer.save(stream,Pocket())
            protein=PDBProtein(stream.getvalue()).to_dict_atom()
            from types import SimpleNamespace
            data=SimpleNamespace(**{'protein_'+k:v for k,v in torchify_dict(protein).items()})
            data=trans.FeaturizeProteinAtom()(data)
            types=[]
            for a in core.GetAtoms():
                key=(a.GetAtomicNum(),a.GetIsAromatic())
                if key not in trans.MAP_ATOM_TYPE_AROMATIC_TO_INDEX:raise ValueError(f'Unsupported core atom: {key}')
                types.append(trans.MAP_ATOM_TYPE_AROMATIC_TO_INDEX[key])
            cache[task['case_id']]=(xyz,np.array(types),data.protein_pos.numpy(),data.protein_atom_feature.numpy())
        xyz,types,pxyz,pfeat=cache[task['case_id']]
        nc=len(xyz);nr=task['rgroup_heavy_atoms'];n=nc+nr;npoc=len(pxyz)
        offset=pxyz.mean(0)/scale
        # Placeholders contain only zeros in normalized protein-centred coordinates.
        fixed_pos=np.concatenate([xyz/scale-offset,np.zeros((nr,3))])
        pos=torch.tensor(np.tile(fixed_pos,(count,1)),dtype=torch.float32,device='cuda')
        kinds=torch.tensor(np.tile(np.r_[types,np.zeros(nr,dtype=int)],count),dtype=torch.long,device='cuda')
        mask=torch.tensor(np.tile(np.r_[np.ones(nc,dtype=bool),np.zeros(nr,dtype=bool)],count),device='cuda')
        b_lig=torch.arange(count,device='cuda').repeat_interleave(n)
        b_pro=torch.arange(count,device='cuda').repeat_interleave(npoc)
        # Guidance computes derivatives w.r.t. coordinates and types. Global
        # inference_mode must be disabled; model parameters remain frozen.
        with torch.no_grad():
            _,chain,_=model.sample(
                protein_pos=torch.tensor(np.tile(pxyz/scale-offset,(count,1)),dtype=torch.float32,device='cuda'),
                protein_v=torch.tensor(np.tile(pfeat,(count,1)),dtype=torch.float32,device='cuda'),
                batch_protein=b_pro,batch_ligand=b_lig,n_nodes=count,sample_steps=200,
                ligand_pos=pos,ligand_v=kinds,ligand_mask=mask,classifiers=classifiers,
                guide_mode='param_naive',pos_grad_weight=50.,type_grad_weight=50.,EPS=0.,W_CFG=0.)
        final=chain[-1];coords=(final[0].detach().cpu().numpy()+offset)*scale
        categories=final[1].detach().argmax(-1).cpu()
        atomic=trans.get_atomic_number_from_index(categories,mode='add_aromatic')
        aromatic=trans.is_aromatic_from_index(categories,mode='add_aromatic')
        out=ROOT/'results'/args.run/args.model/task['case_id'];out.mkdir(parents=True,exist_ok=True)
        index=batch_index.get(task['case_id'],0);batch_index[task['case_id']]=index+1
        diagnostics=dict(coords=coords.tolist(),atomic_numbers=list(map(int,atomic)),aromatic=list(map(bool,aromatic)),n_per_molecule=n,reconstruction_errors=[])
        mols=[]
        for j in range(count):
            sel=slice(j*n,(j+1)*n)
            try:mol=reconstruct_from_generated(coords[sel].astype(np.float64),atomic[sel],aromatic[sel])
            except Exception as e:
                mol=None;diagnostics['reconstruction_errors'].append(dict(sample=j,error=repr(e)))
            mols.append(mol)
        (out/f'raw_batch_{index:03d}.json').write_text(json.dumps(diagnostics))
        return mols
    return generate,dict(checkpoint=str(path),sha256=hashes[path.name],checkpoint_hashes=hashes,
        variant='mask correction' if model.preserve_unmasked_state else 'native zeroing',steps=200,backbone_time_emb_dim=0,guidance='vina_sa',guide_mode='param_naive',pos_grad_weight=50,type_grad_weight=50,
        backbone_weights='EMA' if ckpt.get('_ema_state_dict_ready') else 'state_dict',
        input='fixed fragment, target total count, no removed-R coordinates/types',pocket_radius_A=6,
        native_anchor_condition=False,native_exact_size_condition=True,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
