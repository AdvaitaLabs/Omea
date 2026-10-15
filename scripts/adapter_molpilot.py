# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Upstream source: https://github.com/GenSI-THUAIR/MolCRAFT/blob/743dfb5198c44ca0cec180822bd54258f9d386f1/MolPilot/LICENSE
# Adapter changes: retained-core inputs, fixed raw budgets; see docs/methods.md.
"""MolPilot BFN sampler with explicit retained-core inpainting, no property guidance."""
import hashlib
import inspect
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem

from tasks import ROOT, read_mol


def make_molpilot(args):
    sys.path.insert(0, str(ROOT/'vendor/MolCRAFT/MolPilot'))
    from core.models.bfn4sbdd import BFN4SBDDScoreModel
    from core.datasets.utils import PDBProtein
    from core.datasets.pl_data import torchify_dict
    import core.utils.transforms as trans

    path = ROOT/'checkpoints/molpilot.ckpt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    hp = checkpoint['hyper_parameters']
    model = BFN4SBDDScoreModel(**hp['dynamics'])
    assert not checkpoint.get('_ema_state_dict_ready', False)
    model.load_state_dict({k.removeprefix('dynamics.'):v for k,v in checkpoint['state_dict'].items()}, strict=True)
    model = model.to('cuda').eval()
    model.requires_grad_(False)
    schedule_path = ROOT/'checkpoints/molpilot_schedule.pt'
    schedule = torch.as_tensor(torch.load(schedule_path, map_location='cpu', weights_only=False),device='cuda')
    # Match the official SBDD4Train test_step normalization of grid indices.
    schedule = schedule / schedule.max()
    assert schedule.shape == (101,2) and float(schedule.min()) >= 0 and float(schedule.max()) == 1
    scale = float(hp['data']['normalizer_dict']['pos'])
    current, cache = {}, {}
    original_forward = model.forward
    signature = inspect.signature(original_forward)

    def forward_with_core(*values, **keywords):
        bound = signature.bind(*values, **keywords)
        bound.apply_defaults()
        data = bound.arguments
        core_mask = current['core_mask']
        # Sample the native Bayesian forward distribution of known core values.
        # Unknown R atoms and bonds retain the original unconditional priors.
        coord_time = data['t_pos'] if data['t_pos'] is not None else data['time']
        known_mu, _ = model.continuous_var_bayesian_update(
            coord_time[core_mask], model.sigma1_coord, current['core_xyz'])
        data['mu_pos_t'] = data['mu_pos_t'].clone()
        data['mu_pos_t'][core_mask] = known_mu
        data['theta_h_t'] = data['theta_h_t'].clone()
        data['theta_h_t'][core_mask] = model.discrete_var_bayesian_update(
            data['time'][core_mask], model.beta1, current['core_onehot'], model.num_classes)
        known_edge_mask = current['known_edge_mask']
        edge_time = data['time'][data['ligand_bond_index'][0]][known_edge_mask]
        data['theta_bond_t'] = data['theta_bond_t'].clone()
        data['theta_bond_t'][known_edge_mask] = model.discrete_var_bayesian_update(
            edge_time, model.beta1_bond, current['known_edge_onehot'], model.num_bond_classes)
        output = original_forward(**data)
        # Clamp predicted clean values before both sampler feedback and the
        # native self-conditioning pass. This is conditioning, not a repair.
        output['final_ligand_pos'] = output['final_ligand_pos'].clone()
        output['final_ligand_pos'][core_mask] = current['core_xyz']
        output['final_ligand_v'] = output['final_ligand_v'].clone()
        output['final_ligand_v'][core_mask] = torch.where(current['core_onehot'].bool(), 0., -50.)
        output['final_ligand_e'] = output['final_ligand_e'].clone()
        output['final_ligand_e'][known_edge_mask] = torch.where(current['known_edge_onehot'].bool(), 0., -50.)
        return output

    model.forward = forward_with_core

    def generate(task, count):
        if task['case_id'] not in cache:
            core = read_mol(task['core_path'])
            xyz = core.GetConformer().GetPositions()
            structure = PDBParser(QUIET=True).get_structure('',task['protein_path'])
            selected = set()
            for residue in structure[0].get_residues():
                if not is_aa(residue.get_resname(),standard=True): continue
                points = np.asarray([a.coord for a in residue.get_atoms() if a.element != 'H'])
                if len(points) and np.linalg.norm(points[:,None]-xyz[None,:],axis=-1).min() < 10:
                    selected.add(residue.full_id)
            class Pocket(Select):
                def accept_model(self,m): return m.id == 0
                def accept_residue(self,r): return r.full_id in selected
                def accept_atom(self,a): return a.element != 'H'
            stream = StringIO(); writer = PDBIO(); writer.set_structure(structure); writer.save(stream,Pocket())
            protein = PDBProtein(stream.getvalue()).to_dict_atom()
            data = SimpleNamespace(**{'protein_'+k:v for k,v in torchify_dict(protein).items()})
            data = trans.FeaturizeProteinAtom()(data)
            categories = [trans.MAP_ATOM_TYPE_AROMATIC_TO_INDEX[(a.GetAtomicNum(),a.GetIsAromatic())]
                          for a in core.GetAtoms()]
            kekule = Chem.Mol(core); Chem.Kekulize(kekule,clearAromaticFlags=True)
            known_bonds = np.zeros((core.GetNumAtoms(),core.GetNumAtoms()),dtype=np.int64)
            for bond in kekule.GetBonds():
                left,right = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
                known_bonds[left,right] = known_bonds[right,left] = int(bond.GetBondTypeAsDouble())
            cache[task['case_id']] = (core,xyz,categories,known_bonds,data.protein_pos.numpy(),data.protein_atom_feature.numpy())
        core,xyz,categories,core_bonds,protein_xyz,protein_v = cache[task['case_id']]
        nc,nr = len(xyz),task['rgroup_heavy_atoms']; n = nc+nr
        offset = protein_xyz.mean(0)/scale
        batch_ligand = torch.arange(count,device='cuda').repeat_interleave(n)
        batch_protein = torch.arange(count,device='cuda').repeat_interleave(len(protein_xyz))
        # Preserve the native ordering of fully connected directed bond edges.
        src = np.tile(np.arange(n),n); dst = np.repeat(np.arange(n),n)
        src,dst = src[src != dst],dst[src != dst]
        edge_index = np.concatenate([np.stack([src,dst])+j*n for j in range(count)],axis=1)
        core_mask = torch.tensor(np.tile(np.arange(n) < nc,count),device='cuda')
        current['core_mask'] = core_mask
        current['core_xyz'] = torch.tensor(np.tile(xyz/scale-offset,(count,1)),dtype=torch.float32,device='cuda')
        current['core_onehot'] = torch.nn.functional.one_hot(torch.tensor(categories*count,device='cuda'),model.num_classes).float()
        known = (src < nc) & (dst < nc)
        forbidden = (((src < nc) & (dst >= nc) & (src != task['anchor_in_core'])) |
                     ((dst < nc) & (src >= nc) & (dst != task['anchor_in_core'])))
        known |= forbidden
        kinds = np.zeros(len(src),dtype=np.int64)
        is_core = (src < nc) & (dst < nc)
        kinds[is_core] = core_bonds[src[is_core],dst[is_core]]
        current['known_edge_mask'] = torch.tensor(np.tile(known,count),device='cuda')
        current['known_edge_onehot'] = torch.nn.functional.one_hot(
            torch.tensor(np.tile(kinds[known],count),device='cuda'),model.num_bond_classes).float()
        _,chain,_ = model.sample(
            protein_pos=torch.tensor(np.tile(protein_xyz/scale-offset,(count,1)),dtype=torch.float32,device='cuda'),
            protein_v=torch.tensor(np.tile(protein_v,(count,1)),dtype=torch.float32,device='cuda'),
            batch_protein=batch_protein,batch_ligand=batch_ligand,
            batch_ligand_bond=torch.arange(count,device='cuda').repeat_interleave(len(src)),
            ligand_bond_index=torch.tensor(edge_index,device='cuda'),n_nodes=count,
            include_protein=True,sample_steps=100,t_steps=schedule,desc='MolPilot-inpaint')
        final = chain[-1]
        positions = (final[0].cpu().numpy()+offset)*scale
        atomic = trans.get_atomic_number_from_index(final[1].argmax(-1).cpu(),mode='add_aromatic')
        bond_types = final[3].argmax(-1).cpu().numpy()
        result = []
        # Decode the predicted bond graph directly. The known core already was
        # clamped throughout sampling, including its clean outputs.
        for j in range(count):
            molecule = Chem.RWMol(core)
            for z in atomic[j*n+nc:(j+1)*n]: molecule.AddAtom(Chem.Atom(int(z)))
            predicted = bond_types[j*len(src):(j+1)*len(src)]
            for left,right,kind in zip(src,dst,predicted):
                if left >= right or (left < nc and right < nc) or kind == 0: continue
                molecule.AddBond(int(left),int(right),[None,Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE][kind])
            conformer = Chem.Conformer(n)
            for index,point in enumerate(positions[j*n:(j+1)*n]): conformer.SetAtomPosition(index,list(map(float,point)))
            molecule.RemoveAllConformers(); molecule.AddConformer(conformer)
            result.append(molecule.GetMol())
        return result

    return generate, dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        schedule_sha256=hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_commit='743dfb5198c44ca0cec180822bd54258f9d386f1',timesteps=100,
        schedule='official optimized_path_rect_closest.pt',backbone_weights='state_dict',
        input='protein_retained_core_anchor_R_heavy_atom_count_only',pocket_radius_A=10,
        conditioning='Bayesian forward known core values, clean output clamp at every forward including self-conditioning',
        exit_mask='NOBOND at all non-anchor core-to-new edges',
        variant='MolPilot-inpaint inference adapter; native checkpoint has no partial-core sampler',
        guidance=None,candidate_filter=False,regeneration=False)
