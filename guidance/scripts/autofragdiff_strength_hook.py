"""Single-trajectory strength-controlled coordinate / soft element guidance.

Gradient clipping precedes strength multiplication. Elements use the existing
detached legal-counterfactual expected-score gradient, NOT an exact GNINA
derivative or CFG. No terminal editing, hard locks, trajectory branching,
winner selection or replacement draws. Frozen checkpoint and native posterior.
"""
import hashlib
import json
import types
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from autofragdiff_local_regeneration import LocalRegenerator, sample_restart, NUMBERS
from element_guidance_objective import score_table, feature_gradient
from element_physics_guard import PhysicsGuard
from fixed_core_reconstruction import preserve_core_graph
from gnina_gradient_client import GradientClient
from gnina_element_client import ElementClient
from reanalyze_geometry_stereo import normalize
from tasks import read_mol


def scaled_gradient(gradient, strength, gradient_scale, ncore):
    """Cap g / fixed_scale at unit norm, THEN multiply by lambda.

    For a given gradient, changing lambda scales the actual correction linearly.
    lambda_x is maximum x0 displacement in A; lambda_h is maximum x0 feature L2.
    """
    g = np.asarray(gradient, dtype=np.float64).copy()
    assert gradient_scale > 0 and strength >= 0 and np.isfinite(g).all()
    g[:ncore] = 0
    unit = g / gradient_scale
    norms = np.linalg.norm(unit, axis=1, keepdims=True)
    clipped = unit * np.minimum(1., 1. / np.maximum(norms, 1e-12))
    delta = strength * clipped
    assert np.all(delta[:ncore] == 0)
    return delta, dict(gradient_norm=float(np.linalg.norm(g)),
                       gradient_max_atom_norm=float(np.linalg.norm(g, axis=1).max()),
                       pre_strength_clipped_atoms=int((norms[ncore:] > 1).sum()),
                       requested_max_atom_change=float(np.linalg.norm(delta, axis=1).max()))


def make_noise(seeds, nodes, timesteps=500):
    """Independent CPU RNG per trajectory: stable across batch width and arms."""
    noise = []
    for seed in seeds:
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        x = torch.randn((timesteps + 2, nodes, 3), generator=generator)
        h = torch.randn((timesteps + 2, nodes, 10), generator=generator)
        noise.append(torch.cat([x, h], dim=-1))
    return torch.stack(noise, dim=1)


class NoiseTape:
    def __init__(self, noise):
        self.noise = noise.to('cuda'); self.position = 0
        self.sha256 = hashlib.sha256(noise.numpy().tobytes()).hexdigest()

    def draw(self, n_samples, n_nodes, mask):
        assert self.position < len(self.noise), 'Unexpected additional noise draw'
        value = self.noise[self.position]
        assert value.shape == (n_samples, n_nodes, 13)
        self.position += 1
        return value * mask


class StrengthHook:
    def __init__(self, edm, reconstruct):
        self.edm, self.reconstruct = edm, reconstruct
        self.original = edm.sample_p_zs_given_zt_only_extension
        self.coordinate_client = self.element_client = None
        self.configuration = dict(lambda_x=0., lambda_h=0., last_fraction=.2, interval=5, schedule='flat')
        owner = self
        def wrapper(_edm, *pos, **kw): return owner.step(*pos, **kw)
        edm.sample_p_zs_given_zt_only_extension = types.MethodType(wrapper, edm)
        self.batch = 0

    def configure(self, configuration, output):
        self.configuration = dict(configuration); self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if configuration['lambda_x'] and self.coordinate_client is None:
            self.coordinate_client = GradientClient(self.output/'coordinate_worker.log')
        if configuration['lambda_h'] and self.element_client is None:
            self.element_client = ElementClient(self.output/'element_worker.log')

    def context(self, task, center, core, attempts, seeds):
        self.task, self.center, self.core = task, center, core
        self.attempts, self.seeds = list(attempts), list(seeds)
        self.guard = PhysicsGuard(task['protein_path']) if self.configuration['lambda_h'] else None
        self.batch += 1

    def close(self):
        if self.coordinate_client: self.coordinate_client.close()
        if self.element_client: self.element_client.close()

    def step(self, s, t, z_t, pocket_xh, scaffold_mask, extension_mask, anchors, pocket_anchors, pocket_mask):
        e, config = self.edm, self.configuration
        step = int(round(float(s[0, 0]) * e.T))
        start = round(e.T * config['last_fraction'])
        if not(config['lambda_x'] or config['lambda_h']) or step >= start or step % config['interval']:
            return self.original(s, t, z_t, pocket_xh, scaffold_mask, extension_mask, anchors, pocket_anchors, pocket_mask)
        gamma_s, gamma_t = e.gamma(s), e.gamma(t)
        sigma2_ts, sigma_ts, alpha_ts = e.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_t)
        sigma_s, sigma_t = e.sigma(gamma_s, z_t), e.sigma(gamma_t, z_t)
        alpha_t = e.alpha(gamma_t, z_t)
        epsilon = e.dynamics.forward(xh=z_t, pocket_xh=pocket_xh, t=t, extension_mask=extension_mask,
                                     scaffold_mask=scaffold_mask, anchors=anchors, pocket_anchors=pocket_anchors,
                                     pocket_mask=pocket_mask) * extension_mask.unsqueeze(-1)
        predicted = e.compute_x_pred(epsilon, z_t, gamma_t)
        predicted = predicted * extension_mask.unsqueeze(-1) + z_t * scaffold_mask.unsqueeze(-1)
        x, h = e.unnormalize(predicted[:, :, :3], predicted[:, :, 3:])
        xyz = x.cpu().numpy() + self.center
        features = h.cpu().numpy(); kinds = features.argmax(-1)
        dx, dh = np.zeros_like(xyz), np.zeros_like(features)
        nc = self.core.GetNumAtoms(); records = []
        weight = sigma2_ts / alpha_ts * alpha_t / sigma_t.square()
        if config['schedule'] == 'flat': multiplier = 1.
        elif config['schedule'] == 'late_ramp': multiplier = 1. - step / start
        elif config['schedule'] == 'mean_compensated':
            multiplier = float(np.clip(.05 / float(weight.flatten()[0]), .25, 4.))
        else: raise ValueError('Unknown strength schedule')
        for j, (pos, at, feat) in enumerate(zip(xyz, kinds, features)):
            row = dict(case_id=self.task['case_id'], attempt=self.attempts[j], seed=self.seeds[j], step=step,
                       lambda_x=config['lambda_x'], lambda_h=config['lambda_h'], schedule_multiplier=multiplier,
                       posterior_x0_to_mean_weight=float(weight[j, 0, 0]), coordinate_score_calls=0, element_score_calls=0)
            try:
                mol = self.reconstruct(pos.tolist(), NUMBERS[at].tolist())
                Chem.SanitizeMol(mol); mol = preserve_core_graph(mol, self.core)
                if mol.GetNumAtoms() != len(pos) or len(Chem.GetMolFrags(mol)) != 1:
                    raise ValueError('Invalid intermediate molecule')
                if [a.GetAtomicNum() for a in mol.GetAtoms()] != NUMBERS[at].tolist():
                    raise ValueError('Reconstruction changed atom order')
            except AssertionError: raise
            except Exception as exc:
                row.update(status='skipped_chemistry', reason=repr(exc)); records.append(row); continue
            # All lambda values use the same ligand-centroid score frame.
            score_center = pos.mean(0)
            if config['lambda_x']:
                affinity, pose, gradient = self.coordinate_client.affinity_gradient(mol, pos, self.task['protein_path'], score_center)
                dx[j], info = scaled_gradient(gradient, config['lambda_x'] * multiplier, .2, nc)
                row.update(coordinate_score_calls=1, coordinate_affinity_pK=affinity, coordinate_gradient=info)
            if config['lambda_h']:
                try:
                    candidates, labels, rejected, diagnostics = self.guard.proposals(mol, self.core, self.reconstruct)
                except AssertionError: raise
                except Exception as exc:
                    # Only discrete chemistry/geometry is inside this block;
                    # IPC/scorer failures below remain fatal, never hidden.
                    candidates,labels,rejected,diagnostics=[mol],[None],{'proposal_chemistry_error':repr(exc)},[]
                row.update(legal_element_counterfactuals=len(candidates)-1, proposal_rejections=rejected)
                if len(candidates) > 1:
                    scores, errors = self.element_client.score_candidates(candidates, pos, self.task['protein_path'], score_center)
                    values, allowed = score_table(mol, labels, scores, nc)
                    site = nc + int(np.argmax(values[nc:].max(axis=1)))
                    objective, gradient = feature_gradient(feat, values, allowed, nc, .5)
                    gradient[:site] = 0; gradient[site+1:] = 0
                    if values[site].max() <= 0: gradient[:] = 0
                    dh[j], info = scaled_gradient(gradient, config['lambda_h'] * multiplier, .05, nc)
                    row.update(element_score_calls=len(scores), element_affinity_pK=scores[0],
                               element_typing_errors=errors, element_gradient=info, selected_element_gradient_site=site,
                               local_expected_gain_pK=objective, best_counterfactual_gain_pK=float(values[site].max()))
            row.update(status='guided', actual_x0_coordinate_max_change_A=float(np.linalg.norm(dx[j], axis=1).max()),
                       actual_x0_feature_max_change=float(np.linalg.norm(dh[j], axis=1).max()),
                       actual_reverse_mean_coordinate_max_change_A=float(weight[j,0,0])*float(np.linalg.norm(dx[j],axis=1).max()),
                       actual_reverse_mean_feature_max_change=float(weight[j,0,0])*float(np.linalg.norm(dh[j],axis=1).max()))
            records.append(row)
        delta = torch.cat([torch.as_tensor(dx,device=z_t.device)/e.norm_values[0],
                           torch.as_tensor(dh,device=z_t.device)/e.norm_values[1]],dim=-1).to(z_t.dtype)
        assert torch.count_nonzero(delta * scaffold_mask.unsqueeze(-1)) == 0
        guided_epsilon = epsilon - alpha_t / sigma_t * delta
        mu = z_t / alpha_ts - sigma2_ts / alpha_ts / sigma_t * guided_epsilon
        z_s = e.sample_normal(mu, sigma_ts*sigma_s/sigma_t, extension_mask.unsqueeze(-1))
        result = z_t * scaffold_mask.unsqueeze(-1) + z_s * extension_mask.unsqueeze(-1)
        assert torch.isfinite(result).all()
        trace = self.output/'guidance_traces';trace.mkdir(exist_ok=True)
        np.savez_compressed(trace/f"{self.task['case_id']}_{self.attempts[0]:04d}_s{step:03d}.npz",
                            predicted_xyz=xyz, predicted_features=features, delta_x0_coordinates=dx,
                            delta_x0_features=dh, posterior_weight=weight.cpu().numpy(), seeds=self.seeds)
        with (self.output/'events.jsonl').open('a') as handle:
            for row in records: handle.write(json.dumps(row)+'\n')
        return result


class StrengthEngine(LocalRegenerator):
    def __init__(self):
        super().__init__()
        self.original_noise = self.edm.sample_combined_position_features_noise
        self.hook = StrengthHook(self.edm, self.reconstruct)

    def fresh_inputs(self, task, count):
        key=task['case_id']
        if key not in self.cache:
            ph,px,cx,ch=self.get_pocket(task['protein_path'],task['core_path'],self.atom_dict,self.pocket_atom_dict,dist_cutoff=7)
            core=read_mol(task['core_path']);assert len(px) and np.allclose(cx,core.GetConformer().GetPositions(),atol=1e-6)
            self.cache[key]=ph,px,cx,ch,core
        ph,px,cx,ch,core=self.cache[key];nc,nr=len(cx),task['rgroup_heavy_atoms'];center=cx[task['anchor_in_core']]
        def tensor(v):return torch.as_tensor(np.asarray(v),dtype=torch.float32,device='cuda')
        x=torch.zeros((count,nc+nr,3),device='cuda');h=torch.zeros((count,nc+nr,10),device='cuda')
        x[:,:nc]=tensor(cx-center);h[:,:nc]=tensor(ch)
        fixed=torch.zeros((count,nc+nr),device='cuda');fixed[:,:nc]=1
        anchors=torch.zeros_like(fixed);anchors[:,task['anchor_in_core']]=1
        pm=torch.ones((count,len(px)),device='cuda')
        return dict(x=x,h=h,scaffold_mask=fixed,extension_mask=1-fixed,anchors=anchors,
                    pocket_x=tensor(px-center)[None].repeat(count,1,1),pocket_h=tensor(ph)[None].repeat(count,1,1),
                    pocket_mask=pm,pocket_anchors=torch.zeros_like(pm)),center,core

    def generate(self, task, attempts, seeds, noise):
        inputs,center,core=self.fresh_inputs(task,len(attempts));self.hook.context(task,center,core,attempts,seeds)
        tape=NoiseTape(noise)
        def draw(_edm,n_samples,n_nodes,mask):return tape.draw(n_samples,n_nodes,mask)
        self.edm.sample_combined_position_features_noise=types.MethodType(draw,self.edm)
        try:
            x,h,metadata=sample_restart(self.edm,inputs,'full',self.edm.T)
            assert tape.position==self.edm.T+2
        finally:self.edm.sample_combined_position_features_noise=self.original_noise
        xyz=x.cpu().numpy()+center;numbers=NUMBERS[h.argmax(-1).cpu().numpy()];outputs=[]
        for pos,atoms in zip(xyz,numbers):
            try:
                mol=self.reconstruct(pos.tolist(),atoms.tolist());Chem.SanitizeMol(mol)
                mol=normalize(preserve_core_graph(mol,core))[0]
                for i,p in enumerate(np.round(mol.GetConformer().GetPositions(),4)):mol.GetConformer().SetAtomPosition(i,p)
                outputs.append((mol,None))
            except AssertionError:raise
            except Exception as exc:outputs.append((None,repr(exc)))
        return outputs,xyz,numbers,dict(**metadata,noise_tape_sha256=tape.sha256,noise_draws=tape.position)
