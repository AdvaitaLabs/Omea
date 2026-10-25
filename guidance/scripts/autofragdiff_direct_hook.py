"""Direct joint input gradients with one native diffusion trajectory per seed."""
import json
import types
import numpy as np
import torch
from rdkit import Chem
from autofragdiff_local_regeneration import LocalRegenerator, NUMBERS
from autofragdiff_strength_hook import StrengthEngine as LegacyEngine, make_noise, NoiseTape, scaled_gradient
from fixed_core_reconstruction import preserve_core_graph
from gnina_soft_element_client import SoftElementClient


class DirectHook:
    def __init__(self, edm, reconstruct):
        self.edm, self.reconstruct = edm, reconstruct
        self.original = edm.sample_p_zs_given_zt_only_extension
        self.client = None
        self.configuration = dict(lambda_x=0., lambda_h=0., last_fraction=1., interval=1,
                                  schedule='flat', temperature=.2)
        owner = self
        def wrapper(_edm, *args, **kwargs):
            return owner.step(*args, **kwargs)
        edm.sample_p_zs_given_zt_only_extension = types.MethodType(wrapper, edm)

    def configure(self, configuration, output):
        self.configuration = dict(configuration)
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        if (configuration['lambda_x'] or configuration['lambda_h']) and self.client is None:
            self.client = SoftElementClient(output/'joint_worker.log')

    def context(self, task, center, core, attempts, seeds):
        self.task, self.center, self.core = task, center, core
        self.attempts, self.seeds = list(attempts), list(seeds)
        self.score_center = core.GetConformer().GetPositions().mean(0)

    def close(self):
        if self.client:
            self.client.close()

    def step(self, s, t, z_t, pocket_xh, scaffold_mask, extension_mask, anchors, pocket_anchors, pocket_mask):
        e, config = self.edm, self.configuration
        step = int(round(float(s[0, 0])*e.T))
        start = round(e.T*config['last_fraction'])
        if not (config['lambda_x'] or config['lambda_h']) or step >= start or step % config['interval']:
            return self.original(s,t,z_t,pocket_xh,scaffold_mask,extension_mask,anchors,pocket_anchors,pocket_mask)
        assert config['schedule'] == 'flat'
        gamma_s, gamma_t = e.gamma(s), e.gamma(t)
        sigma2_ts, sigma_ts, alpha_ts = e.sigma_and_alpha_t_given_s(gamma_t,gamma_s,z_t)
        sigma_s, sigma_t = e.sigma(gamma_s,z_t), e.sigma(gamma_t,z_t)
        alpha_t = e.alpha(gamma_t,z_t)
        epsilon = e.dynamics.forward(xh=z_t,pocket_xh=pocket_xh,t=t,extension_mask=extension_mask,
                                     scaffold_mask=scaffold_mask,anchors=anchors,pocket_anchors=pocket_anchors,
                                     pocket_mask=pocket_mask)*extension_mask.unsqueeze(-1)
        predicted = e.compute_x_pred(epsilon,z_t,gamma_t)
        predicted = predicted*extension_mask.unsqueeze(-1)+z_t*scaffold_mask.unsqueeze(-1)
        x,h = e.unnormalize(predicted[:,:,:3],predicted[:,:,3:])
        xyz = x.cpu().numpy()+self.center
        features = h.cpu().numpy()
        dx,dh = np.zeros_like(xyz),np.zeros_like(features)
        nc = self.core.GetNumAtoms()
        weight = sigma2_ts/alpha_ts*alpha_t/sigma_t.square()
        rows = []
        for j,(pos,feat) in enumerate(zip(xyz,features)):
            # Optional discrete context is used only to identify chemical roles.
            # Failed reconstruction falls back to a graph-free density encoding;
            # it never suppresses scoring/gradients or changes generated atoms.
            context = None
            context_error = None
            try:
                context = self.reconstruct(pos.tolist(),NUMBERS[feat.argmax(-1)].tolist())
                Chem.SanitizeMol(context)
                context = preserve_core_graph(context,self.core)
                assert context.GetNumAtoms() == len(pos)
                assert [a.GetAtomicNum() for a in context.GetAtoms()] == NUMBERS[feat.argmax(-1)].tolist()
            except AssertionError:
                raise
            except Exception as exc:
                context = None
                context_error = repr(exc)
            affinity,pose,gx,gh,diagnostics = self.client.joint_gradient(
                self.core,pos,feat,self.task['protein_path'],self.score_center,config['temperature'],context)
            dx[j],xinfo = scaled_gradient(gx,config['lambda_x'],.2,nc)
            dh[j],hinfo = scaled_gradient(gh,config['lambda_h'],.05,nc)
            rows.append(dict(case_id=self.task['case_id'],attempt=self.attempts[j],seed=self.seeds[j],step=step,
                             status='guided',lambda_x=config['lambda_x'],lambda_h=config['lambda_h'],
                             schedule_multiplier=1.,posterior_x0_to_mean_weight=float(weight[j,0,0]),
                             coordinate_score_calls=1,element_score_calls=0,joint_model_forward_calls=1,
                             joint_backward_calls=1,scored_element_substitution_candidates=0,
                             affinity_pK=affinity,pose_probability=pose,coordinate_gradient=xinfo,feature_gradient=hinfo,
                             diagnostics=diagnostics,context_reconstruction_error=context_error,
                             actual_x0_coordinate_max_change_A=float(np.linalg.norm(dx[j],axis=1).max()),
                             actual_x0_feature_max_change=float(np.linalg.norm(dh[j],axis=1).max()),
                             actual_x0_feature_total_change=float(np.linalg.norm(dh[j])),
                             actual_x0_feature_active_sites=np.flatnonzero(np.linalg.norm(dh[j],axis=1)>0).tolist(),
                             actual_reverse_mean_coordinate_max_change_A=float(weight[j,0,0])*float(np.linalg.norm(dx[j],axis=1).max()),
                             actual_reverse_mean_feature_max_change=float(weight[j,0,0])*float(np.linalg.norm(dh[j],axis=1).max())))
        delta = torch.cat([torch.as_tensor(dx,device=z_t.device)/e.norm_values[0],
                           torch.as_tensor(dh,device=z_t.device)/e.norm_values[1]],dim=-1).to(z_t.dtype)
        assert torch.count_nonzero(delta*scaffold_mask.unsqueeze(-1)) == 0
        guided_epsilon = epsilon-alpha_t/sigma_t*delta
        mu = z_t/alpha_ts-sigma2_ts/alpha_ts/sigma_t*guided_epsilon
        z_s = e.sample_normal(mu,sigma_ts*sigma_s/sigma_t,extension_mask.unsqueeze(-1))
        result = z_t*scaffold_mask.unsqueeze(-1)+z_s*extension_mask.unsqueeze(-1)
        assert torch.isfinite(result).all()
        trace = self.output/'guidance_traces'
        trace.mkdir(exist_ok=True)
        np.savez_compressed(trace/f"{self.task['case_id']}_{self.attempts[0]:04d}_s{step:03d}.npz",
                            predicted_xyz=xyz,predicted_features=features,delta_x0_coordinates=dx,
                            delta_x0_features=dh,posterior_weight=weight.cpu().numpy(),seeds=self.seeds)
        with (self.output/'events.jsonl').open('a') as handle:
            for row in rows:
                handle.write(json.dumps(row)+'\n')
        return result


class StrengthEngine(LegacyEngine):
    def __init__(self):
        LocalRegenerator.__init__(self)
        self.original_noise = self.edm.sample_combined_position_features_noise
        self.hook = DirectHook(self.edm,self.reconstruct)
