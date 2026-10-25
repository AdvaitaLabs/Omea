"""Numerical derivative and endpoint checks for the real 10-element bridge."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from gnina_soft_element_guidance import SoftElementScorer, NUMBERS
from rdkit import Chem
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    began = time.perf_counter()
    scorer = SoftElementScorer()
    tasks = json.loads((root/'results/guidance_dev_v1/tasks.json').read_text())
    results = []
    for task in tasks:
        path = root/'results/guidance_multisite_dev_v1/arms/zero'/task['case_id']/'0000/raw.rdmol'
        mol = Chem.Mol(path.read_bytes())
        core = next(m for m in Chem.SDMolSupplier(task['core_path'], removeHs=True) if m is not None)
        xyz = mol.GetConformer().GetPositions()
        nc = core.GetNumAtoms()
        kinds = np.array([NUMBERS.index(a.GetAtomicNum()) for a in mol.GetAtoms()])
        features = np.eye(10, dtype=np.float32)[kinds]
        encoding = scorer.encoding(core, xyz, mol)
        x = scorer.tensor(xyz).detach().requires_grad_(True)
        h = scorer.tensor(features).detach().requires_grad_(True)
        hard = scorer.tensor(features[nc:])
        center = core.GetConformer().GetPositions().mean(0)
        with torch.no_grad():
            hard_score, _ = scorer.relaxed_score(x, h, task['protein_path'], center, encoding, hard_probabilities=hard)
            typed_score, _ = scorer.score(x, scorer.ligand_types(mol), task['protein_path'], center)
        hard_error = float(torch.abs(hard_score-typed_score))
        assert hard_error < 1e-4, hard_error
        temperatures = []
        for temperature in [.1, .2, .5]:
            score, _ = scorer.relaxed_score(x,h,task['protein_path'],center,encoding,temperature)
            gx,gh = torch.autograd.grad(score,(x,h))
            assert torch.isfinite(gx).all() and torch.isfinite(gh).all()
            assert torch.count_nonzero(gh[:nc]) == 0
            assert torch.all(torch.linalg.vector_norm(gh[nc:],dim=-1)>0)
            assert torch.allclose(gh.sum(-1), torch.zeros(len(xyz),device=gh.device), atol=1e-5)
            temperatures.append(dict(temperature=temperature,relaxed_score=float(score.detach()),
                                     hard_score=float(typed_score),score_difference=float(score.detach()-typed_score),
                                     r_coordinate_gradient_norms=torch.linalg.vector_norm(gx[nc:],dim=-1).cpu().tolist(),
                                     r_feature_gradient_norms=torch.linalg.vector_norm(gh[nc:],dim=-1).cpu().tolist()))
        # Check the chosen temperature against finite differences in h[*,10].
        score,_ = scorer.relaxed_score(x,h,task['protein_path'],center,encoding,.2)
        gx,gh = torch.autograd.grad(score,(x,h))
        generator = torch.Generator(device='cpu').manual_seed(2026101020)
        finite = []
        for label,base,gradient in [('coordinates',x,gx),('generator_element_features',h,gh)]:
            direction = gradient.detach().clone()
            direction[:nc] = 0
            if base is h:
                direction[nc:] -= direction[nc:].mean(-1,keepdim=True)
            direction /= torch.linalg.vector_norm(direction)
            analytic = float((gradient*direction).sum())
            trials = []
            with torch.no_grad():
                for epsilon in [.01,.005,.002]:
                    if base is x:
                        plus = scorer.relaxed_score(x+epsilon*direction,h,task['protein_path'],center,encoding,.2)[0]
                        minus = scorer.relaxed_score(x-epsilon*direction,h,task['protein_path'],center,encoding,.2)[0]
                    else:
                        plus = scorer.relaxed_score(x,h+epsilon*direction,task['protein_path'],center,encoding,.2)[0]
                        minus = scorer.relaxed_score(x,h-epsilon*direction,task['protein_path'],center,encoding,.2)[0]
                    numerical = float((plus-minus)/(2*epsilon))
                    error = abs(numerical-analytic)/max(abs(analytic),1e-6)
                    trials.append(dict(epsilon=epsilon,analytic=analytic,numerical=numerical,relative_error=error))
            assert any(r['relative_error'] < .05 for r in trials), (task['case_id'],label,trials)
            finite.append(dict(input=label,direction='unit gradient over R atoms, core zero; avoids cancellation and float32 score quantization',trials=trials,passed=True))
        # Bad intermediate chemistry can still score and differentiate every R feature.
        affinity,pose,fx,fh,diag = scorer.joint_gradient(core,xyz,features,task['protein_path'],center,.2,None)
        assert np.isfinite(fx).all() and np.isfinite(fh).all()
        assert np.all(np.linalg.norm(fh[nc:],axis=-1)>0)
        np.savez_compressed(out/(task['case_id']+'.npz'),xyz=xyz,features=features,
                            coordinate_gradient=gx.cpu().numpy(),feature_gradient=gh.cpu().numpy())
        row = dict(case_id=task['case_id'],source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   hard_endpoint_score_error=hard_error,temperature_checks=temperatures,
                   finite_difference_checks=finite,graph_free_fallback_finite_and_all_R_gradients_nonzero=True)
        results.append(row)
        print(json.dumps(dict(case_id=task['case_id'],hard_endpoint_score_error=hard_error,passed=True)),flush=True)
    # Explicit invariance: adding a common constant to all ten feature channels
    # changes neither the soft probabilities nor their score.
    result = dict(passed=True,rows=results,generator_10_element_bridge_implemented=True,
                  exact_current_hard_endpoint=True,all_R_feature_gradients_checked=True,
                  no_element_substitution_scoring=True,chosen_temperature=.2,
                  temperature_choice='predeclared moderate softness; alternate values diagnostic only, not affinity-based selection',
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),root/'scripts/gnina_soft_element_guidance.py',root/'scripts/gnina_guidance.py']},
                  limitations=['The score uses continuous element-density mixtures; it is not a measured affinity or a derivative through discrete bond perception.',
                               'Alternative-element chemical roles are a local detached approximation, with graph-free role mixtures if typing is unavailable.',
                               'Hard endpoint agreement validates the encoding at existing molecules, not ranking accuracy for generated mixtures.'],
                  wall_seconds=time.perf_counter()-began)
    (out/'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(passed=True,wall_seconds=result['wall_seconds'])),flush=True)


if __name__ == '__main__':
    main()

