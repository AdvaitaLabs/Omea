"""Isolated ABI worker: one joint score/gradient, no substitution scoring."""
import json
import sys
import traceback
from gnina_soft_element_guidance import SoftElementScorer
from rdkit import Chem
import torch

torch.set_num_threads(2)
scorer = SoftElementScorer()
print(json.dumps(dict(status='ready', cnn='default', element_features=10)), flush=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
        if req.get('stop'):
            break
        core = Chem.MolFromMolBlock(req['core_molblock'], removeHs=True, sanitize=True)
        assert core is not None
        context = None
        if req.get('context_molblock'):
            context = Chem.MolFromMolBlock(req['context_molblock'], removeHs=True, sanitize=True)
        affinity, pose, gx, gh, diagnostics = scorer.joint_gradient(
            core, req['xyz'], req['features'], req['protein'], req['center'], req['temperature'], context)
        print(json.dumps(dict(status='ok', affinity=affinity, pose=pose,
                              coordinate_gradient=gx.tolist(), feature_gradient=gh.tolist(), diagnostics=diagnostics)), flush=True)
    except Exception:
        print(json.dumps(dict(status='error', traceback=traceback.format_exc())), flush=True)
