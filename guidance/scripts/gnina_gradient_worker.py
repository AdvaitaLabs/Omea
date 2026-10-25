"""Persistent isolated scorer; avoids RDKit/libmolgrid Boost registry collisions."""
import json
import sys
import traceback
from gnina_guidance import CoordinateScorer
from rdkit import Chem
import torch

torch.set_num_threads(2)
scorer = CoordinateScorer('default')
print(json.dumps(dict(status='ready', cnn='default', device=torch.cuda.get_device_name(0))), flush=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
        if req.get('stop'):
            break
        mol = Chem.MolFromMolBlock(req['molblock'], removeHs=True, sanitize=True)
        if mol is None:
            raise ValueError('Cannot read ligand from generator')
        aff, pose, grad = scorer.affinity_gradient(mol, req['xyz'], req['protein'], req['center'])
        print(json.dumps(dict(status='ok', affinity=aff, pose=pose, gradient=grad.tolist())), flush=True)
    except Exception:
        print(json.dumps(dict(status='error', traceback=traceback.format_exc())), flush=True)
