"""Generator IPC for direct derivatives of the continuous element encoding."""
import atexit
import json
import subprocess
from pathlib import Path
import numpy as np
from rdkit import Chem
from gnina_gradient_client import GradientClient


class SoftElementClient(GradientClient):
    def __init__(self, stderr_path):
        root = Path(__file__).resolve().parents[1]
        self.log = Path(stderr_path).open('w')
        self.process = subprocess.Popen([str(root/'envs/guidance/bin/python'),
                                         str(root/'scripts/gnina_soft_element_worker.py')],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        atexit.register(self.close)
        if self.receive().get('status') != 'ready':
            raise RuntimeError('Soft element scorer not ready')

    def joint_gradient(self, core, xyz, features, protein, center, temperature, context):
        request = dict(core_molblock=Chem.MolToMolBlock(core), xyz=xyz.tolist(), features=features.tolist(),
                       protein=str(protein), center=center.tolist(), temperature=temperature,
                       context_molblock=Chem.MolToMolBlock(context) if context is not None else None)
        self.process.stdin.write(json.dumps(request)+'\n')
        self.process.stdin.flush()
        result = self.receive()
        if result.get('status') != 'ok':
            raise RuntimeError(result)
        gx = np.asarray(result['coordinate_gradient'], dtype=np.float32)
        gh = np.asarray(result['feature_gradient'], dtype=np.float32)
        assert gx.shape == xyz.shape and gh.shape == features.shape
        assert np.isfinite(gx).all() and np.isfinite(gh).all()
        return result['affinity'], result['pose'], gx, gh, result['diagnostics']
