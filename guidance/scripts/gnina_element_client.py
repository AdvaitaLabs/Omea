"""Local IPC client for discrete-element counterfactual scoring."""
import atexit
import json
import subprocess
from pathlib import Path
from rdkit import Chem
from gnina_gradient_client import GradientClient


class ElementClient(GradientClient):
    def __init__(self,stderr_path):
        root=Path(__file__).resolve().parents[1]
        self.log=Path(stderr_path).open('w')
        self.process=subprocess.Popen([str(root/'envs/guidance/bin/python'),str(root/'scripts/gnina_element_worker.py')],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        atexit.register(self.close)
        if self.receive().get('status')!='ready':raise RuntimeError('Element worker not ready')

    def score_candidates(self,mols,xyz,protein,center):
        req=dict(molblocks=[Chem.MolToMolBlock(m) for m in mols],xyz=xyz.tolist(),protein=str(protein),center=center.tolist())
        self.process.stdin.write(json.dumps(req)+'\n');self.process.stdin.flush()
        response=self.receive()
        if response.get('status')!='ok':raise RuntimeError(response)
        if len(response['affinities'])!=len(mols):raise ValueError('Incomplete candidate scores')
        return response['affinities'],response['typing_errors']
