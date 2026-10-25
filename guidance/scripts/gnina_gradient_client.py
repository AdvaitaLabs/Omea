"""Synchronous local pipe client; no scoring packages imported by the generator."""
import atexit
import json
import select
import subprocess
from pathlib import Path
import numpy as np
from rdkit import Chem


class GradientClient:
    def __init__(self, stderr_path):
        root = Path(__file__).resolve().parents[1]
        self.log = Path(stderr_path).open('w')
        self.process = subprocess.Popen([str(root/'envs/guidance/bin/python'),
                                        str(root/'scripts/gnina_gradient_worker.py')],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=self.log, text=True, bufsize=1)
        atexit.register(self.close)
        response = self.receive()
        if response.get('status') != 'ready':
            raise RuntimeError(response)

    def receive(self):
        if not select.select([self.process.stdout], [], [], 120)[0]:
            raise TimeoutError('Isolated GNINA worker timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'GNINA worker terminated: {self.process.poll()}')
        return json.loads(line)

    def affinity_gradient(self, mol, xyz, protein, center):
        req = dict(molblock=Chem.MolToMolBlock(mol), xyz=xyz.tolist(),
                   protein=str(protein), center=center.tolist())
        self.process.stdin.write(json.dumps(req)+'\n')
        self.process.stdin.flush()
        result = self.receive()
        if result.get('status') != 'ok':
            raise RuntimeError(f'GNINA scoring failed: {result}')
        grad = np.asarray(result['gradient'], dtype=np.float32)
        if grad.shape != xyz.shape or not np.isfinite(grad).all():
            raise ValueError('Invalid coordinate gradient from worker')
        return result['affinity'], result['pose'], grad

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"stop":true}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.log.close()
