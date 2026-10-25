"""Batched fixed-pose scoring of valid element counterfactuals; isolated ABI."""
import json
import sys
import traceback
from gnina_guidance import CoordinateScorer
import molgrid
from rdkit import Chem
import torch

torch.set_num_threads(2)
scorer=CoordinateScorer('default')
print(json.dumps(dict(status='ready',cnn='default',batch_size=8)),flush=True)
for line in sys.stdin:
    try:
        req=json.loads(line)
        if req.get('stop'):break
        rx,rt,rr=scorer.receptor(req['protein'])
        xyz=scorer.tensor(req['xyz'])
        coords=torch.cat([rx,xyz],0).contiguous()
        coords_to_grid=molgrid.Coords2Grid(scorer.gmaker,center=tuple(float(v) for v in req['center']))
        values=[None]*len(req['molblocks']);errors={}
        for start in range(0,len(values),8):
            grids=[];ids=[]
            for i in range(start,min(start+8,len(values))):
                mol=Chem.MolFromMolBlock(req['molblocks'][i],sanitize=True,removeHs=True)
                if mol is None:
                    raise ValueError(f'Worker cannot deserialize candidate {i}')
                try:
                    lt,lr=scorer.ligand_types(mol)
                except ValueError as exc:
                    errors[i]=str(exc)
                    continue
                with torch.no_grad():
                    grid=coords_to_grid(coords,torch.cat([rt,lt],0).contiguous(),torch.cat([rr,lr],0).contiguous())
                grids.append(grid);ids.append(i)
            if grids:
                with torch.no_grad():
                    aff=scorer.model(torch.stack(grids))[1].flatten().cpu().tolist()
                for i,value in zip(ids,aff):values[i]=value
        print(json.dumps(dict(status='ok',affinities=values,typing_errors=errors)),flush=True)
    except Exception:
        print(json.dumps(dict(status='error',traceback=traceback.format_exc())),flush=True)
