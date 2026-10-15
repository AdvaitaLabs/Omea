"""Native DELETE / MaSIF surface features, cropped using retained core only.

Run in the isolated Python 3.7 / PyMesh environment. A separate exporter writes
core XYZ as JSON, avoiding any need to downgrade the main inference environment.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser,PDBIO,Select,NeighborSearch,Selection
from scipy.spatial import KDTree,distance
import pymesh

ROOT=Path(os.environ.get('RGROUP_BENCHMARK_ROOT', str(Path(__file__).resolve().parents[1]))).expanduser().resolve()
sys.path.insert(0,str(ROOT/'vendor/DELETE/utils/masif'))
from computeMSMS import computeMSMS
from computeCharges import computeCharges,assignChargesToNewMesh
from computeHydrophobicity import computeHydrophobicity
from compute_normal import compute_normal
from fixmesh import fix_mesh
from save_ply import save_ply


def prepare(task,apbs,pdb2pqr,multivalue):
    folder=ROOT/'data/delete_surfaces'/task['case_id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():return
    start=time.time();xyz=np.asarray(task['core_xyz'])
    structure=PDBParser(QUIET=True).get_structure('protein',task['protein_path'])[0]
    search=NeighborSearch(Selection.unfold_entities(structure,'A'))
    residues=Selection.uniqueify([r for point in xyz for r in search.search(point,13.,level='R')])
    class Pocket(Select):
        def accept_residue(self,r):
            return r in residues and (all(name in r for name in ('N','CA','C','O')) or r.resname=='HOH')
    pocket=folder/'pocket.pdb';writer=PDBIO();writer.set_structure(structure);writer.save(str(pocket),Pocket())
    cropped=PDBParser(QUIET=True).get_structure('',str(pocket))[0]
    atoms=Selection.unfold_entities(cropped,'A')
    closest=int(np.argmin([distance.euclidean(xyz.mean(0),a.coord) for a in atoms]))
    msms=ROOT/'tools/delete/msms.x86_64Linux2.2.6.1'
    vertices,faces,_,names,_=computeMSMS(str(pocket),protonate=True,one_cavity=closest,
        msms_bin=str(msms),workdir=str(folder))
    distances,_=KDTree(xyz).query(vertices)
    kept=set(np.where(distances <= 8.)[0].tolist())
    faces_keep=[i for i,face in enumerate(faces) if all(int(v) in kept for v in face)]
    if not faces_keep:raise ValueError('No surface faces near retained core')
    hbond=computeCharges(task['protein_path'][:-4],vertices,names)
    hphob=computeHydrophobicity(names)
    mesh=pymesh.submesh(pymesh.form_mesh(vertices,faces),faces_keep,0)
    regular=fix_mesh(mesh,1.0)
    regular,_=pymesh.remove_degenerated_triangles(regular)
    normals=compute_normal(regular.vertices,regular.faces,eps=1.e-6)
    hbond=assignChargesToNewMesh(regular.vertices,vertices,hbond,True)
    hphob=assignChargesToNewMesh(regular.vertices,vertices,hphob,True)
    # Same APBS/PARSE preparation as the native helper, with checked exit codes
    # and absolute paths instead of its hard-coded working-directory names.
    import subprocess
    log=(folder/'electrostatics.log').open('w')
    binary_environment = os.environ.copy()
    library_dirs = [str(apbs.parent.parent/'lib'),
                    str(ROOT/'tools/delete/legacy_libraries/lib/x86_64-linux-gnu'),
                    str(ROOT/'tools/delete/legacy_libraries/usr/lib/x86_64-linux-gnu')]
    binary_environment['LD_LIBRARY_PATH'] = ':'.join(library_dirs + [binary_environment.get('LD_LIBRARY_PATH', '')])
    subprocess.run([str(pdb2pqr),'--ff=PARSE','--whitespace','--noopt','--apbs-input',str(pocket),'temp1'],
        cwd=str(folder),stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([str(apbs),'temp1.in'],cwd=str(folder),env=binary_environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    np.savetxt(str(folder/'temp1.csv'),regular.vertices,delimiter=',')
    subprocess.run([str(multivalue),'temp1.csv','temp1.dx','temp1_out.csv'],cwd=str(folder),env=binary_environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    log.close();charges=np.loadtxt(str(folder/'temp1_out.csv'),delimiter=',')[:,3]
    regular.add_attribute('vertex_mean_curvature');regular.add_attribute('vertex_gaussian_curvature')
    H=regular.get_attribute('vertex_mean_curvature');K=regular.get_attribute('vertex_gaussian_curvature')
    delta=H*H-K;delta[delta<0]=1.e-8
    k1,k2=H+np.sqrt(delta),H-np.sqrt(delta)
    shape=np.arctan((k1+k2)/(k1-k2))*(2/np.pi)
    if not all(np.isfinite(value).all() for value in (regular.vertices,charges,hbond,hphob,shape)):
        raise ValueError('Non-finite native protein surface features')
    ply=folder/'pocket.ply'
    save_ply(str(ply),regular.vertices,regular.faces,normals=normals,charges=charges,
        normalize_charges=True,hbond=hbond,hphob=hphob,si=shape)
    manifest=dict(case_id=task['case_id'],source='DELETE native MSMS/PyMesh/APBS/PARSE surface pipeline',
        core_sha256=task['core_sha256'],protein_sha256=task['protein_sha256'],
        surface_sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),vertices=len(regular.vertices),
        surface_radius_A=8.,residue_crop_A=13.,mesh_resolution_A=1.,core_only_crop=True,
        removed_R_not_loaded=True,seconds=time.time()-start,
        binary_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (msms,apbs,pdb2pqr,multivalue)})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    base=ROOT/'tools/delete'
    apbs=next(base.rglob('bin/apbs'));pdb2pqr=base/'pdb2pqr-linux-bin64-2.1.1/pdb2pqr'
    multivalue=next(base.rglob('multivalue'))
    tasks=json.loads((ROOT/'data/delete_surface_inputs.json').read_text())[:args.limit]
    for task in tasks:prepare(task,apbs,pdb2pqr,multivalue)


if __name__=='__main__':main()
