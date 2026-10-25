"""Versioned audit: infer double-bond stereo from unchanged 3D coordinates.

RDKit's V2000 writer can encode an unset generated double bond as EITHERDOUBLE.
That serialization flag blocks AssignStereochemistryFrom3D on reload, although
the in-memory generator audit inferred the stereo. Clear that flag uniformly,
without changing topology/coordinates, then use the same frozen geometry audit.
Original generation/audit/scoring artifacts are never overwritten.
"""
import collections
import concurrent.futures as cf
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from rdkit import Chem,RDLogger
from tasks import ROOT,audit,read_mol
from score_results import score_one

RUN=ROOT/'results/strict_single_site_formal_v1'
OUT=RUN/'analysis_geometry_stereo_v2'


def normalize(mol):
    copied=Chem.Mol(mol);changed=[]
    for bond in copied.GetBonds():
        if bond.GetBondType()==Chem.BondType.DOUBLE and (bond.GetStereo()==Chem.BondStereo.STEREOANY or bond.GetBondDir()==Chem.BondDir.EITHERDOUBLE):
            changed.append(bond.GetIdx())
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            bond.SetBondDir(Chem.BondDir.NONE)
    assert np.array_equal(mol.GetConformer().GetPositions(),copied.GetConformer().GetPositions())
    return copied,changed


def write_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def main():
    RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((RUN/'manifest.json').read_text())
    tasklist=json.loads((RUN/'tasks.json').read_text());tasks={t['case_id']:t for t in tasklist}
    controls=[]
    for task in tasklist:
        mol,_=normalize(read_mol(task['reference_path']))
        assert audit(mol,task)['strict_success'],task['case_id']
        controls.append(task['case_id'])
    # A direct regression: the same 3D molecule before/after SDF round trip.
    case='6G6Y-bio1_EOK_A_403__a015_r002'
    original=Chem.SDMolSupplier(str(RUN/'FLOWR.root'/case/'raw.sdf'),sanitize=False,removeHs=False)[2]
    normalized,changed=normalize(original)
    assert changed and not audit(original,tasks[case])['strict_success']
    assert audit(normalized,tasks[case])['strict_success']
    Chem.SanitizeMol(normalized);Chem.AssignStereochemistryFrom3D(normalized)
    reread=Chem.MolFromMolBlock(Chem.MolToMolBlock(normalized),sanitize=False,removeHs=False)
    assert audit(reread,tasks[case])['strict_success']
    # Flip the fixed imine substituent across the double-bond axis. The inferred
    # E/Z must fail even though the atom graph is unchanged.
    wrong=Chem.Mol(normalized);xyz=wrong.GetConformer().GetPositions()
    axis=xyz[0]-xyz[11];axis/=np.linalg.norm(axis)
    v=xyz[10]-xyz[11];xyz[10]=xyz[11]+2*np.dot(v,axis)*axis-v
    wrong.GetConformer().SetAtomPosition(10,xyz[10])
    assert not audit(wrong,tasks[case])['core_stereo_preserved']
    checks=dict(reference_positive_controls=len(controls),sdf_roundtrip_regression_passed=True,
        wrong_EZ_rejected=True,models={},parent_manifest_sha256=hashlib.sha256((RUN/'manifest.json').read_bytes()).hexdigest(),
        method_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rule='Clear only unspecified double-bond stereo flags; infer stereo from unchanged generated 3D coordinates using the frozen common audit. Apply to every model. No topology or coordinate edits.',
        changed_files='Only new files under analysis_geometry_stereo_v2; all parent results retained.')
    shutil.copy2(__file__,OUT/Path(__file__).name)
    allrows=[];scored=[];jobs=[];refs={};seen=set()
    for model in manifest['models']:
        oldscores=json.loads((RUN/f'judge_results_{model}.json').read_text())['rows']
        cache={(r['case_id'],r['attempt']):r for r in oldscores if r['model']==model}
        for r in oldscores:
            if r['model']=='reference':
                if r['case_id'] in refs:
                    for key in ['vina_kcal_mol','cnn_affinity_pK','posebusters_pass']:assert refs[r['case_id']][key]==r[key]
                refs[r['case_id']]=r
        counts=collections.Counter();detailed=[]
        for task in tasklist:
            folder=RUN/model/task['case_id'];path=folder/'attempts.json'
            if not (folder/'summary.json').exists():continue
            old=json.loads(path.read_text());assert len(old)==100
            by_id={r['attempt']:dict(r) for r in old};assert set(by_id)==set(range(100))
            mols={}
            for mol in Chem.SDMolSupplier(str(folder/'raw.sdf'),sanitize=False,removeHs=False):
                if mol is None:continue
                idx=int(mol.GetProp('attempt'));assert idx not in mols
                mols[idx]=mol
            for idx,row in by_id.items():
                key=(model,task['case_id'],idx);assert key not in seen;seen.add(key)
                was_strict=row['strict_success'];was_new=row.get('new_design')
                mol=mols.get(idx);flags=[]
                if mol is not None:
                    mol,flags=normalize(mol)
                    if flags:
                        row.update(audit(mol,task));counts['attempts_with_unspecified_double_bond_flags']+=1
                        if row['strict_success']:
                            Chem.SanitizeMol(mol);Chem.AssignStereochemistryFrom3D(mol)
                counts['attempts']+=1
                counts['strict_before']+=was_strict;counts['strict_after']+=row['strict_success']
                counts['became_strict']+=bool(row['strict_success'] and not was_strict)
                counts['lost_strict']+=bool(was_strict and not row['strict_success'])
                row.update(model=model,case_id=task['case_id'],pdb_id=task['pdb_id'],stratum=task['stratum'],
                    audit_version='geometry_stereo_v2',original_strict_success=was_strict,
                    original_new_design=was_new,normalized_double_bond_indices=flags)
                if row['strict_success']:
                    if (task['case_id'],idx) in cache and not flags:
                        scored.append(dict(cache[(task['case_id'],idx)],score_provenance='unchanged v1 score'))
                        counts['cached_scores']+=1
                    else:
                        jobs.append((model,task,idx,mol,OUT/'judges'/model/task['case_id']/f'{idx:04d}'))
                        counts['score_jobs']+=1
                detailed.append(row)
        checks['models'][model]=dict(counts)
        allrows.extend(detailed);write_json(OUT/f'attempts_{model}.json',detailed)
        print(json.dumps(dict(model=model,**counts)),flush=True)
    assert len(allrows)==22400 and len(refs)==45
    write_json(OUT/'quality_checks.json',checks)
    write_json(OUT/'status.json',dict(stage='scoring',jobs=len(jobs),completed=0))
    completed=0
    with cf.ThreadPoolExecutor(max_workers=24) as pool:
        pending={pool.submit(score_one,mol,task['protein_path'],out):(model,task,idx) for model,task,idx,mol,out in jobs}
        for future in cf.as_completed(pending):
            model,task,idx=pending[future];score=future.result();reference=refs[task['case_id']]
            record=dict(model=model,case_id=task['case_id'],attempt=idx,**score,score_provenance='geometry_stereo_v2')
            if score['status']=='scored':
                record['delta_vina_kcal_mol']=score['vina_kcal_mol']-reference['vina_kcal_mol']
                record['delta_cnn_affinity_pK']=score['cnn_affinity_pK']-reference['cnn_affinity_pK']
            scored.append(record);completed+=1
            if completed%25==0:write_json(OUT/'status.json',dict(stage='scoring',jobs=len(jobs),completed=completed))
    assert len(scored)==sum(r['strict_success'] for r in allrows)
    failed=[r for r in scored if r['status']!='scored' or r.get('posebusters_error') or 'posebusters_checks' not in r]
    write_json(OUT/'judge_results.json',dict(rows=list(refs.values())+scored))
    scoremap={(r['model'],r['case_id'],r['attempt']):r for r in scored}
    sites=[]
    for model in manifest['models']:
        for task in tasklist:
            rows=[r for r in allrows if r['model']==model and r['case_id']==task['case_id']]
            good=[r for r in rows if r['strict_success'] and scoremap[(model,r['case_id'],r['attempt'])].get('posebusters_pass',False)]
            novel=[r for r in good if r.get('new_design')]
            improved=[r for r in novel if scoremap[(model,r['case_id'],r['attempt'])].get('delta_vina_kcal_mol',0)<0 and scoremap[(model,r['case_id'],r['attempt'])].get('delta_cnn_affinity_pK',0)>0]
            sites.append(dict(model=model,case_id=task['case_id'],pdb_id=task['pdb_id'],stratum=task['stratum'],
                attempts=len(rows),strict_success=sum(r['strict_success'] for r in rows),
                strict_and_posebusters_pass=len(good),new_strict_and_posebusters_attempts=len(novel),
                unique_new_strict_and_posebusters=len({r['smiles'] for r in novel}),
                new_improved_both_attempts=len(improved),unique_new_improved_both=len({r['smiles'] for r in improved})))
    with (OUT/'per_site.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(sites[0]));w.writeheader();w.writerows(sites)
    write_json(OUT/'status.json',dict(stage='complete' if not failed else 'needs_inspection',
        generation_attempts=22400,scored_candidates=len(scored),scored_references=45,new_score_jobs=len(jobs),scoring_failures=len(failed)))
    print(json.dumps(read_status:=json.loads((OUT/'status.json').read_text())),flush=True)


if __name__=='__main__':main()
