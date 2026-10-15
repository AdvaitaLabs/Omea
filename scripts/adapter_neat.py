# SPDX-License-Identifier: MIT
# Upstream source: https://github.com/molinfo-vienna/NEAT-POCKET/blob/ce7de3f23d28b99f5316903220d4bacfa2757463/LICENSE
# Adapter changes: retained-core inputs, fixed raw budgets; see docs/methods.md.
"""NEAT-POCKET official SPINDR checkpoint, native fragment completion."""
import hashlib
import importlib.util
import inspect
import json
import time
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from tasks import ROOT, read_mol, write_sdf


def make_neat(args):
    from neat_attention_compat import install
    backend = install()
    repo = ROOT / 'vendor/NEAT-POCKET'
    sys.path.insert(0, str(repo / 'src'))
    from neat.model import NEAT
    from neat.model.bond_predictor import BondPredictor
    from neat.model.molecule_builder import MoleculeBuilder
    from neat.dataset.dataset_spindr import SpindrDataSet
    spec = importlib.util.spec_from_file_location('neat_pdb_helpers', repo / 'scripts/generation_from_pdb.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    # Native Hydride relaxation can cycle indefinitely at constant energy.
    # Bound only this preprocessing step, retaining the native objective and
    # fixed protein heavy atoms. Every case records whether it reached the cap.
    preparation_reports = {}
    def bounded_relax(structure, protein_path):
        import hydride
        started = time.perf_counter()
        heavy = structure.element != 'H'
        before = structure.coord[heavy].copy()
        xyz, energies = hydride.relax_hydrogen(structure, iterations=1000, return_energies=True)
        assert np.array_equal(xyz[heavy], before)
        preparation_reports[str(protein_path)] = dict(
            method='native Hydride hydrogen relaxation with finite iteration limit',
            max_iterations=1000, iterations=len(energies),
            iteration_limit_reached=len(energies)>=1000,
            final_energies=[float(x) for x in energies[-10:]],
            protein_heavy_atoms_unchanged=True, seconds=time.perf_counter()-started)
        return xyz
    helper._benchmark_relax_hydrogen = bounded_relax
    loader_source = inspect.getsource(helper.load_protein_structure)
    needle = 'structure.coord = hydride.relax_hydrogen(structure)'
    assert loader_source.count(needle) == 1
    loader_source = loader_source.replace(needle,
        'structure.coord = _benchmark_relax_hydrogen(structure, protein_path)')
    exec(compile(loader_source, 'neat_bounded_protein_preparation', 'exec'), helper.__dict__)
    snapshot = ROOT / 'results' / args.run / args.model / 'source'
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / 'neat_load_protein_structure_bounded.py').write_text(loader_source)
    checkpoints = ROOT / 'checkpoints/neat_pocket/trained_models'
    checkpoint = checkpoints / 'neat_spindr/checkpoints/periodic-epoch=4999.ckpt'
    bc = checkpoints / 'bond_predictor/bp_spindr/checkpoints/bond_predictor_SPINDR-epoch=17.ckpt'
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = NEAT(**data['hyper_parameters'])
    model.load_state_dict(data['state_dict'], strict=True)
    del data
    model = model.to('cuda').eval().requires_grad_(False)
    data = torch.load(bc, map_location='cpu', weights_only=True)
    bond = BondPredictor(**data['hyper_parameters'])
    bond.load_state_dict(data['state_dict'], strict=True)
    bond = bond.to('cuda').eval().requires_grad_(False)
    del data
    builder = MoleculeBuilder(vocab='SPINDR')
    cache = {}

    def generate(task, count):
        if task['case_id'] not in cache:
            dest = ROOT / 'results' / args.run / args.model / task['case_id'] / 'inputs'
            dest.mkdir(parents=True, exist_ok=True)
            core = read_mol(task['core_path'])
            # Match native hydrogen-explicit fragments using only retained-core
            # chemistry. Remove the temporary cap H at the declared R anchor.
            capped = Chem.AddHs(core, addCoords=True)
            anchor = capped.GetAtomWithIdx(task['anchor_in_core'])
            hs = [a.GetIdx() for a in anchor.GetNeighbors() if a.GetAtomicNum() == 1]
            if not hs: raise ValueError('Cannot identify temporary anchor cap H')
            edit = Chem.RWMol(capped); edit.RemoveAtom(max(hs)); fragment = edit.GetMol()
            write_sdf(dest / 'capped_core.sdf', [capped])
            pocket = helper.extract_spindr_pocket(task['protein_path'], task['core_path'],
                                                  pocket_cutoff=6., add_hs_to_protein=True)
            (dest / 'hydrogen_preparation.json').write_text(json.dumps(preparation_reports[str(task['protein_path'])], indent=2))
            pocket.write_pdb(dest / 'pocket.pdb', include_bonds=True)
            data = helper.preprocess_pocket(dest / 'pocket.pdb', dest / 'capped_core.sdf')
            center = helper.compute_preprocess_center(dest / 'capped_core.sdf')
            fragment = helper.translate_mol(fragment, -center.numpy())
            pocket_center = np.asarray(pocket.atoms.coord, dtype=np.float64).mean(0)
            cache[task['case_id']] = (data, fragment, pocket_center, core.GetConformer().GetPositions())
        data, fragment, center, core_xyz = cache[task['case_id']]
        pi = SpindrDataSet.collate_pocket_info([data], samples_per_pocket=count, device='cuda')
        fi = helper.prepare_fragment_info(fragment, count)
        output = model.generate(batch_size=count, max_atoms=100, num_time_steps=60,
            time_step_spacing='linear', integration_method='euler_maruyama',
            cfg_factor=.5, pocket_info=pi, fragment_info=fi, device=torch.device('cuda'))
        output.pos += torch.as_tensor(center, dtype=output.pos.dtype, device=output.pos.device)
        for i in range(count):
            fixed = output.pos[output.batch == i][:len(core_xyz)].detach().cpu().numpy()
            if np.max(np.abs(fixed-core_xyz)) > .01:
                raise ValueError('NEAT fixed core did not round trip in the protein frame')
        mols = builder.generate_rdkit_molecules_via_bond_predictor(
            output.x, output.pos, output.batch, bond_predictor=bond, progress_bar=False)
        if len(mols) != count: raise ValueError('NEAT reconstruction changed raw attempt count')
        return mols

    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            while chunk := f.read(8*1024*1024): h.update(chunk)
        return h.hexdigest()
    return generate, dict(checkpoint=str(checkpoint), sha256=sha(checkpoint),
        bond_checkpoint=str(bc), bond_checkpoint_sha256=sha(bc),
        source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        source_patch_sha256=hashlib.sha256(subprocess.check_output(['git','-C',str(repo),'diff','--binary'])).hexdigest(),
        adapter_sha256=sha(Path(__file__)), attention_backend=backend,
        attention_compat_sha256=sha(Path(__file__).parent/'neat_attention_compat.py'),
        protein_hydrogen_relaxation_max_iterations=1000,
        protein_preparation_patch_sha256=hashlib.sha256(loader_source.encode()).hexdigest(),
        sampler='native fragment_info completion', max_added_atoms_including_H=100,
        native_exact_size_condition=False, size_policy='native stop token; required R heavy-atom count assessed as outcome',
        native_anchor_condition='uncapped anchor in hydrogen-explicit fixed fragment; attachment assessed by common audit',
        fragment_hydrogens='RDKit AddHs from retained core; remove one temporary anchor cap H',
        input='fixed core and pocket only; no deleted R atom identities, bonds, coordinates or properties',
        pocket_radius_A=6., flow_steps_per_atom=60, cfg_factor=.5, post_filters=False)
