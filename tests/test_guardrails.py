"""Scientific controls and budget/archive checks; no model weights are needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import tasks
from benchmark_io import read_saved, save_molecules
from generate import run_cases
from metrics import REQUIRED_CHECKS, outcomes
from summarize import aggregate


def conformer(smiles):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(molecule, randomSeed=123) != 0:
        raise ValueError('Synthetic test fixture embedding failed')
    return Chem.RemoveHs(molecule)


class AuditControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.reference = conformer('CCc1ccccc1')
        with patch.object(tasks, 'ROOT', self.root):
            self.task = tasks.make_tasks(self.reference, 'fixture', self.root/'protein.pdb', 'synthetic_test')[0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_reference_passes_and_invalid_chemistry_keeps_failure(self):
        self.assertTrue(tasks.audit(self.reference, self.task)['strict_success'])
        self.assertFalse(tasks.audit(None, self.task)['strict_success'])

    def test_wrong_anchor_and_moved_core_are_rejected(self):
        wrong = dict(self.task, anchor_in_core=(self.task['anchor_in_core']+1) % self.task['fixed_heavy_atoms'])
        self.assertFalse(tasks.audit(self.reference, wrong)['single_site'])
        moved = Chem.Mol(self.reference)
        for i in self.task['fixed_reference_indices']:
            p = moved.GetConformer().GetAtomPosition(i)
            moved.GetConformer().SetAtomPosition(i, (p.x+1., p.y, p.z))
        self.assertFalse(tasks.audit(moved, self.task)['core_pose_preserved'])

    def test_charge_and_fixed_bonds_are_audited(self):
        charged = Chem.Mol(self.reference)
        charged.GetAtomWithIdx(self.task['fixed_reference_indices'][0]).SetFormalCharge(1)
        self.assertFalse(tasks.audit(charged, self.task)['strict_success'])
        edited = Chem.RWMol(self.reference)
        for b in edited.GetBonds():
            if b.GetBeginAtomIdx() in self.task['fixed_reference_indices'] and b.GetEndAtomIdx() in self.task['fixed_reference_indices']:
                edited.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                break
        self.assertFalse(tasks.audit(edited.GetMol(), self.task)['strict_success'])

    def test_exact_size_is_an_outcome(self):
        wrong = dict(self.task, reference_heavy_atoms=self.task['reference_heavy_atoms']+1)
        self.assertFalse(tasks.audit(self.reference, wrong)['size_match'])

    def test_fixed_budget_counts_every_failed_attempt_and_resume_skips_completed_case(self):
        seen = []
        def adapter(inputs, count):
            self.assertNotIn('reference_path', inputs)
            self.assertNotIn('reference_smiles', inputs)
            self.assertNotIn('removed_reference_indices', inputs)
            seen.append(count)
            return [None]*count
        dest = self.root/'generated'
        with patch('torch.cuda.is_available', return_value=False):
            run_cases(adapter, [self.task], dest, 'fixture', 5, 2, 11, 'case_seed_once')
            run_cases(adapter, [self.task], dest, 'fixture', 5, 2, 11, 'case_seed_once')
        self.assertEqual(seen, [2, 2, 1])
        folder = dest/self.task['case_id']
        rows = json.loads((folder/'attempts.json').read_text())
        self.assertEqual([r['attempt'] for r in rows], list(range(5)))
        self.assertEqual(json.loads((folder/'summary.json').read_text())['attempts'], 5)
        self.assertTrue(all(not r['strict_success'] for r in rows))

    def test_adapter_count_mismatch_stops_instead_of_replenishing(self):
        def bad_adapter(inputs, count):
            return [None]*(count+1)
        with self.assertRaisesRegex(RuntimeError, 'output count mismatch'):
            run_cases(bad_adapter, [self.task], self.root/'bad', 'fixture', 4, 2, 11, 'case_seed_once')


class ArchiveAndMetrics(unittest.TestCase):
    def test_aromatic_h_and_double_precision_coordinates_survive_export(self):
        molecule = conformer('CCc1cc[nH]c1')
        before = Chem.MolToSmiles(molecule)
        coordinates = molecule.GetConformer().GetPositions().copy()
        with tempfile.TemporaryDirectory() as tmp:
            save_molecules(Path(tmp), [(7, molecule)])
            restored = read_saved(tmp, 7)
        self.assertEqual(Chem.MolToSmiles(restored), before)
        np.testing.assert_array_equal(restored.GetConformer().GetPositions(), coordinates)
        np.testing.assert_array_equal(molecule.GetConformer().GetPositions(), coordinates)

    def test_missing_energy_check_cannot_pass(self):
        attempt = dict(strict_success=True, new_design=True, charge_match=True)
        reference = dict(status='scored', vina_kcal_mol=-5., cnn_affinity_pK=5.)
        score = dict(status='scored', vina_kcal_mol=-6., cnn_affinity_pK=6.,
                     posebusters_checks={key: True for key in REQUIRED_CHECKS})
        self.assertTrue(outcomes(attempt, score, reference)['dual_improved'])
        del score['posebusters_checks']['internal_energy']
        output = outcomes(attempt, score, reference)
        self.assertFalse(output['valid_new'])
        self.assertIn('internal_energy', output['missing_geometry_checks'])

    def test_raw_denominator_and_equal_target_weight_are_different(self):
        from summarize import COUNT_KEYS
        rows = []
        for group, count in [('A', 100), ('A', 100), ('B', 0)]:
            row = {key: 0 for key in COUNT_KEYS}
            row.update(attempts=100, valid_new=count, dual_improved=count, target_group=group)
            rows.append(row)
        result = aggregate('fixture', rows)
        self.assertEqual(result['attempts'], 300)
        self.assertAlmostEqual(result['valid_new_percent'], 200/3)
        self.assertEqual(result['target_equal_weight_valid_new_percent'], 50.)


if __name__ == '__main__':
    unittest.main()
