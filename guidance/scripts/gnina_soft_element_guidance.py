"""A differentiable 10-element density relaxation of the frozen GNINA CNN.

No element-substitution molecules are scored. Each R atom contributes a mixture
of element-specific densities. Chemical-role assignments are detached local
context, not derivatives through discrete bonds or a chemistry oracle.
"""
import numpy as np
import torch
import molgrid
from rdkit import Chem
from gnina_guidance import CoordinateScorer

NUMBERS = [6, 7, 8, 16, 5, 35, 17, 15, 53, 9]
RADII = [1.9, 1.8, 1.7, 2.0, 1.92, 2.0, 1.8, 2.1, 2.2, 1.5]
CHANNELS = [0, 7, 9, 11, 13, 4, 5, 12, 4, 6]
EXPECTED_NAMES = ['AliphaticCarbonXSHydrophobe', 'AliphaticCarbonXSNonHydrophobe',
                  'AromaticCarbonXSHydrophobe', 'AromaticCarbonXSNonHydrophobe',
                  'Bromine_Iodine', 'Chlorine', 'Fluorine',
                  'Nitrogen_NitrogenXSAcceptor', 'NitrogenXSDonor_NitrogenXSDonorAcceptor',
                  'Oxygen_OxygenXSAcceptor', 'OxygenXSDonorAcceptor_OxygenXSDonor',
                  'Sulfur_SulfurAcceptor', 'Phosphorus',
                  'GenericMetal_Boron_Manganese_Magnesium_Zinc_Calcium_Iron']


class SoftElementScorer(CoordinateScorer):
    def __init__(self):
        super().__init__('default')
        assert list(self.lig_typer.get_type_names()) == EXPECTED_NAMES
        self.core_cache = {}

    def encoding(self, core, xyz, context=None):
        """Freeze chemical-role context while keeping element weights continuous.

        The current hard element gets its exact GNINA type/radius if typing is
        available. Other elements use local aromatic/neighbor/H context. Without
        a reconstructable state, C/N/O role uncertainty uses fixed mixtures;
        this fallback does NOT disable the gradient or choose another output.
        """
        nc, nr = core.GetNumAtoms(), len(xyz)-core.GetNumAtoms()
        assert nr > 0
        key = Chem.MolToMolBlock(core)
        if key not in self.core_cache:
            self.core_cache[key] = self.ligand_types(core)
        core_types, core_radii = self.core_cache[key]
        current_types = current_radii = None
        context_status = 'graph_free_role_mixture'
        if context is not None:
            assert context.GetNumAtoms() == len(xyz)
            try:
                current_types, current_radii = self.ligand_types(context)
                core_types, core_radii = current_types[:nc], current_radii[:nc]
                context_status = 'current_reconstruction_roles'
            except ValueError:
                context = None
        mapping = np.zeros((nr, 10, self.nt), dtype=np.float32)
        radii = np.tile(np.asarray(RADII, dtype=np.float32), (nr, 1))
        for j in range(nr):
            for k, channel in enumerate(CHANNELS):
                mapping[j, k, self.nr+channel] = 1.
            if context is None:
                for k, pair in [(0, [0, 1]), (1, [7, 8]), (2, [9, 10])]:
                    mapping[j, k] = 0
                    mapping[j, k, self.nr+np.array(pair)] = .5
            else:
                atom = context.GetAtomWithIdx(nc+j)
                # This role choice is local context, held fixed during autograd.
                aromatic = bool(atom.GetIsAromatic())
                nonhydrophobic = any(a.GetAtomicNum() not in [1, 6] for a in atom.GetNeighbors())
                mapping[j, 0] = 0
                mapping[j, 0, self.nr+2*int(aromatic)+int(nonhydrophobic)] = 1.
                donor_context = atom.GetTotalNumHs() > 0
                mapping[j, 1] = 0
                mapping[j, 1, self.nr+(8 if donor_context else 7)] = 1.
                mapping[j, 2] = 0
                mapping[j, 2, self.nr+(10 if donor_context else 9)] = 1.
                number = atom.GetAtomicNum()
                assert number in NUMBERS
                k = NUMBERS.index(number)
                mapping[j, k] = current_types[nc+j].detach().cpu().numpy()
                radii[j, k] = float(current_radii[nc+j])
        assert np.allclose(mapping.sum(axis=-1), 1.)
        return core_types, core_radii, self.tensor(mapping), self.tensor(radii), context_status

    def relaxed_score(self, x, h, protein, center, encoding, temperature=.2, hard_probabilities=None):
        """The full differentiable chain h -> softmax -> densities -> CNN score."""
        nc = len(encoding[0])
        core_types, core_radii, mapping, element_radii, _ = encoding
        rx, rt, rr = self.receptor(protein)
        probabilities = torch.softmax(h[nc:]/temperature, dim=-1) if hard_probabilities is None else hard_probabilities
        # Virtual density components share one physical atom coordinate. They
        # are grid interpolation terms, not additional atoms in generated output.
        virtual_coords = x[nc:, None, :].expand(-1, 10, -1).reshape(-1, 3)
        virtual_types = (probabilities[..., None]*mapping).reshape(-1, self.nt)
        coords = torch.cat([rx, x[:nc], virtual_coords], dim=0).contiguous()
        types = torch.cat([rt, core_types, virtual_types], dim=0).contiguous()
        radii = torch.cat([rr, core_radii, element_radii.reshape(-1)], dim=0).contiguous()
        grid = molgrid.Coords2Grid(self.gmaker, center=tuple(float(v) for v in center))(coords, types, radii)
        outputs = self.model(grid.unsqueeze(0))
        return outputs[1].reshape(()), outputs[0].exp()[0, 1]

    def joint_gradient(self, core, xyz, features, protein, center, temperature=.2, context=None):
        assert np.shape(features) == (len(xyz), 10) and temperature > 0
        encoding = self.encoding(core, xyz, context)
        with torch.enable_grad():
            x = self.tensor(xyz).detach().requires_grad_(True)
            h = self.tensor(features).detach().requires_grad_(True)
            affinity, pose = self.relaxed_score(x, h, protein, center, encoding, temperature)
            gx, gh = torch.autograd.grad(affinity, (x, h))
        assert torch.isfinite(gx).all() and torch.isfinite(gh).all()
        nc = core.GetNumAtoms()
        assert torch.count_nonzero(gh[:nc]) == 0
        diagnostics = dict(encoding_context=encoding[-1], model_forward_calls=1, backward_calls=1,
                           scored_element_substitution_candidates=0,
                           coordinate_gradient_norms=torch.linalg.vector_norm(gx[nc:], dim=-1).cpu().tolist(),
                           feature_gradient_norms=torch.linalg.vector_norm(gh[nc:], dim=-1).cpu().tolist(),
                           temperature=temperature, type_role_assignment_detached=True,
                           all_R_features_participate=True)
        return float(affinity.detach()), float(pose.detach()), gx.cpu().numpy(), gh.cpu().numpy(), diagnostics
