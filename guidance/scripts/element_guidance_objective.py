"""Local expected-score surrogate over chemically valid element substitutions.

The GNINA scores are detached coefficients for the current denoised structure.
Only the softmax expectation is differentiated. This is NOT an exact derivative
of GNINA through discrete chemistry or a pretrained noisy-state classifier.
"""
import numpy as np
import torch
from rdkit import Chem, rdBase
from posebusters.modules.distance_geometry import check_geometry

NUMBERS=[6,7,8,16,5,35,17,15,53,9]


def enumerate_substitutions(mol,ncore):
    candidates=[Chem.Mol(mol)];labels=[None];rejected=[]
    xyz=mol.GetConformer().GetPositions()
    edges=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx(),str(b.GetBondType())) for b in mol.GetBonds()]
    for i in range(ncore,mol.GetNumAtoms()):
        current=mol.GetAtomWithIdx(i)
        for kind,number in enumerate(NUMBERS):
            if number==current.GetAtomicNum():continue
            edit=Chem.RWMol(mol);atom=edit.GetAtomWithIdx(i)
            atom.SetAtomicNum(number)
            atom.SetNumExplicitHs(0);atom.SetNoImplicit(False);atom.SetNumRadicalElectrons(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            # Retain formal charge and all heavy-atom edges. Recompute implicit
            # hydrogens for the proposed element; do not repair invalid valence.
            candidate=edit.GetMol()
            try:
                with rdBase.BlockLogs():
                    Chem.SanitizeMol(candidate)
                    Chem.AssignStereochemistryFrom3D(candidate,replaceExistingTags=True)
                if [(b.GetBeginAtomIdx(),b.GetEndAtomIdx(),str(b.GetBondType())) for b in candidate.GetBonds()]!=edges:
                    raise ValueError('Substitution changed bond orders')
                if any(a.GetNumRadicalElectrons() for a in candidate.GetAtoms()):
                    raise ValueError('Radical substitution')
                changed=candidate.GetAtomWithIdx(i)
                if number in [9,17,35,53] and (changed.GetDegree()!=1 or changed.GetTotalValence()!=1):
                    raise ValueError('Only monovalent halogen substitutions allowed')
                # Existing PoseBusters distance-geometry checks, independent of
                # affinity. This constrains proposed substitutions, not final
                # success counting or resampling of generated outputs.
                with rdBase.BlockLogs():
                    geometry=check_geometry(candidate)['results']
                if not all(geometry[k] for k in ['bond_lengths_within_bounds','bond_angles_within_bounds','no_internal_clash']):
                    raise ValueError('Substitution fails intramolecular distance-geometry checks')
                if not np.array_equal(xyz,candidate.GetConformer().GetPositions()):
                    raise AssertionError('Substitution changed coordinates')
                candidates.append(candidate);labels.append((i,kind))
            except AssertionError:
                raise
            except Exception as exc:
                rejected.append(dict(atom=i,kind=kind,error=str(exc)[:160]))
    return candidates,labels,rejected


def score_table(mol,labels,scores,ncore):
    if scores[0] is None or not np.isfinite(scores[0]):
        raise ValueError('Current intermediate molecule has no finite GNINA score')
    values=np.zeros((mol.GetNumAtoms(),len(NUMBERS)),dtype=np.float64)
    allowed=np.zeros_like(values,dtype=bool)
    for i in range(ncore,mol.GetNumAtoms()):
        allowed[i,NUMBERS.index(mol.GetAtomWithIdx(i).GetAtomicNum())]=True
    for label,score in zip(labels[1:],scores[1:]):
        if score is not None and np.isfinite(score):
            i,k=label;allowed[i,k]=True;values[i,k]=score-scores[0]
    return values,allowed


def expected_improvement(features,values,allowed,ncore,temperature=0.5):
    """Sum of per-site expected single-substitution score differences.

Multiple simultaneous substitutions are only a first-order approximation;
their combined score is not assumed additive at final evaluation.
"""
    logits=features[ncore:]/temperature
    permit=torch.as_tensor(allowed[ncore:],device=features.device)
    gains=torch.as_tensor(values[ncore:],dtype=features.dtype,device=features.device)
    probabilities=torch.softmax(logits.masked_fill(~permit,float('-inf')),dim=-1)
    return (probabilities*gains).sum()


def feature_gradient(features,values,allowed,ncore,temperature=0.5):
    with torch.enable_grad():
        h=torch.as_tensor(features,dtype=torch.float64).detach().requires_grad_(True)
        objective=expected_improvement(h,values,allowed,ncore,temperature)
        grad,=torch.autograd.grad(objective,h)
    g=grad.detach().numpy()
    if not np.isfinite(g).all() or np.any(g[:ncore]!=0):
        raise ValueError('Invalid or non-R feature gradient')
    return float(objective.detach()),g
