# Third-party notices

The independently authored benchmark orchestration, data preparation and evaluation are covered by the scoped root MIT license. Upstream-derived adapters and patches retain the following source licenses and attribution; the root MIT license does not relicense them.

Model weights, GNINA binaries, DELETE surface tools, public structures and complete upstream repositories are not distributed here. Their terms apply separately when obtained from their providers.

| Included adapter / patch | Upstream source and attribution | License notice |
|---|---|---|
| adapter_autofragdiff.py | [AutoFragDiff](https://github.com/keiserlab/autofragdiff/blob/84f0885cb12e6ac4abc7558870f8d304c78c8a38/LICENSE) | [MIT](licenses/AutoFragDiff-LICENSE.txt) |
| adapter_diffusion.py | [DiffDec](https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/LICENSE) | [MIT](licenses/DiffDec-LICENSE.txt) |
| adapter_diffusion.py | [DiffSBDD](https://github.com/arneschneuing/DiffSBDD/blob/5d0d38d16c8932a0339fd2ce3f67ade98bbdff27/LICENSE) | [MIT](licenses/DiffSBDD-LICENSE.txt) |
| adapter_delete.py | [DELETE](https://github.com/OdinZhang/Delete/blob/ba57693822ac82950bf0aa081b788772039373de/LICENSE) | [MIT](licenses/DELETE-LICENSE.txt) |
| adapter_moljo.py / patches/MolCRAFT.patch (MolJO portions) | [MolJO](https://github.com/GenSI-THUAIR/MolCRAFT/blob/743dfb5198c44ca0cec180822bd54258f9d386f1/MolJO/LICENSE) | [CC-BY-NC-SA-4.0](licenses/MolJO-LICENSE.txt) |
| adapter_molpilot.py | [MolPilot](https://github.com/GenSI-THUAIR/MolCRAFT/blob/743dfb5198c44ca0cec180822bd54258f9d386f1/MolPilot/LICENSE) | [CC-BY-NC-SA-4.0](licenses/MolPilot-LICENSE.txt) |
| adapter_diffgui.py | [DiffGui](https://github.com/QiaoyuHu89/DiffGui/blob/f2f90b560ad32db9d584a3b7b153420dc1e8d258/LICENSE) | [MIT](licenses/DiffGui-LICENSE.txt) |
| adapter_pocketxmol.py | [PocketXMol](https://github.com/pengxingang/PocketXMol/blob/65488cf635c856101dbe703ac97e2f10f58e005c/LICENSE) | [MIT](licenses/PocketXMol-LICENSE.txt) |
| adapter_pfizer.py | [Pfizer-Coupled-flow](https://github.com/pfizer-opensource/pilot-sbdd/blob/5b7e668ea88baac175536d804599e59236b60afd/LICENSE) | [Apache-2.0](licenses/PfizerCoupled-LICENSE.txt) |
| adapter_pfizer.py | [Pfizer-Coupled-diffusion](https://github.com/pfizer-opensource/pilot-sbdd/blob/5b7e668ea88baac175536d804599e59236b60afd/LICENSE) | [Apache-2.0](licenses/PfizerCoupled-LICENSE.txt) |
| adapter_flowr.py | [FLOWR.root](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/LICENSE) | [MIT](licenses/flowr_root-LICENSE.txt) |
| adapter_neat.py | [NEAT-POCKET](https://github.com/molinfo-vienna/NEAT-POCKET/blob/ce7de3f23d28b99f5316903220d4bacfa2757463/LICENSE) | [MIT](licenses/NEAT-POCKET-LICENSE.txt) |
| adapter_lddm.py | [LDDM](https://github.com/LPDI-EPFL/lddm/blob/8fac7133c29ccdce576eb38bf7d43c09df9cb498/LICENSE.md) | [MIT](licenses/LDDM-LICENSE.md.txt) |


Upstream files contain their own copyright notices, preserved verbatim in licenses/. Local changes are described in docs/methods.md and CHANGELOG.md; all source revisions and patch checksums are in configs/sources.json. The MolCRAFT patch includes a path compatibility change to its SA scorer, which has its own GPLv2 attribution. Reconstruction implementations called from AutoFragDiff, DiffGui and MolJO also carry GPLv2 notices; they remain in external source checkouts rather than being copied into this repository.

The MOAD test identifiers originate from the fixed MIT-licensed DiffSBDD repository. The 45-case single-site manifest and target-group annotations are benchmark-derived metadata; molecular coordinate files are fetched separately. No general commercial-use claim is made for the collection of optional models and assets.
