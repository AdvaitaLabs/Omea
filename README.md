# R-group design benchmark

Compare 13 molecular generation methods on single-attachment-site R-group replacement within protein pockets: preserve the structure and coordinates of all remaining atoms and regenerate only one R-group at a specified site. Models use official pretrained weights and are evaluated consistently for structural constraints, geometric quality, and proxy scores relative to the original ligand.

This task draws on DiffSBDD's scaffold-preserving design approach and extends it to a strict single-attachment-site setting. It is not a point-by-point reproduction of the paper's original experiments, and proxy scores are not treated as true ΔΔG values.

The repository also includes joint CNN guidance of coordinates and element features, local geometry constraints, and coordinate-update projection for AutoFragDiff. The implementation scope ends at the version using native final reconstruction reported in the 2026-10-25 weekly report.

## Benchmark setup

- A fixed set of 45 cases, with **100 raw generation attempts** per method per case. Each case combines a protein structure, an original ligand, and a specified replacement site.
- The main comparison uses a shared set of 30 multi-atom R-group cases spanning 7 target groups. Single-atom R-group cases are also included, and results across all cases are summarized separately. The case list is in `cohorts/single_site_45.json`.
- Preserve the core's elements, formal charges, aromaticity, internal bonds, stereochemistry, and positions in the protein coordinate frame. Only one SINGLE bond may be attached at the specified core atom. The R-group must contain the same number of heavy atoms as the removed fragment.
- Failures, incorrect sizes, and duplicate molecules all count toward the raw attempt budget. There are no replacement attempts or best-of-K selection. Unique molecule counts are reported separately.
- The generator receives the protein, retained core, attachment site, and R-group atom count. The original R-group's elements, bonds, coordinates, and scores are not provided to the generator.

The full protocol is defined in [configs/protocol.json](configs/protocol.json). Overlap with training data in the public datasets has not yet been ruled out, so these results cannot be described as a ranking on a strictly independent test set.

## Metrics

| Metric | Definition |
|---|---|
| Strict success | The chemical structure can be parsed, and all constraints on the core, attachment site, connectivity, atom count, and positions are satisfied |
| Valid new | Strict success, plus passing the full PoseBusters geometry checks, successful scoring, and a chemical structure different from the original ligand |
| Vina | Vina score of the generated pose, in kcal/mol; lower is better. No redocking is performed |
| GNINA CNNaffinity | Neural network affinity prediction for the generated pose, on a pK scale; higher is better |
| Dual improvement | Valid new, with a lower Vina score and a higher CNNaffinity score than the original ligand |

Success rates use all raw attempts as the denominator. Missing PoseBusters checks do not count as passes. All methods and original ligands use the same hydrogen addition, hydrogen-only optimization, and CPU GNINA 1.1 scoring procedure. The maximum core RMSD is 0.1 Å, and the maximum displacement of any individual atom is 0.3 Å.

Results are also reported with equal weighting by case and by target, so targets with more cases do not dominate the overall score. Vina and GNINA are fast proxies and cannot be directly interpreted as improvements in experimental affinity or free-energy ΔΔG. This repository does not run OpenFE.

## AutoFragDiff guidance optimization

During generation, only GNINA CNNaffinity is used to guide coordinates and element features, with local geometry constraints added in the late stages. Vina is used for evaluation after the configurations are frozen; it is not used for guidance or configuration selection in this round. Each configuration uses the same 30 cases with 100 attempts per case, and failures remain in the denominator.

| Configuration | Dual improvement count | Dual improvement rate |
|---|---:|---:|
| No score guidance | 360 / 3000 | 12.00% |
| Score guidance with local geometry constraints | 492 / 3000 | 16.40% |
| Stronger local geometry constraints | 494 / 3000 | 16.47% |
| Restrict coordinate updates that worsen local geometry | 495 / 3000 | **16.50%** |

These are verified results from the original development-set experiments, not independent test results obtained by rerunning this release. Final outputs still use native reconstruction; subsequent experiments on preserving the core in the final output are outside the scope of this version. See the [guidance optimization notes](docs/guidance.md) for the main implementation, parameters, and runtime dependencies, and the [validation summary](docs/guidance_validation.json) for source verification.

## Repository layout

```text
benchmark.py       Unified command entry point
scripts/           Data preparation, adapters for 13 methods, generation, scoring, and aggregation
guidance/          Joint CNN guidance, local geometry constraints, and experiment scripts from the weekly report
configs/           Benchmark rules, method metadata, official resources, and weight SHA256 checksums
cohorts/           Fixed benchmark cases and the official MOAD split
requirements/      Dependency files and version records from the actual runtime environments
patches/           Compatibility patches for pinned upstream versions
licenses/          Original upstream license texts
tests/             Checks for core constraints, fixed budgets, persistence, and evaluation rules
docs/              Installation, method adaptation notes, and known issues
```

Protein structures, weights, complete third-party repositories, and per-attempt generation results are not included in the release directory. The guidance component retains the corresponding version's code, parameters, and statistical summaries; resources downloaded at runtime and generated outputs are excluded by `.gitignore`.

## Usage

First, follow the [installation instructions](docs/installation.md) to set up the CPU evaluation environment and the generation environment for your selected model. Different models require different Lightning/PyTorch versions, so installing them all in a single environment is not recommended.

```bash
# List the 13 methods and their corresponding environments
python benchmark.py models

# Fetch the pinned upstream source and the model's weights
python benchmark.py setup-sources --model AutoFragDiff
python benchmark.py assets download --model AutoFragDiff

# Download, prepare, and verify the fixed cases using public MOAD and PDB CCD data
python benchmark.py prepare
python benchmark.py validate --unit-tests

# Run in the model's generation environment; defaults to 100 attempts per case
python benchmark.py generate --model AutoFragDiff --run benchmark_v1 --cohort primary

# Score in the CPU evaluation environment using GNINA 1.1
python benchmark.py score --model AutoFragDiff --run benchmark_v1 --gnina /path/to/gnina-1.1
python benchmark.py summarize --run benchmark_v1 --models AutoFragDiff
```

The commands above explicitly select the 30 shared cases. All models should use the same `--cohort primary` setting. The default `--cohort all` includes all 45 prepared cases, while `--cohort single_atom` includes 14 single-atom cases. Input 1RVX, which FLOWR.root is known not to support, is outside the main comparison set. Running FLOWR.root directly on all 45 cases fails early rather than fabricating counts for attempts that were never executed.

If you already have the weights, place them at the paths specified in `configs/assets.json`, then run `assets verify --model ...`. If you have a complete MOAD/CCD cache, use `prepare --from-cache`; if you already have a task pool, use `prepare --freeze-only`.

By default, the `benchmark.py` commands above store resources and outputs in this repository directory. You can instead set `RGROUP_BENCHMARK_ROOT=/path/to/workspace` to store data, upstream source, tools, weights, and results in a separate workspace. The command entry point can be invoked from any working directory. Relative paths within task manifests are resolved against that workspace root. The guidance experiment scripts retain the original experiment paths and do not read this environment variable; see the [guidance optimization notes](docs/guidance.md) for setup instructions.

Small installation checks must be explicitly marked and must not be included in official results:

```bash
python benchmark.py generate --model AutoFragDiff --run smoke_v1 --smoke --limit 1 --attempts 2
python benchmark.py score --model AutoFragDiff --run smoke_v1 --gnina /path/to/gnina-1.1
python benchmark.py summarize --run smoke_v1 --models AutoFragDiff --smoke
```

Runs can resume from completed cases; interrupted cases are rerun in full from their original seeds. Use a new run name whenever inputs, code, weights, or budgets change. Run manifests record checksums for inputs, code, and weights, and the full RDKit state of the raw molecules is preserved. SDF is used only as an export format.

Summary outputs are written to `results/<run>/summary/`: `per_case.csv`, `comparison_primary_30.csv`, `comparison_all_cases.csv`, `per_target_primary.csv`, and `summary.json`. In the aggregate tables, `unique_*` is the sum of per-case unique counts, not the total number of molecules deduplicated across pockets.

## Methods and scope of use

See [docs/methods.md](docs/methods.md) for models, papers, licenses, and adaptation notes. The actual adaptations of MolJO, DiffGui, MolPilot, and DELETE remain explicitly identified in configurations and outputs; they are not presented as unmodified native samplers.

The 13-model benchmark uses a new output layout and fixes the Pfizer diffusion mask-binding issue identified in the release code. The corresponding historical tables have not been recomputed or published alongside these code changes. The newly added guidance results are retained only as statistical summaries of their original experiments and must not be combined with new runs from this release. See [docs/known_issues.md](docs/known_issues.md) for details.

Licensing varies by file scope: independent benchmark orchestration, data preparation, and evaluation code use the MIT License; upstream adaptations and patches retain their respective licenses, particularly the noncommercial restrictions for MolJO/MolPilot. Weight licenses are separate from source-code licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
