# guidance_report_20261025

- Extend the existing benchmark export with the weekly-report CNN coordinate/element guidance, local geometry constraints and coordinate-update projection.
- Keep 47 historical experiment scripts under guidance/, including the 45 frozen sources, without replacing the benchmark's task or scoring modules.
- Record parameters, upstream revisions, environment versions and the audited 16.50% native-final result; keep Vina evaluation-only for this experiment.
- Retain the original README and benchmark commands; add the guidance implementation and runtime notes.
- Exclude the subsequent final core-preservation experiment, precision repair and independent verification code.

# github_release_v1

- Extract the 13 benchmark adapters and fixed 45-case/30-case cohorts into a standalone directory.
- Add common prepare, generate, score, summarize and validate commands; remove private absolute paths.
- Preserve raw attempt counts, official checkpoint hashes, upstream revisions, patch provenance and per-case seed policies.
- Store native molecules losslessly before SDF export; require all frozen geometry checks for success.
- Bind Pfizer diffusion's copied sampler to its declared non-anchor edge mask. Historical generation results are not rewritten.
- Exclude model weights, molecular structure downloads, historical outputs, OpenFE and experimental joint-guidance code.
