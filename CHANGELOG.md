# github_release_v1

- Extract the 13 benchmark adapters and fixed 45-case/30-case cohorts into a standalone directory.
- Add common prepare, generate, score, summarize and validate commands; remove private absolute paths.
- Preserve raw attempt counts, official checkpoint hashes, upstream revisions, patch provenance and per-case seed policies.
- Store native molecules losslessly before SDF export; require all frozen geometry checks for success.
- Bind Pfizer diffusion's copied sampler to its declared non-anchor edge mask. Historical generation results are not rewritten.
- Exclude model weights, molecular structure downloads, historical outputs, OpenFE and experimental joint-guidance code.
