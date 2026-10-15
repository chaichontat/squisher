# Task notebook

Keep task logs here; README.md is project documentation, not a log.

User-facing name: `fit-residual`. Historical code, test, and artifact-path
identifiers retain their original names for provenance.

## 2026-09-08 — Fit-residual simplification

- Goal: simplify the recent fit-residual implementation while preserving fitting,
  validation, output artifacts, and atomic publication.
- Starting state: extensive uncommitted work, including new `post_basic.py`,
  `post_basic_qc.py`, and their tests. Changes are scoped to this flow.
- Result: consolidated preview and other-plane correction/QC into one loop;
  artifact paths are rooted when written, without downstream rewriting.
  Replaced overlap-edge special cases with a one-pixel neighborhood and reused
  dataclass replacement when correcting sampled tiles. Fit mathematics, input
  checks, held-out validation, and atomic publication are preserved.
- Checks: baseline workflow/QC tests: 25 passed. Final command:
  `/home/chaichontat/miniforge3/envs/multi/bin/python -m pytest lightsheet/tests/test_post_basic.py lightsheet/tests/test_post_basic_qc.py lightsheet/tests/test_cli.py deconv/tests/test_basic_profiles.py deconv/tests/test_residual_field.py deconv/tests/test_tile_gains.py -q`
  — 80 passed. Added overlap-edge coverage and per-plane artifact hash/path checks.
  A diagnostic comparison against `/tmp/post_basic_before.py` found exact seam
  sample equality across 100 randomized four-tile cases (NumPy seed 42).
- Ruff lint passes for the implementation and tests; implementation formatted.
  Existing test formatting was left alone. Repository-wide `git diff --check`
  reports an unrelated trailing blank line in
  `lightsheet/tests/test_registration_workflow.py:619`; that file was not edited.
- No production dataset was rerun; this pass validates implementation equivalence,
  not acceptance of a new fitted correction. No commits were created.
- Moved this entry out of README.md following the user's correction.
## 2026-09-10 - Fit-residual source fields

- Goal: make `lightsheet fit-residual` fit and apply both the shared camera field and regularized source-specific spatial fields, in addition to scalar source gains.
- Inputs: `lightsheet/src/squisher_lightsheet/post_basic.py`, `residual_correction.py`, CLI, fusion loader, and focused tests; validation uses the `multi` environment.
- Starting evidence: the v2 correction schema stores one shared coefficient matrix and one scalar gain per source. The 514 diagnostic found persistent broad within-grid contrast after exact boundary feathering.
- Plan: add a held-out-selected source-field term at the fitting owner, serialize it per source, apply the combined field lazily during fusion reads, expose its penalty in both CLI aliases, and add synthetic recovery/schema/application tests.

- TDD: the new CLI forwarding and synthetic source-field recovery tests initially failed because `--source-field-penalty` and `fit_source_fields` did not exist.
- Implementation: sparse per-source cosine-field design; pair-disjoint penalty selection; all-pair deployment refit; correction/manifest schema v3; per-source coefficients combined lazily with the shared field during fusion reads; conservative global scaling includes both fields; QC emits `source_fields.csv`.
- Validation so far: `72 passed` across `test_residual_correction.py`, `test_post_basic.py`, `test_post_basic_qc.py`, and `test_cli.py`; focused Ruff passed before the requested simplifier/reviewer passes.
- Sol simplification: removed one unused source-coefficient allocation and renamed
  sparse row/column bookkeeping; no model or schema behavior changed.
- Astra review found and the implementation now fixes three boundary contracts:
  schema v3 rejects unequal shared/source coefficient shapes, held-out Z
  validation selects penalties from pooled training planes before all-pair
  comparator refits, and `source_fields.csv` covers every deployment source.
  Astra's follow-up found no remaining defects and checked held-out-plane
  isolation, unsupported-source identity behavior in validation, and strict
  support enforcement for deployment.
- Added coverage for sparse four-pair single planes, mismatched coefficient
  shapes, complete source-field QC, depth-varying source fields, and source-order
  independent loading.
- Final focused validation: `74 passed` across `test_residual_correction.py`,
  `test_post_basic.py`, `test_post_basic_qc.py`, and `test_cli.py`; focused Ruff
  and `git diff --check` pass. The host-GPU full lightsheet run reached 571
  passes with two unrelated dependency-API failures in method-10 rotation test
  construction and a private multiview-stitcher fusion call.
- The completed 514 v2 output remains preserved. A separate 514 v3 data run has
  not been started while the active 488 registrations are using the accelerators.

## 2026-09-12 - Package ventricular UV surface construction

- Goal: add a `lightsheet` CLI subcommand that reproduces the accepted prod2
  ventricular mesh and geometric UV-chart construction without dataset-specific
  paths.
- Inputs: a 3-D binary identity-direction NRRD mask, physical signed-distance
  smoothing in micrometres, and a new output directory.
- Contract: extract the outermost +X zero crossing on each nonempty ZY ray,
  require one oriented manifold disk, then create a length-spaced circular-boundary
  uniform-weight Tutte embedding. Cell-table projection remains downstream and is
  not part of this surface-building command.
- Planned validation: focused unit tests for surface extraction and UV topology,
  a Typer end-to-end test over a synthetic NRRD, CLI help, Ruff, and diff checks.
- Implementation: added `lightsheet uv-surface --mask PATH --output-dir DIR
  [--smoothing-um 19.2]`. It atomically publishes `surface.npz`, `surface.ply`,
  `chart.npz`, and `metadata.json`; added direct `libigl` and `SimpleITK`
  dependencies and updated `uv.lock`. Dataset-specific cell projection remains
  outside this command.
- Validation: focused UV/CLI tests passed (`39 passed`); focused Ruff and diff
  checks passed; installed command help exposes the documented required options.
  A packaged CLI run against the accepted prod2 mask produced byte-identical
  `surface.npz` (`24cffc8e...`) and `chart.npz` (`acd2f36a...`) artifacts, with
  46,416 vertices, 91,737 faces, one component, no nonmanifold edges, and Euler
  characteristic 1. The comparison output is temporary at
  `/tmp/squisher-uv-prod2-cGfXCF/surface`.
- Code-simplifier review: two read-only subagents independently audited the new
  numerical owner and its CLI/test/dependency integration. Both found the module
  boundaries and validation/publication helpers justified. Applied only local
  cleanups: inline the one-use transition array, compute the repeated per-cell
  corner count once, clarify the component-count name, and remove two one-use
  aliases. Rejected a proposed pair of one-use metadata count variables because
  they would only relocate expressions.
- Post-simplification validation: the same focused suite passed (`39 passed`),
  Ruff and diff checks passed, and a fresh packaged prod2 run at
  `/tmp/squisher-uv-simplified-kCua1f/surface` again produced byte-identical
  `surface.npz` and `chart.npz` hashes.

## 2026-09-12 - Bilateral ventricular UV surfaces for 230Tnc

- Extended the same owner with `lightsheet bilateral-uv-surface --mask PATH
  --output-dir DIR --split-x-index INDEX [--smoothing-um 19.2]`. It extracts
  the low-X +X and high-X -X medial faces separately, excludes rays crossing
  the artificial split cap, retains the largest face component per hemisphere,
  and atomically publishes two independently validated manifold-disk meshes and
  charts plus `bilateral.json`.
- Astra visual review selected array-X split index 226 for
  `/working/230Tnc/cortex_refit_42um_mask.nrrd`: low-X/right is X<226 and
  high-X/left is X>=226, with a physical split plane at 2185.244973 µm.
- Focused validation after the component policy passed (`42 passed`) and Ruff
  passed. Production exited 0 at `/working/230Tnc/vz-surface`: low-X/right has
  54,289 vertices and 106,990 faces; high-X/left has 55,353 vertices and
  108,952 faces. Both outputs independently passed manifest-hash, finite-array,
  topology, split containment, and face-normal checks.
- A post-change direct prod2 regression reproduced the accepted byte hashes:
  `surface.npz` `24cffc8e...` and `chart.npz` `acd2f36a...`. A concurrent,
  unrelated `cli.py` import of the absent `planar_tilt` module prevented a fresh
  packaged CLI startup after production, so this last regression invoked the
  same `build_ventricular_uv_surface` owner directly.

## 2026-09-12 - Trim 230Tnc surfaces to the main sheets

- Added optional `--keep-main-sheet` to the bilateral owner. It keeps rays before
  the cut bridge, follows the bridge's per-Y upper Z envelope, and removes the
  posterior wrap. Added `--main-sheet-min-z-index` as an explicit, provenance-
  recorded lower cutoff; it is rejected unless main-sheet trimming is enabled.
- Astra reviewed the envelope-only render, identified residual bridge tabs below
  approximately Z=50, and recommended the final `--main-sheet-min-z-index 50`.
  Production exited 0 at `/working/230Tnc/vz-surface-main-sheet-z50`; earlier
  outputs were preserved.
- Final meshes: low-X/right 50,382 vertices and 99,818 faces; high-X/left 51,233
  vertices and 101,519 faces. Both are one manifold disk with no nonmanifold
  edges and Euler characteristic 1. Independent checks passed for artifact
  hashes, finite arrays and UVs, topology, split containment, and opposite medial
  normal orientations. Focused CLI/UV suite: 49 passed; Ruff passed.
- Astra accepted
  `/working/230Tnc/vz-surface-main-sheet-z50-render/bilateral_surfaces.png` as a
  main-sheet-focused result: large basal loops and lower tabs are absent and the
  broad sheets remain intact. A small orange medial hook remains in the oblique
  view, so the output is documented as a cropped main-sheet extraction rather
  than a completely curl-free or anatomically validated ventricular surface.

## 2026-09-12 - Orthogonal surface intensity maps

- Added packaged `lightsheet surface-intensity-map`. It validates a complete
  aligned ZYX micrometre OME-Zarr, loads one bounded pyramid level, samples raw
  intensity along each surface's negative oriented vertex normal, clips each ray
  to its first contiguous mask interval, and max-reduces onto the vertex. It
  atomically writes per-vertex values, support, last valid depth, 512×512
  triangle-interpolated UV rasters, a shared-scale PNG, and hashed provenance.
- Focused tests cover first-interval clipping, normal orientation, projection
  direction, synthetic OME-Zarr/mesh output, and CLI forwarding. Final focused
  CLI/surface suite passed (`58 passed`); Ruff passed.
- Real 230Tnc run used the selected 561-crop Zstd viewing store at level 2
  (178×561×862 uint16, 172,155,192 decoded bytes), 4.8 µm depth steps, and a
  2496 µm cap. A 2,048-ray pilot took 1.98 s with 679,032 kB peak RSS and no
  cap-reaching rays. Production exited 0 at
  `/working/230Tnc/vz-surface-main-sheet-z50-intensity-561` with finite values
  for all 101,615 vertices and no cap-reaching rays; hashes and numeric contracts
  passed independent readback.
- Astra accepted `intensity-map.png` as a first diagnostic. Coherent bands may be
  signal, while fan-like streaks and triangular sectors indicate geometric UV
  distortion. The hemispheres have independent UV orientation, and different
  first-interval lengths can affect maxima, so screen-position and raw brightness
  comparisons are not anatomical evidence without source-volume QC.

## 2026-09-12 - Replace the global Z cutoff with a discontinuity filter

- The apparent head truncation was present in the generated ventricular mesh,
  not marching-cubes closure or rendering: `--main-sheet-min-z-index 50`
  globally removed every retained ray below source Z=50.
- Added optional `bilateral-uv-surface --max-face-edge-um`. It removes faces
  spanning longer physical edges before the existing largest-component step and
  records the threshold and removed-face count in `bilateral.json`.
- For 230Tnc, the bridge-envelope-only surface with a 100 µm edge limit removed
  289 long faces on low-X and 251 on high-X, exposing 3 and 6 source components.
  Retaining the largest component produced one manifold disk per hemisphere:
  low-X 51,560 vertices/102,082 faces and high-X 52,217 vertices/103,415 faces.
  This restores 2,162 vertices relative to the Z=50 fit without a horizontal
  cutoff. Astra accepted the geometry as retaining the main sheet while removing
  the bridge projections visible in the Z=40/no-cutoff candidates.
- Production surface: `/working/230Tnc/vz-surface-main-sheet-edge100`.
  Reprojected 561 output:
  `/working/230Tnc/vz-surface-main-sheet-edge100-intensity-561`; selected camera
  render: `final-view-complete-head.png` (`elev=-15.1`, `azim=-91.3`,
  `roll=180`). Focused tests passed (`55 passed`); Ruff and format checks passed.

## 2026-09-12 - Publication-width native-space render

- Re-rendered the accepted edge-filtered surface at the selected camera with the
  complete cortical-mask marching-cubes context (124,624/124,624 faces), a pooled
  1st–99.5th percentile intensity scale, and a camera-matched signed XYZ key.
- The ORX figure audit passed at 6.75-inch width and 300 DPI. PNG, PDF, and SVG
  outputs share the stem
  `/working/230Tnc/vz-surface-main-sheet-edge100-intensity-561/ventricular-projection-native`.
- The filtering comparison was relabeled to distinguish Z cutoffs from the
  100 µm edge criterion and now reports displayed rather than loaded vertices.
  Its audited PNG/PDF/SVG outputs share the stem
  `/working/230Tnc/vz-surface-filtering-comparison`.

## 2026-09-13 - Direct-fusion source-read amplification

- Goal: replace the batch-scoped corrected-slab cache with a session-owned,
  bounded source reader so direct OME-Zarr fusion reuses source data without
  materializing a corrected source volume.
- Reproducer: 230Ptp 561 channel-0 final fusion, source level-0 Zarr v3 arrays
  shaped `(2, 1194, 1920, 1920)`, outer chunks `(1, 48, 960, 960)`, inner
  sharded chunks `(1, 1, 240, 240)`, JPEG-XR; output chunks
  `(24, 960, 960)`, one GPU, auto batch 252.
- Evidence: after about 81 minutes the fusion process had issued 1.72 TB of
  physical reads and 277 GB of writes, used roughly 1--2 CPU cores, and left
  the RTX 5090 near 2% utilization. The temporary corrected cache held 588
  uncompressed `.npy` slabs totaling 252.7 GiB (440 MiB average).
- Mechanism: one GPU produces one serial block group; the correction-cache
  context is recreated for every output batch; request-shaped keys duplicate
  overlapping intervals such as `z0:64` and `z32:96`; disk hits eagerly load
  whole corrected slabs before slicing the requested ROI.
- Intended change: one persistent reader per channel/device fusion session,
  canonical raw source-region identity, exact-ROI correction during fusion,
  strict accounting for retained and in-flight bytes, and no corrected disk
  cache in the default path. Output completion markers remain the sole resume
  authority.
- Planned validation: focused cache identity/budget/error tests followed by an
  identical-output benchmark over at least two adjacent production z batches,
  reporting wall time, physical IO, temporary writes, cache reuse, and RSS.
- Implemented the reader at the direct-fusion owner. One channel fusion now
  owns a raw-source LRU and per-key in-flight future table; concurrent duplicate
  loads share one result, pending allocations reserve bytes, and units larger
  than the configured budget use an exact ROI read. BaSiC/residual correction
  remains an exact-ROI GPU operation. No corrected source volume or corrected
  disk cache is created.
- Cache identity includes source tile/channel, array path, fixed selections,
  dropped dimensions, and canonical storage-grid index. For sharded Zarr v3
  sources, the unit is the outer shard (`array.shards`), not the inner codec
  chunk exposed by `array.chunks`; this avoids hundreds of xarray deserialize
  calls for one production output block.
- Threaded GPU execution now owns one serial queue per CUDA device and routes
  every block through the established direct worker, preserving view-candidate
  culling, crop-sharpness source indices, OOM cleanup/retry, and synchronous
  error propagation. The CLI default is `--batch-size auto
  --source-cache-max-gib 64`.
- Validation after Astra's two reviews: affected uv suite 124 passed with one
  unrelated installed-private-API mismatch test deselected; Ruff passed;
  production `multi` focused suite 8 passed. Tests cover corrected and
  uncorrected reuse, outer-shard granularity, oversized fallback, concurrent
  deduplication, direct-worker dispatch, and error propagation. The real-data
  adjacent-batch benchmark remains pending because the old implementation is
  still writing the production output.
- Two code-simplifier subagents reviewed non-overlapping ownership areas, and
  the primary agent inspected their edits. The pass removed an unused loaded
  byte field, impossible optional-budget branches, duplicated correction-offset
  arithmetic, a one-value logging parameter, and one redundant default-forwarding
  test invocation. Concurrency futures/reservations, incremental chunk assembly,
  direct-worker routing, and separate behavioral tests were retained because
  they enforce distinct contracts. Post-refactor validation remained 124 passed
  (one unrelated dependency-mismatch test deselected), Ruff passed, and the
  production `multi` focused suite remained 8 passed.
- The first production restart exposed two resume/provenance issues without
  losing completed output: deterministic wrapper rewrites changed only mtimes
  of content-identical position/registration JSON, and the failed zero-marker
  workspace was newer than the valid workspace. Resume comparison now ignores
  legacy mtimes only for identities protected by equal SHA-256 hashes, still
  treats raw-file mtimes as semantic, and selects the matching workspace with
  the most completion markers before using mtime. The run correctly reattached
  to 5,292 completed blocks. Crop-sharpness provenance uses the same semantic
  comparison of its persisted full identity.
- The first cache benchmark found an upper-edge request (`x=1403:1921` for a
  1920-pixel source) that the established reader truncates. A fail-fast cache
  check was incorrect. The final reader caches valid-start upper-edge requests
  using the same truncated z/y/x shapes and original origin, without padding;
  negative and empty/outside starts retain the established direct-reader path
  and are counted with requested/materialized bytes and elapsed time.
- Production measurements on one GPU, 252 blocks per z plane: old recent
  batches were 314--354 s. Initial cache version measured 263 s cold and 198 s
  warm. Final upper-edge-cached version measured 227 s cold and 160 s adjacent
  warm. At 3,000 requests it had 2,553 hits and 447 misses (85.1%), retained
  36.8 GiB with no eviction, and process physical reads increased by roughly
  11 GiB across the two batches. These are adjacent resumed production batches,
  not a randomized controlled benchmark. The workflow remains running in tmux
  session `codex-230ptp-cross-dataset-match`.
- A final two-subagent code-simplifier pass covered the post-benchmark cache and
  provenance changes. It removed only one-use locals; upper-edge truncation,
  bypass accounting, semantic identity recursion, and progress-first workspace
  selection were retained as distinct contracts. Primary-agent validation after
  integration: 130 affected tests passed with one unrelated private-API mismatch
  test deselected, Ruff passed, and 24 focused production-`multi` tests passed.

## 2026-09-13 - Pipeline direct fusion stages

- Goal: overlap raw source preparation, one serialized CUDA fusion stage, and
  ordered JPEG-XR/Zarr writes without materializing a corrected source volume.
- Production evidence: PID 639465 on 561 channel 0 used about 1.43 CPU cores;
  RTX 5090 stayed mostly at 2--6% with short 23--24% bursts; RTX 4090 was idle
  by the sequential-channel policy; process IO averaged about 12 MiB/s read and
  6 MiB/s write; RSS was about 69.6 GiB with a full 64 GiB raw-shard cache.
- Starting hypothesis: after fixing source read amplification, the combined
  per-block prepare/GPU/encode-write critical section prevents stage overlap.
- Safety: develop and validate from commit `fe4c726` in a temporary clone so
  the running 561 process and its imminent 638 subprocess cannot import a
  partially edited module. Preserve one CUDA consumer per device, ordered
  writes, synchronous batch completion, resume markers, OOM retry, and explicit
  staging-byte backpressure.
- Planned validation: deterministic concurrency/failure tests, focused fusion
  tests, Astra review, then a bounded real-data comparison when it cannot
  interfere with the production chain.
- Implemented a bounded three-stage direct-Zarr pipeline. Raw source regions
  are prepared on one CPU thread per active device, each device retains one
  serial CUDA consumer, and output assignment is globally serialized in
  original batch order. Fusion still calls multiview-stitcher's existing
  `_fuse_block_zarr_backed`; correction remains on the GPU and no corrected
  intermediate is materialized.
- A session-wide reservation manager admits blocks in batch order and reserves
  estimated raw input plus output bytes before preparation. It accounts pending
  writes until completion, permits only one explicitly logged oversized block
  while staging is otherwise empty, and cancels waiters on any stage failure.
  If all device queues cannot run concurrently, execution uses the established
  independent serial queue path to avoid an admission-order deadlock.
- Astra reviewed three iterations. Its findings led to strict global instead
  of per-device accounting, global ordered writes, zero-cache compatibility,
  loky normalization ownership, queued-failure cleanup, the insufficient-worker
  fallback, and removal of a stale output-array reference. Final verdict: no
  remaining concrete scheduling, accounting, or shard-write blocker; proceed
  to bounded production validation.
- Validation in both the isolated clone and integrated tree: 92 relevant tests
  passed with the known installed-private-API `trim_overlap` mismatch test
  deselected; Ruff passed without cache. Added coverage proves stage overlap,
  byte backpressure, cross-GPU write order, insufficient-worker fallback,
  queued prepare/compute/write failures, raw-region reuse, and reuse after an
  interrupted GPU attempt. The active workflow is already building the 561
  pyramid, so a competing real-data benchmark was not launched.

## 2026-09-14 - Parallel held-out Z validation

- Goal: parallelize the independent fit-residual leave-one-Z-out validation folds
  without changing serial results, raw-tile sampling, or atomic publication.
- Starting state: `_validate_z_models` evaluates every held-out plane serially;
  the `fit-residual` CLI exposes only the upstream sampling
  `--workers`. The active 638 correction run is left untouched.
- Contract: add a separate positive `validation_workers` setting, default 1;
  use explicit spawn processes initialized once with the already-created seam
  rows; submit only held-out Z values; preserve sorted output order; fail with
  held-out-Z context and publish no completed artifact on worker failure.
- Baseline: `uv run --package squisher-lightsheet pytest -q
  lightsheet/tests/test_post_basic.py::test_z_validation_accepts_sparse_per_plane_pair_support
  lightsheet/tests/test_cli.py::test_residual_cli_aliases_forward_one_contract`
  passed (3 tests, 0.97 s).
- Planned validation: focused serial/parallel equivalence, deterministic order,
  worker-count validation/capping, spawn execution and error propagation tests;
  Ruff; bounded 1/2/4-process timing and peak-memory benchmark.
- Implemented: `_validate_z_models` keeps its serial default and delegates each
  held-out plane to a top-level fold function. For `validation_workers > 1`, an
  explicit spawn `ProcessPoolExecutor` initializes each child once with the
  existing seam rows and immutable fit configuration; submitted tasks contain
  only the held-out Z. Results are stored by sorted plane index, excess workers
  are capped to plane count, and child failures are re-raised with Z context.
  Each child applies `threadpoolctl.threadpool_limits(1)` when available.
- CLI/API: added positive `--validation-workers` (default 1) to
  `fit-residual`, `fit_residual_correction`, and `run_post_basic`;
  manifests record the requested count. The existing upstream `--workers`
  retains its separate sampling role.
- Validation: 99 tests passed across `test_post_basic.py`,
  `test_residual_correction.py`, and `test_cli.py` (28.78 s). Focused spawned
  tests prove exact serial/parallel dictionary equality, sorted output, counts
  1/2/8 for three folds, positive-count rejection, sparse support, and
  contextual worker failure. Ruff lint passed on all changed files. Ruff format
  passed on the four changed fit-residual/test files but the already-dirty
  `cli.py` remains globally unformatted; formatting a temporary copy showed
  repo-wide pre-existing changes, so it was not rewritten.
- Benchmark: bounded synthetic workload of 10 Z planes, 20 sources, 40 overlap
  pairs/plane, and 250 samples/pair. Serial: 11.277 s, 363.7 MiB aggregate peak
  RSS. Two spawn processes: 5.333 s, 1028.4 MiB (2.11x speedup). Four: 4.290 s,
  1702.4 MiB (2.63x). Initial tiny 8-plane timing was discarded because startup
  dominated, and the first 2/4 measurements were mistakenly run concurrently;
  all reported final measurements were rerun sequentially with `/proc` process-
  tree RSS sampling. Production scaling remains unmeasured; use two workers
  first when seam-row duplication makes memory uncertain.
- Safety: no production command was run and the active 638 fit-residual process
  was neither inspected nor disturbed.
- Code-simplifier pass: removed the unused module-global reference to the
  `threadpoolctl` limiter. Constructing the limiter applies the process-wide
  native-thread limit; no context exit restores it inside the worker. Spawn
  configuration, deterministic aggregation, and failure behavior are unchanged.
  Post-simplification validation passed the same 99 focused tests in 27.99 s;
  focused Ruff lint, format checks, and `git diff --check` passed.

## 2026-09-14 - 230Ptp 638 TL/TR residual match

- Goal: remove the remaining TL/TR intensity discontinuity before rebuilding
  the deleted 638 fusion.
- Inputs: completed channel-1 correction at
  `/working/230Ptp/561638-run-20260912/post-fusion/joined-final-post-basic-rerun/ch1/correction.json`,
  final-affine 638 registration, and the preliminary 561 fixed grid.
- Starting evidence: median post-correction boundary discrepancy is 0.087 log2
  within TL, 0.217 within TR, and 0.440 across TL/TR; the cross-view p90 is
  0.637. The joint correction therefore leaves a distinct acquisition-level
  offset.
- Plan: derive exact L/R source subsets from the joint correction without
  changing its values, run packaged `lightsheet cross-dataset-match` over the
  standard 39 Z percentiles with L fixed and R moving, and inspect the generated
  before/after QC before fusion.
- Actions: split the joint correction into exact 26-source L and 23-source R
  payloads under `joined-final-post-basic-rerun-split/ch1`, preserving every
  coefficient, gain, and common scale. Ran the packaged matcher with L fixed,
  R moving, channel 1, source level 2, stride 1, eight sampling workers, and all
  39 standard Z percentiles. Output is
  `joined-final-cross-dataset-match-rerun/ch1`; the tmux run exited 0 with the
  completion marker and 2,007 pair/depth measurements.
- Result: the fitted R factor is 0.9901593886. Median absolute cross-view
  discrepancy changed from 0.15026 to 0.15157 log2 (slightly worse), while p95
  changed from 0.60421 to 0.59712 (minor improvement). Pair medians disagree in
  sign and reach about +/-0.5 log2; early-Z outliers exceed +2 log2. Therefore
  one global view factor does not own the visible discontinuity, and this output
  should not be used for final fusion without further modeling/review.
- Next: choose between a cross-view-weighted per-source adjustment constrained
  to preserve within-view seams, or registration/overlap QC for the extreme
  early-Z and source-pair outliers. Final 638 fusion has not been started.

## 2026-09-14 - 230Ptp 638 registered overlap correction field

- Replaced the rejected global TL/TR scalar with patch-level measurements in
  registered physical ZYX. The 39-depth gate produced 11,849 measurements in
  about 95 seconds. Among bins with at least three observations and two source
  pairs, median sign agreement was 90%; 67% reached at least 75% agreement and
  9.2% had a strong local conflict.
- Architecture: keep schema-v3 raw-tile residual corrections unchanged. Fit a
  separate coarse registered-coordinate log-ratio field, apply the identical
  factor to every R source at a fused output location after resampling and
  ownership selection, and apply a common scale to all contributions for
  clipping safety. Unsupported regions and low-confidence/conflicting bins
  fade smoothly to the robust mean correction.
- Validated field:
  `/working/230Ptp/561638-run-20260912/post-fusion/joined-final-registered-view-correction-cv/ch1/registered-view-correction.json`.
  Its 5-fold held-out spatial blocks improved median absolute mismatch from
  0.16735 to 0.12380 log2 and p95 from 0.56805 to 0.415996. The QC is a single
  five-depth contact sheet beside the artifact.
- Implementation: added the registered field loader/evaluator and fitter,
  `lightsheet registered-view-correction-fit`, patch-level cross-dataset
  measurements, `lightsheet fuse --registered-view-correction`, absolute
  expanded-block origin/spacing plumbing, source-view subsetting after direct
  candidate culling, and resume/provenance hashing. Registered corrections are
  restricted to 3D direct `crop-sharpness-seam` fusion and disable planewise
  dispatch so the coordinate contract remains explicit.
- Validation: 58 focused crop, cross-dataset, CLI, and fusion-wrapper tests
  passed; Ruff passed all changed files. A broader fusion test run had 73 passes
  and one unrelated installed-dependency mismatch where a pre-existing test
  passes removed `trim_overlap` to multiview-stitcher.
- Production: started `ptp638_registered_field_fusion` on isolated GPU 1 using
  the final-affine 561 native grid, the completed joint fit-residual correction,
  10 um crop-sharpness feathering, and the validated registered field. Output:
  `/working/230Ptp/561638-run-20260912/post-fusion/final-fusion-affine/638/fused.ch1.ome.zarr`.
## 2026-09-15 - Budgeted fusion GPU sub-blocking

- Goal: bound peak CUDA memory for dense direct OME-Zarr fusion blocks without changing the output grid, chunks, dtype, codec, crop-sharpness ownership algorithm, or registered-view correction contract.
- Trigger: prod2 channel 405 has up to 32 exact source windows for a `24x960x960` output block; the installed multiview-stitcher materializes all transformed views and blending weights before the fusion callback, so callback-only streaming cannot prevent the observed CUDA OOM.
- Planned change: add an opt-in GPU-memory budget that recursively partitions only the internal x/y compute region, repeats exact view intersection per halo-expanded sub-block, runs the existing fuser per sub-block, and assembles the original output chunk in host memory.
- Validation: focused planning/assembly/CLI tests first, then a bounded dense real block with synchronized CUDA wall time and peak memory plus output/seam comparison before launching prod2-405.
- Implemented an opt-in `--fusion-gpu-memory-max-gib` path through the public
  CLI/API and direct CUDA executor. The planner recursively partitions x/y,
  keeps the output OME-Zarr region and chunk unchanged, derives registered
  correction coordinates per expanded child, and fails if no x/y child can
  satisfy the estimate. The setting remains outside pixel-affecting resume
  identity.
- A first real prod2-405 comparison exposed that re-culling each child changed
  crop ownership: 9 GiB versus 18 GiB changed 2,779,049/22,118,400 voxels with
  maximum error 3440. Preserving the parent candidate and raw-source context is
  therefore an algorithmic invariant, not an optional optimization; the planner
  now carries that context into every child and accounts for its full input.
- Corrected dense-block benchmark: block `(78, 9, 5)`, 32 candidates, output
  core `24x960x960`, halo `9x18x18`. At 18 GiB the plan used two
  `24x480x960` children, estimated 17,252,867,148 bytes each, retained
  15,906,897,920 CUDA bytes, and ran in 2.665 s. At 9 GiB it used four
  `24x480x480` children, estimated 9,055,649,868 bytes each, retained
  8,407,482,368 CUDA bytes, and ran in 3.975 s. The two results differed at
  1,461 voxels (0.0066%), all by one uint16 count; only 3/92,064 inspected seam
  voxels differed. This is affine-resampling roundoff, with no ownership-scale
  error.
- Verification: 59 focused sub-block/API/GPU-batch/CLI tests and all five
  crop-sharpness tests pass. The broader fusion/GPU-batch/CLI run has 123 passes
  and one pre-existing failure where a test sends removed installed-MVS keyword
  `trim_overlap`; Ruff lint and `git diff --check` pass for the changed code.
- Production handoff: prod2's 405 launcher now requests an 18 GiB fusion GPU
  budget; the 561 command remains unchanged. Launch remains gated on an
  uncontended physical 5090.
- Production launch: after the 5090 became idle, started prod2-405 in tmux
  `codex-eduseg-prod2-fusion-405-v4`. The dry-run and live command both carry
  the 18 GiB budget. CUDA PID 2873324 was verified on physical UUID
  `GPU-5d7f9022-ea7f-32f4-71b5-e424ae96aafd`; the temporary scale-0 array is
  OME-Zarr v3 with JPEG-XR level 0.7. Initial 22- and 31-block pipeline batches
  completed without a new OOM or traceback, reaching 212 output files / 904.9
  MiB at the first 60-second progress report. The production render remains a
  background job and is not yet claimed complete.
- Code-simplifier pass: removed the benchmark's unused candidate-map return and
  dynamic `ExitStack`, inlined one-use forwarding values in the sub-block plan
  and execution path, and renamed the memory-forwarding test to match its full
  contract. Retained the planner, estimator, kwargs builder, explicit source
  context, validation locals, and cleanup ordering because they own measured
  memory, voting, error, or resource-lifecycle invariants. Post-refactor focused
  validation passed 65 tests; Ruff, benchmark formatting, shell syntax, CLI
  discovery, and `git diff --check` also passed.
- Production completion: prod2-405 exited 0 at 2026-09-15 13:50:47 EDT after
  completing the resumed scale-0 fusion and seven-level pyramid. The final
  `fused.ch0.ome.zarr` is marked `squisher_complete=true`; its fusion provenance
  is complete with 979 accepted sources, 9 rejected registrations, and zero
  unresolved sources. All levels are uint16 sharded JPEG-XR 0.7 arrays, and the
  successful invocation contains no OOM, traceback, exception, error, or failed
  marker.
- The final four-channel completion manifest is `complete`. Channels 405, 561,
  638, and 514 have identical OME multiscale dataset metadata (canonical SHA-256
  `6fdda7ba3bb6761b2cf154bac30ec6a135ba6683c79f3d4b633215d1ab82a8d5`),
  sharing the expected ZYX shape, transforms, and seven dataset paths. The
  manifest's 561/638/514 correction hashes match the files on disk.

## 2026-09-15 16:31 EDT - Canonical fused-fixed recovery and QC CLI

- Goal: move the reusable prod3 recovery and PNG QC behavior into the
  `squisher-lightsheet` package without changing recovery decisions or
  numerical transforms.
- Implementation: `fused_fixed_recovery.py` now owns the former 1,600-line
  recovery script; the old script path is a compatibility entry point.
  `complete_recovery.py` owns the fixed four-stage Method 6 policy, private
  spatial/linear promotion stages, content identity, strict reuse validation,
  and atomic publication. `fused_fixed_qc.py` owns pre-fusion hole and
  interpolated-window QC, while `qc.py` owns fused middle-Z QC.
- CLI: added `lightsheet cross-register recover-native`,
  `complete-recovery`, `prefusion-qc`, `interpolated-window-qc`, and root
  `fused-middle-z-qc`. Current prod3 v4, 638, and 561 launchers call the
  packaged QC commands. The approved historical v4 consumer validates its
  stored summary/window hashes without recomputing the new producer identity.
- Real-data validation input:
  `/working/eduseg/prod3/235-2-561638514-3/xreg-514-m6-v3-recovery/fused_fixed_method8_summary.json`
  (1,173 Method 6 windows). The packaged `complete-recovery` result under the
  production `multi` environment matched all 1,173 accepted v4 window JSON
  records exactly. Temporary replay output:
  `/tmp/squisher-complete-parity-514-multi`.
- Environment finding: the repository uv environment (NumPy 2.4.6, SciPy
  1.17.1) produced the same membership and decisions but 136 rows differed by
  about 1e-13 from the accepted artifact. The production `multi` environment
  (NumPy 2.5.2, SciPy 1.18.1, single-threaded BLAS) produced zero row
  differences. Exact artifact replay therefore requires the production
  numerical environment.
- Verification: 70 focused recovery, publication, CLI, geometry, and PNG tests
  pass; Ruff passes all touched Python files; four updated prod3 launchers pass
  `bash -n`; both new CLI command groups were inspected through the installed
  `multi` entry point with the checkout on `PYTHONPATH`.

## 2026-09-16 21:07 EDT - One-path EDU workflow runner

- Goal: turn `WORKFLOW_EDU.md` into a resumable command that starts from one
  representative OME-TIFF and uses the packaged CLIs for every supported stage.
- Entry points: `lightsheet edu-workflow ONE.ome.tif` and the environment-setting
  wrapper `/home/chaichontat/squisher/run-edu-workflow.sh ONE.ome.tif`.
- Discovery: the runner reads OME metadata, requires one three-channel
  561/638/514 acquisition and one unambiguous sibling 405 acquisition with the
  same normalized dataset key, rejects partial transfers, and writes the exact
  source resolution to `workflow.json`.
- Workflow: dataset-specific BaSiC, scaling, deconvolution, native review,
  native registration, per-channel fit-residual, coarse-Preibisch 561/638
  fusion, reference-initialized 638 refit, mandatory 514/405 cross-registration
  and recovery, identity-bound pre-fusion QC, level-0 materialization/fusion,
  final alignment QC, Cellpose subset/postprocessing, and paired exports.
- Human gates: `native-thresholds.json`, `fused-561-threshold.json`,
  `prefusion-approval.json`, and `segmentation-approval.json`. The CLI exits 3
  at a pending gate; rerunning the same command reuses validated stage markers.
- Cleanup: only exact 514/405 materialized-window directories are removed, and
  only after the corresponding fusion carries `squisher_complete=true`.
  Completed fusions prevent rematerialization on later gate-driven reruns.
- Validation: 55 focused EDU and existing lightsheet CLI tests pass in `multi`;
  Ruff, `bash -n`, `git diff --check`, CLI help discovery, and direct syntax
  compilation pass. No production dataset was launched by this implementation
  task; first real-data execution remains unverified.

## 2026-09-16 21:15 EDT - EDU runner simplification

- Goal: simplify the newly added EDU custom runner without changing its stage
  order, gates, resume identities, validation, cleanup, or CLI behavior.
- Changes: removed two unused metadata fields and the unused review-exception
  path attribute, inlined the one-call output-root and workflow-manifest setup,
  removed an identity path comparison, and flattened the completed-fusion
  branch. The workflow module remains separate because it owns the resumable
  stage lifecycle and identity-bound review gates; the shell wrapper continues
  to own `multi` environment activation.
- Validation: 55 focused EDU and existing CLI tests pass in `multi`; Ruff,
  `bash -n`, direct syntax compilation, `git diff --check`, and live CLI help
  discovery pass. The code-simplifier skill's requested
  `references/detailed-audit-guide.md` was absent from the installed skill, so
  the audit used the complete base skill. Real-data execution remains
  unverified.
- Follow-up: removed eager checks for commands and assets needed only after
  later review gates, stopped re-hashing runtime artifacts on resume, and
  reduced Cellpose export verification to the required paired-file counts.
  Completion checks remain at deconvolution/fusion reuse and before intermediate
  deletion. The narrow validation was six EDU runner tests plus Ruff; both pass.

## 2026-09-17 05:52 EDT - Resume prod3 coarse-Preibisch chain

- The first run completed the prod3 561 fusion but stopped because its launcher
  compared Zarr's logical chunk shape `(1,960,960)` with the storage shard shape
  `(24,960,960)`. Removed that duplicate assertion; the existing first-array
  metadata check already verifies the requested storage chunking.
- Resumed `/working/eduseg/prod3/run-prod3-preibisch-refusion-v1.sh` in tmux
  session `codex-eduseg-prod3-preibisch-v3`. It reused the completed 561 store
  and started the 2,400-window reference-initialized 638 Method 6 refit on GPU
  UUID `GPU-94714ab6-ba56-a7d9-6dc3-adc866b00bd6`.
- Persistent log: `/working/eduseg/prod3/run-logs/preibisch-refusion-v3.log`.
- Correction: stopped the Method 6 refit at tile 20 after the user specified
  that prod3 must use the reference transform directly. The interrupted output
  remains isolated at `xreg-638-global-preibisch-v1` and is not consumed.
- Updated the launcher to call packaged `lightsheet apply-channel-affine` with
  the hash-pinned prod2 calibration, composing that physical affine onto all 80
  prod3 561 registration tiles. The v4 run validated the transferred
  registration and started the 638 coarse-Preibisch fusion in tmux
  `codex-eduseg-prod3-preibisch-v4`; log and run record are
  `run-logs/preibisch-refusion-v4.log` and `preibisch-refusion-v4.json`.

## 2026-09-17 15:12 EDT - Canonical OME-Zarr to NRRD export

- Goal: replace the proven one-off prod3 level-3 561 export with a packaged
  `lightsheet export-nrrd` command.
- Representative input:
  `/working/eduseg/prod3/235-2-561638514-3/fusion-561-preibisch-v1/fused.ch0.ome.zarr`,
  level 3, uint16 ZYX shape `(285, 1334, 987)`.
- Contract: preserve decoded voxels and the selected NGFF scale/translation;
  write acquisition XYZ as identity-direction LPS XYZ in millimetres; publish
  neither the NRRD nor its JSON record until streamed readback validation passes.
- Planned validation: focused synthetic OME-Zarr unit/CLI tests, Ruff, CLI help,
  and parity against the completed prod3 export.
- Implementation: added `squisher_lightsheet.nrrd_export` and
  `lightsheet export-nrrd SOURCE OUTPUT --level N --channel LABEL`. The writer
  accepts 3-D uint16 micrometre ZYX levels, streams configurable Z slabs into
  gzip, embeds acquisition XYZ as LPS XYZ millimetres, and refuses existing
  final or temporary paths. It publishes the final path only after SimpleITK
  reads the spatial header and a streamed gzip readback matches the source byte
  count and SHA-256.
- Failure found by the focused test: the first temporary filename ended in
  `.tmp`, so SimpleITK could not select its NRRD reader. The temporary artifact
  now remains hidden but ends in `.nrrd`.
- Verification: all 53 exporter and lightsheet CLI tests passed; Ruff and
  `git diff --check` passed; packaged command help shows the intended options.
  A real level-6 prod3 export passed at
  `/tmp/squisher-nrrd-real-level6-561-20260917.nrrd`.
- Real parity: the packaged level-3 run wrote
  `/tmp/squisher-nrrd-real-level3-561-20260917.nrrd` in 27.64 s. Its shape,
  dtype, scale, origin, and decoded voxel SHA-256
  `40afb5d58d065ff2382cf04c53f3cda3d5989e042bfc9222d4f707ad470c64e0`
  exactly match the earlier prod3 561 export. SimpleITK compatibility passed;
  interactive 3D Slicer loading remains untested.

## 2026-09-17 15:26 EDT - NRRD exporter simplification

- Scope: only the new NRRD exporter module, its CLI wiring, and its focused
  tests. The skill's `references/detailed-audit-guide.md` was absent, so the
  audit used the complete base skill plus repository instructions.
- Changes: removed forwarding aliases for the two public defaults, inlined the
  one-use dataset index, used Zarr's logical `nbytes`, flattened fixed NRRD
  header construction, removed redundant dimensional/axis checks, and made the
  test gzip reader lifecycle explicit.
- Retained boundaries: `nrrd_export.py` remains separate because it owns the
  streaming serialization and atomic validation lifecycle. Input, NGFF-unit,
  metadata-value, and readback helpers remain because they enforce distinct
  untrusted-data or resource contracts.
- Verification: all 53 exporter and lightsheet CLI tests passed after the main
  refactor; the final four exporter tests, Ruff, and `git diff --check` passed
  after the redundant-check cleanup. The CLI, voxel values, physical metadata,
  progress order, refusal behavior, and output schema are unchanged.

## 2026-09-19 - Reduce prod3 postprocessing GPU memory

- Goal: preserve the prod3 postprocessing configuration and label-ranking
  semantics while eliminating the CuPy allocation that caused both full-brain
  retries to fail.
- Input: `prod3/cellpose-full-preibisch-both-v1/output_segmentation-sam.zarr`;
  postprocessing uses sigma `(1,2,2)`, `V_min=500`, margin `30`, and one worker
  per GPU.
- Failure: both retries failed before Gaussian smoothing in
  `cp.unique(masks_gpu, return_inverse=True)`. For a `(400,832,832)` block, the
  failing 2,215,116,800-byte allocation is exactly one int64 per voxel.
- Change: compute sorted unique values and dense ranks with
  `cp.searchsorted`, releasing the source and label table before casting ranks
  to int32. The sorted-rank result intentionally remains identical to CuPy's
  prior unique-inverse behavior, including crops without background.
- Planned validation: focused CPU-backed equivalence cases for background,
  absent background, sparse labels, and uint32 IDs above `2**31`; then the
  distributed postprocessing test module. GPU peak memory remains to be
  measured on the production-sized block.
- Result: all 32 distributed postprocessing tests pass and Ruff is clean. A
  real CuPy parity check passed on sparse uint32 labels, including values above
  `2**31`. The unchanged prod3 launcher was restarted in tmux session
  `codex-eduseg-prod3-cellpose-full`; it reused completed segmentation and
  started all 1,620 postprocessing blocks at 2026-09-19 14:39 EDT. Both workers
  successfully passed dense relabeling and Gaussian smoothing on full
  `(400,832,832)` crops. Persistent log:
  `/working/eduseg/prod3/run-logs/cellpose-full-preibisch-both-v1.log`.
- Follow-up failure: later retries passed dense relabeling but failed inside
  Gaussian smoothing when transient external GPU use prevented another
  1.1-GiB float32 allocation. Goal: route the reconstructible Gaussian phase
  through the existing bounded CUDA-OOM retry contract, preserving sigma,
  label voting, block geometry, and output semantics. Planned validation is a
  forced OOM-then-success test plus the distributed postprocessing suite.
- Implementation: moved the existing CUDA lease/OOM retry owner into the
  shared `squisher.gpu_retry` module, updated every lightsheet consumer to the
  shared owner, removed the obsolete lightsheet compatibility module, and
  wrapped only per-block Gaussian smoothing. An OOM now clears
  CuPy device and pinned pools, waits with bounded exponential backoff, and
  reconstructs the Gaussian transaction from the unchanged host label array.
- Verification: the forced OOM test observed one cleanup and succeeded on its
  second smoothing attempt; all 33 distributed postprocessing tests and all 4
  shared retry tests pass. Ruff and `git diff --check` pass for the changed
  files.
- Simplification: removed the one-use Gaussian retry forwarding helper and
  inlined the shared retry call at the phase boundary. The successful return is
  intentionally ignored because smoothing's `in_place=True` contract updates
  the captured host array only after GPU work and host transfer complete. The
  retry regression now exercises the full block processor. The 33 postprocess
  tests, focused downstream lightsheet tests, Ruff, and diff checks pass.

## 2026-09-19 - Investigate stitch_20x fusion VRAM use

- Goal: identify the owning allocations and scheduling behavior responsible for
  high VRAM use in
  `lightsheet/src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py`, then
  obtain an independent Astra recommendation for reducing it.
- Inputs: the current dirty working-tree implementation and its focused fusion
  tests; existing user changes will be inspected read-only and preserved.
- Planned validation: trace the fusion task graph, derive per-block allocation
  sizes from the code and defaults, inspect any existing profiling/log evidence,
  and compare those findings with Astra's proposed remedies. No production
  fusion run or code change is authorized by this investigation.
- Live evidence: PID 3883185 on the 24 GiB RTX 4090 retained 22,600 MiB while
  configured with an 18 GiB estimate; PID 3883197 on the 32 GiB RTX 5090
  retained 26,074 MiB with a 26 GiB estimate. Both use the verified `multi`
  interpreter, one serialized GPU queue, direct OME-Zarr input,
  `content-preibisch-coarse`, and `24x960x960` output chunks. Their candidate
  plans have median 2, p95 4, and maximum 4 source views, so failed candidate
  culling is not the cause.
- Owning mechanism: sigma 7/17 with stride `(1,8,8)` requires a per-side halo
  of `(96,104,104)`, expanding a nominal chunk to `216x1168x1168` (294,672,384
  voxels). One full float32 field per view is 1.098 GiB. Installed
  multiview-stitcher retains uploaded sources, per-view transformed fields and
  their stack, then per-view blend fields and their stack before invoking the
  coarse content-weight function, which also constructs quality, validity, and
  upsampled weight arrays.
- The configured GPU budget is a planning estimate, not an allocator limit. It
  recursively splits only x/y using `source_bytes + expanded_voxels *
  (24*views + 20)`. For four views, its no-source terms are 31.83 GiB for the
  full block, 18.75 GiB for a `24x480x960` child, and 11.05 GiB for a
  `24x480x480` child.
- Retention mechanism: these runs did not request hidden
  `--per-chunk-cupy-cleanup`. The default `cupy_cleanup_context(False)` replaces
  multiview-stitcher's cleanup routine with a no-op until the entire channel
  fusion exits, so `nvidia-smi` reports CuPy's accumulated reusable pool and
  fragmentation as well as current live arrays. The worker's apparent cleanup
  after each output block is therefore disabled in these jobs.
- Astra review: current `nvidia-smi` measurements demonstrate retained-pool
  growth, not that live allocations exceeded the estimator. The safest current
  action is to leave healthy jobs running. For a later authorized restart,
  expose and enable the existing cleanup policy through packaged `lightsheet
  fuse`; when enabled, clear after each CPU-materialized internal sub-block and
  log CuPy `used_bytes`, `total_bytes`, and device free bytes around compute and
  cleanup. Preserve the present budgets until active peaks are measured.
- Rejected first-line alternatives: z sub-blocking barely helps because the
  192-plane halo dominates depth; changing z stride changes the weighting
  algorithm; changing output chunks violates the requested storage contract;
  and a CuPy pool limit excludes CUDA overhead and can create OOMs. Streaming
  views is a larger algorithm change because normalization couples all views.
  Lowering the x/y budget also needs a sampling-phase fix/verification: coarse
  weights sample on a local 8-pixel grid, so arbitrary child origins can change
  pixels even when the output grid and chunks remain fixed.
- Result: diagnosis only; no production process, launcher, or fusion code was
  changed. Remaining measurement gap is the live/cached split and transient
  peak inside the worker; `nvidia-smi` alone cannot provide either.

## 2026-09-19 - Default fusion CuPy cleanup

- Goal: make the existing output-block CuPy pool cleanup the packaged fusion
  default, while retaining an explicit opt-out for measured allocator-reuse
  workloads.
- Contract: cleanup changes allocator retention only; fusion pixels, output
  grid, chunks, weighting, resume identity, and the currently running processes
  remain unchanged.
- Planned validation: focused legacy parser, fusion-wrapper, and packaged CLI
  tests; Ruff and diff whitespace checks. A production timing comparison is not
  part of this change, so allocator-overhead impact will remain unmeasured.
- Implementation: legacy fusion now defaults `per_chunk_cupy_cleanup=True` and
  accepts mutually exclusive hidden flags `--per-chunk-cupy-cleanup` and
  `--retain-cupy-pool`. The public `fuse_tiles` API and packaged `lightsheet
  fuse` command expose the same default-on policy and explicitly forward the
  selected behavior. README guidance documents the default and opt-out.
- Verification: the four focused default/opt-out tests pass; packaged command
  help renders both option names; Ruff passed all five touched Python files;
  `git diff --check` passed. The full `test_fusion.py` plus `test_cli.py` run had
  116 passes and two pre-existing installed-MVS compatibility failures because
  `_fuse_block_zarr_backed` requires the removed/changed `trim_overlap`
  argument. Those failures do not execute or inspect cleanup policy.
- Remaining gap: cleanup-performance overhead and live CuPy used/retained byte
  traces have not been measured. Existing active fusion processes loaded the
  prior default before this edit and are unaffected.

## 2026-09-19 - Diagnose live fusion read amplification

- Goal: quantify source-read amplification in the two active 488 dataset-local
  fusions and identify the reader/cache owner without changing either process.
- Inputs: PIDs 3883185 (R/RTX 4090) and 3883197 (L/RTX 5090), direct deconvolved
  OME-Zarr sources, `content-preibisch-coarse`, `24x960x960` output chunks,
  fit-residual correction, and `--source-cache-max-gib 0`.
- Planned validation: compare live process IO with output progress, inspect
  source shard/chunk layout and exact reader path, and derive overlap reuse from
  the halo and output-grid traversal. No code or running-job mutation is
  authorized by this diagnosis.
- Source layout: representative level-0 sources are uint16
  `(1,1194,1920,1920)` with inner chunks `(1,1,240,240)` and outer shards
  `(1,48,960,960)`. Each full outer shard is 84.375 MiB decoded; the two sample
  stores are 8.20 GiB logical but only 302 MiB and 695 MiB stored.
- Owning cache path: `source_cache_max_bytes == 0` bypasses `source_region`
  entirely and calls the original Zarr deserializer for every transformed ROI.
  The active EDU launcher and workflow explicitly override the packaged 64 GiB
  default with zero.
- Reuse geometry: the 96-plane output-Z halo advances by only 24 output planes.
  Current source requests are roughly 90--108 source planes deep, so adjacent Z
  blocks reread about 90% of the same source depth and repeatedly revisit the
  same 48-plane shards. Y/X halo overlap and shard-boundary sharing add further
  reuse.
- Additional multiplier: budgeted x/y sub-blocking replaces every child entry's
  exact source views with the parent's full view records, then prepares every
  child up front. Thus a two- or four-child computation rereads and retains the
  same parent source ROIs two or four times. The code comment establishes this
  as a crop-sharpness invariant, but the active mode is
  `content-preibisch-coarse`; mode-specific relaxation remains unverified.
- Live 30-second sample after resume: with no new completion markers, L logical
  reads increased by 127.8 MiB and R by 159.8 MiB, while physical reads were
  flat (R added only 16 KiB). Linux page cache is hiding disk traffic, but not
  repeated Zarr decoding and memory copies. The machine had 215 GiB available.
- Existing representative evidence: the prior 230Ptp source-cache benchmark in
  this notebook retained 36.8 GiB, reached 85.1% hits at 3,000 requests, and
  improved adjacent warm batches from 314--354 s without the cache to 160 s.
- Result: confirmed read/decode amplification. The narrow operational remedy is
  to stop passing `--source-cache-max-gib 0` and use the packaged bounded cache;
  48--64 GiB per current job fits available host memory, but no restart or
  launcher edit was made. A later code optimization can avoid duplicated parent
  ROI reads for content-Preibisch sub-blocks after pixel-equivalence testing.

## 2026-09-19 - Enable host source cache for active 488 fusions

- Goal: remove the explicit cache bypass from the canonical EDU workflow and
  resume both active 488 dataset-local fusions with a 64 GiB per-process raw
  source-shard cache.
- Inputs: tmux session `codex-230ptp-488-local-fusions`, launcher
  `/home/chaichontat/nvme/lightsheet/230Ptp/run_488_dataset_local_fusions_20260919.sh`,
  preserved completion markers (L 360, R 180 at the diagnostic snapshot), and
  215 GiB available system RAM before restart.
- Safety: interrupt the owning tmux session gracefully, preserve both temporary
  fusion workspaces, update only the cache option, relaunch the packaged CLI in
  the verified `multi` environment, and confirm interpreter, resume counts,
  nonzero cache configuration, and early cache hits from persistent logs.
- Changes: updated canonical `WORKFLOW_EDU.md` and the active production
  launcher from `--source-cache-max-gib 0` to `64`. Gracefully interrupted the
  original tmux session; both old workers exited and their temporary workspaces
  remained intact.
- Relaunch: recreated tmux session `codex-230ptp-488-local-fusions` from
  `/home/chaichontat/nvme/lightsheet/230Ptp` after explicit Bash activation of
  `multi`. The pane records
  `actual_job_interpreter=/home/chaichontat/miniforge3/envs/multi/bin/python`.
  Persistent logs remain under
  `514488-run-20260916/logs/488-dataset-local-fusions-20260919/`.
- Verification: both legacy workers run from the `multi` interpreter and log a
  64.0 GiB channel-session raw-shard cache. R resumed with 90 completed blocks
  skipped. At 100 requests, L had 66 hits/34 misses and R had 67 hits/33 misses;
  at 200 requests they reached 146/54 and 150/50 with no eviction or rejection.
  Cache residency was 4.4 and 4.1 GiB and the host still had 162 GiB available.
- Caveat: the GPUs acquired unrelated concurrent workers after the original
  launch. At verification, total usage was 23.5/32 GiB on GPU 0 and 10.0/24 GiB
  on GPU 1. The fusion processes were alive and making progress, but dense
  blocks may contend for VRAM; no unrelated process was stopped or modified.
- Status at approximately 17:00: tmux session
  `codex-230ptp-488-local-fusions` and both `multi` workers remained alive with
  no logged traceback, error, or OOM. Completion markers advanced to L 720 and
  R 360; neither final store had been published yet. The source caches reached
  L 5,300 hits/400 misses (93.0%, 33.0 GiB retained) and R 8,153/347 (95.9%,
  28.6 GiB retained), with no eviction or rejection. Per-process CUDA memory
  was only 1,388 MiB for L and 1,706 MiB for R, confirming that default CuPy
  cleanup prevents the previous pool-driven VRAM retention while work proceeds.
## 2026-09-20 - Package the ventricular/Eomes spatial workflow

- Goal: move the reviewed surface, canonical Eomes-profile, recovered-boundary, implicit-contour QC, and later cell-layer stages into `squisher_lightsheet` without retaining prod2-hardcoded paths.
- Inputs: existing package `uv_surface.py`/`surface_intensity.py`; prod2 owners under `/working/eduseg/20260808-prod2/full-run/{vz-surface,svz-cut}`; reviewed prod3 artifacts under `/working/eduseg/prod3/{vz-surface,svz-cut}`.
- Constraints: preserve full NRRD direction geometry; keep direct evidence separate from recovered fields; contour only implicit `r_vz - cutoff`; write fresh atomic artifact directories; do not add a master runner or another normal-ray workflow.
- Planned validation: focused geometry, profile, boundary, contour, CLI, and rotated end-to-end tests followed by a code-simplifier pass limited to this task's diff.
- Implementation: generalized `uv-surface` to preserve rotated/reflected NRRD
  geometry and optional index-zero stride; added reusable exact triangle
  projection, canonical volume-binned profiles, independently supported ordered
  Eomes fits, and direct/recovered implicit-contour QC. New CLI stages are
  `eomes-profiles`, `eomes-boundaries`, and `eomes-boundary-qc`.
- Artifacts: each stage refuses an existing output directory and publishes
  atomically. QC emits PNG, PDF, SVG, and JSON only. Surface artifacts bind
  original and working geometry; profile, chart, boundary, and QC artifacts are
  hash-linked.
- Scope: cell projection and layer assignment remain deferred until contour QC
  is accepted. No master runner, generic artifact framework, normal-ray CLI, or
  production-data run was added.
- Simplifier pass: reused one surface-bundle writer and the existing artifact
  hash owner, removed forwarding methods and unused solver returns, and reduced
  QC to one exact mesh projection per section. No new orchestration abstraction
  was introduced.
- Validation: 97 focused surface, CLI, and regression tests pass, including a
  rotated profiles-to-boundaries-to-QC fixture. Focused Ruff and `git diff
  --check` pass. Added `osqp` 1.1.3 to the lockfile.

## 2026-09-20 - Prod2 packaged ventricular/Eomes validation

- Goal: run the packaged ventricular/Eomes stages on prod2 and compare their
  geometry, profile statistics, recovered boundaries, and section QC against
  the historical prod2 owners.
- Inputs: reviewed mask
  `/working/eduseg/20260808-prod2/Cortex_smoothed_v6_processed.seg.nrrd` and
  accepted relocated 561 fusion
  `/working/eduseg/20260808-prod2/234-3-514561638/fusion-561-v4/fused.ch0.ome.zarr`.
  Its level-2 shape, spacing, and origin exactly match the deleted `fusion-561-v2`
  source recorded by the historical run; content equality is not assumed.
- Historical references:
  `/working/eduseg/20260808-prod2/full-run/vz-surface/smooth` and
  `/working/eduseg/20260808-prod2/full-run/svz-cut/profiles-v3.npz`; review
  sections are source Z indices 210, 294, and 378.
- Output root:
  `/working/eduseg/prod2-surface-validation-20260920`. Each packaged stage will
  use a fresh atomic output directory and the profile stage will run under tmux
  with a persistent log.
- Planned checks: exact surface/chart hashes and array equality where the
  canonical implementation is expected to reproduce prod2; profile shape,
  counters, finite support, and numerical comparison to the historical output;
  boundary ordering and support; metadata hash links; and visual inspection of
  the PNG contour sheet at the three historical sections.
- Run result: the surface stage reproduced the historical 46,416 vertices,
  91,737 faces, and UV coordinates exactly. File hashes differ because the
  packaged surface now records world/intrinsic geometry and the chart binds the
  new surface artifact. Profiles retained exactly the historical depth grid,
  vertex order, valid-count array, valid-volume array, and 1,711,464 nonempty
  bins. Intensities differ because the available accepted input is the newer
  `fusion-561-v4`; they remain strongly concordant with the deleted-v2 result
  (Pearson r = 0.985).
- Production bug and fix: one valid rising crossing landed at exactly 0 um.
  This is a valid retained raw measurement but cannot enter a logarithmic fit.
  `measure_peak_boundaries` now excludes nonpositive crossings from direct fit
  support while preserving the raw value for audit and recovering that vertex
  from neighbors. A focused zero-crossing regression was added; 11 boundary
  and end-to-end integration tests plus focused Ruff pass.
- Boundary validation: wall support and estimated support match the historical
  prod2 0.60/30 um inner and 0.75/80 um outer artifacts exactly. There are zero
  ordering violations. New-versus-historical median absolute differences are
  0.75 um inner and 0.40 um outer; fitted p5/median/p95 are
  9.40/30.82/54.60 um inner and 46.78/76.73/117.83 um outer.
- QC: completed at nearest reference-mask planes 26, 37, and 47 (intrinsic Z
  499.2, 710.4, and 902.4 um) corresponding to the historical requested source
  sections. Both direct and recovered inner/outer contours are present on all
  three sections. Visual inspection shows recovered contours following the
  Eomes-positive ventricular band without crossing; direct gaps are exposed as
  intended. Every surface/profile/boundary/QC provenance link and emitted-file
  hash verifies.
- Final artifacts:
  `/working/eduseg/prod2-surface-validation-20260920`; contour sheet
  `/working/eduseg/prod2-surface-validation-20260920/eomes-boundary-qc/boundary-qc.png`.

## 2026-09-20 - Prod3 packaged ventricular/Eomes run

- Goal: run the prod2-validated packaged ventricular/Eomes workflow on prod3
  through recovered-boundary QC.
- Inputs: reviewed cortex mask
  `/working/eduseg/prod3/cortex_refit_50um_mask.nrrd` (SHA-256
  `e18fcb412e54d9c3c66eb2781b7338a12a683c41836dae3e0bf1ad7485019590`)
  and accepted 561 fusion
  `/working/eduseg/prod3/235-2-561638514-3/fusion-561-preibisch-v1/fused.ch0.ome.zarr`.
- Surface parameters: preserve the reviewed prod3 intrinsic-grid reduction
  `--stride-zyx 8,4,2`, 19.2 um smoothing, intrinsic +X cavity selection, and
  explicit multi-interval-ray acceptance. The historical reviewed reference is
  `/working/eduseg/prod3/vz-surface/ventricular-v1`.
- Profile/boundary parameters: packaged defaults (level 2, stride `2,4,4`,
  20 um mesh pooling, 4.8 um depth bins, rising 0.60/30 um inner fit, falling
  0.75/80 um ordered outer fit). QC sections use reviewed intrinsic Z positions
  354.608248, 1492.117824, and 2438.507370 um.
- Fresh output root: `/working/eduseg/prod3/surface-workflow-v1`; persistent log:
  `/working/eduseg/prod3/surface-workflow-v1/logs/workflow.log`; tmux session:
  `codex-squisher-prod3-eomes`.
- Planned validation: compare packaged surface arrays and topology with the
  reviewed prod3 reference; verify profile support, finite measurements,
  boundary ordering/recovery, every provenance hash link, and all three QC
  contour pairs before downstream cell projection or layer assignment.
- Launch status at 17:17 EDT: the completion-triggered chain is active in tmux
  session `codex-squisher-prod3-eomes`. `uv-surface` completed in 21 seconds;
  world vertices, intrinsic vertices, faces, and topology are exactly equal to
  the reviewed prod3 artifact (37,101 vertices, 73,343 faces, one manifold disk).
  The new UV chart differs only by a rigid 2-D rotation (Procrustes maximum
  residual `4.7e-15`), consistent with its deliberately unoriented disk chart.
  `eomes-profiles` is now active at roughly 307% CPU and 5.1 GB RSS; boundaries
  and three-section QC remain completion-gated behind it.
- Completion: Luna monitoring observed `stage=complete` and `status=0` at
  17:24:24 EDT, 8 minutes 58 seconds after launch. Profile, boundary, and QC
  metadata parsed successfully; `profiles.npz` and `boundaries.npz` opened as
  valid NPZ artifacts, and their recorded output hashes verified. QC PNG, PDF,
  and SVG are present and valid; the PNG is 2420 x 3167 RGBA and the PDF has one
  page. Final QC path:
  `/working/eduseg/prod3/surface-workflow-v1/eomes-boundary-qc/boundary-qc.png`.

## 2026-09-20 - Prod3 1492-um surface continuity experiment

- Trigger: recovered-boundary QC showed a non-anatomical Eomes-band jump near
  intrinsic Z 1492.1 um. Read-only tracing found an upstream 248.03 um surface
  jump between adjacent intrinsic-Y rays; the outermost-+X extractor switches
  between two cortex-mask intervals and triangulates the gap.
- Rejected fixes: stronger boundary smoothing cannot alter the surface;
  removing long faces at 40--200 um breaks the disk/manifold contract; choosing
  the first or widest X interval still leaves a 197.26 um jump because the main
  interval later disappears.
- Experiment: add optional `uv-surface --max-adjacent-y-step-um`. On each
  intrinsic-Z row it splits candidate vertices across excessive adjacent-Y
  world-distance jumps, keeps the longest continuous segment, and rebuilds the
  structured mesh before topology validation. No behavior changes when the
  option is omitted.
- Prod3 candidate: 240 um, just below the measured 248.03 um branch jump. The
  real-data dry diagnostic removes 597 of 37,101 candidate vertices over Z
  1363.2--1492.1 um and preserves one connected oriented manifold disk (Euler
  1, one boundary loop). Focused surface/CLI validation passes: 15 tests plus
  Ruff and diff checks.
- Fresh output root:
  `/working/eduseg/prod3/surface-workflow-y240-v1`; the original mask,
  `/working/eduseg/prod3/surface-workflow-v1`, and reviewed artifacts remain
  unchanged. The full profiles, boundaries, and matched-section QC must be
  regenerated because their geometry identity is surface-hash bound.
- Completion: the full surface -> profiles -> boundaries -> QC chain completed
  successfully. The continuity trim removed 597 vertices across 15 Z rows;
  the resulting surface has 36,504 vertices, 72,073 faces, one connected
  oriented manifold component, Euler characteristic 1, and no non-manifold
  edges. At intrinsic Z 1492.1 um, the largest adjacent-Y world step is now
  22.84 um and the largest intrinsic-X step is 22.37 um; visual QC confirms
  that the 248.03 um horizontal jump is gone in both direct and recovered
  contours. The separate recovered-only hook at Z 2438.5 um remains and was
  not targeted by this surface-domain correction.
- Verification: all surface/profile/boundary/QC provenance hash links match,
  all direct and recovered contour pairs are present at the three requested
  sections, and 32,174 finite wall-support vertices have zero outer-before-
  inner ordering violations. The focused unit, CLI, and workflow integration
  suite passes (16 tests).
- Corrected QC:
  `/working/eduseg/prod3/surface-workflow-y240-v1/eomes-boundary-qc/boundary-qc.png`.
- Mask-overlay review: the QC renderer now draws the binary NRRD boundary as
  a dashed yellow 0.5 contour in the same intrinsic-XY physical coordinates.
  At Z 1492.1 um, the continuity trim retains 103 of the original 142 surface
  samples and changes the row extent from Y 1447.75--2102.03 um to
  1447.75--1921.06 um (39 samples, or 27.5% of that row, removed). The overlay
  confirms that this deletes a real distal part of the masked cortical band,
  so the 240-um longest-segment experiment is too aggressive to adopt as the
  production correction even though it removes the false bridge. Overlay QC:
  `/working/eduseg/prod3/surface-workflow-y240-v1/eomes-boundary-qc-mask-overlay/boundary-qc.png`.
- Decision: reject continuity trimming. The optional trim implementation, CLI
  flag, and focused trim test were removed; 23 focused surface/QC/workflow
  tests and Ruff pass afterward. Production review returned to the original
  untrimmed surface, with mask-overlay QC at
  `/working/eduseg/prod3/surface-workflow-v1/eomes-boundary-qc-mask-overlay/boundary-qc.png`.
- Astra geometry audit: the apparent 1492-um jump is a shortcut across a real
  NRRD-contour overhang, not foreground that should be deleted. At Z 1492.118,
  the continuous smoothed-field contour folds through approximately
  `(1039.2,1921.1) -> (1143.3,1941.3) -> (1240.6,1925.7) ->
  (1287.2,1925.7)` intrinsic XY. The same seam spans roughly Z
  1059.2--2136.9 um. A faithful repair therefore requires an overhang-capable
  conforming mesh strip and a replacement for the current Y-sorted graph
  assumption in `wall_support`; a single-vertex/X interpolation would be
  incorrect. No manual geometry patch was applied pending that broader choice.

## 2026-09-20 - Prod3 reviewed ventricular overhang repair

- Implemented an explicit hash-bound surface patch operation. It preserves all
  original vertices and retained faces, requires an exact directed seam,
  rejects non-disk topology and CGAL-detected self-intersections, validates
  replacement vertices/edge midpoints/centroids against the reconstructed SDF,
  and publishes the concrete patch plus provenance atomically. The historical
  graph extractor remains unchanged.
- Added `trim_caps=False` to boundary fitting and exposed it as
  `eomes-boundaries --no-trim-caps`. This retains every repaired/distal vertex
  and constrains outer >= inner over the complete mesh; the historical trimmed
  graph behavior remains the default.
- Prod3 patch `/working/eduseg/prod3/surface-overhang-patch-v1.npz` replaces
  7,626 parent faces with 18,662 faces and appends 5,518 exact zero-contour
  vertices. Published repaired surface:
  `/working/eduseg/prod3/surface-workflow-overhang-v1/vz-surface`.
- Geometry checks: one connected manifold disk, Euler 1, one 857-edge exterior
  loop, no non-manifold edges or self-intersections. Dense validation sampled
  839,790 patch points: maximum SDF-distance estimate 1.895 um, 99.9th
  percentile 0.498 um, minimum outward-normal cosine 0.888, and no >80 um edge
  in the reviewed region. Original vertices and retained faces are byte-exact.
  Astra found no visible defect or collar blocker in
  `/working/eduseg/prod3/surface-workflow-overhang-v1/surface-repair-qc/surface-repair-qc.png`.
- At 18:47:50 EDT, launched completion-triggered tmux chain
  `codex-squisher-prod3-overhang`: fresh profiles -> no-cap boundaries -> six-
  section matched QC. Persistent log:
  `/working/eduseg/prod3/surface-workflow-overhang-v1/logs/workflow.log`.
- Rejected v1 for FEM use: an appended contour point was only 8.403e-6 um
  from original vertex 23107, producing a face with aspect ratio 2.513e6.
  Minimum twice-area fell from the original 42.74 um^2 to 7.84e-5 um^2;
  stiffness and biharmonic system scaling became pathological. Exact diagnostic
  residuals were 0.0042 inner and 0.246 outer, and OSQP was projected to exhaust
  its 100,000-iteration budget. The v1 chain was stopped during boundaries; no
  v1 boundary artifact was published.
- Rebuilt the same anatomical arcs as patch v2 with uniform 4.64-um arc-length
  spacing and endpoint deduplication. V2 minimum quality is 0.019356 and maximum
  aspect 51.663, equal to the original baseline; minimum twice-area is 41.02
  um^2. Added a production patch-quality rejection floor of 0.01 and regression
  coverage for near-duplicate patch vertices.
- Final v2 root:
  `/working/eduseg/prod3/surface-workflow-overhang-v2`. Dense geometry checks
  again pass (785,430 samples, maximum SDF-distance estimate 1.895 um, minimum
  outward cosine 0.903, unchanged distal extent and no >80-um repaired-region
  edges). Fresh exact boundary-system residuals are 3.97e-12 inner and 2.39e-10
  outer, below the 1e-8 gate. Profiles, no-cap boundaries, and six-section QC
  completed at 19:20:32 EDT. All artifact hashes match; support includes all
  42,015 vertices, outer-before-inner violations are zero, and all 12 requested
  direct/recovered contours are present. Final QC:
  `/working/eduseg/prod3/surface-workflow-overhang-v2/eomes-boundary-qc/boundary-qc.png`.

## 2026-09-20 - Prod3 wall-domain projection correction

- Trigger: full-resolution review of the overhang-v2 QC showed that the cyan
  inner contour returned around the distal cap and that recovered contours
  still contained distal disconnections. The v2 geometry/FEM checks were valid,
  but `eomes-boundaries --no-trim-caps` made every cap face eligible and QC
  selected the closest face before applying vertex eligibility.
- Mechanism: adjacent tissue points could switch from the ventricular wall to
  a spatially closer original cap face. At intrinsic Z 1492.1 um this changed
  the interpolated cutoff from 89.11 to 65.54 um. Masking cap vertices after
  projection removed the return loop but left the main outer contour truncated.
- Code correction: `SurfaceProjector` now accepts a face mask while preserving
  full-mesh face IDs. `eomes-profiles --projection-domain ventricular-wall`
  derives one hash-bound domain from the repaired surface's verified parent,
  restricts projection/assignment/neighborhoods to it, and persists its vertex
  mask and digest. Boundary fitting inherits that mask, and QC restricts
  candidate faces before closest-point projection. `WORKFLOW_EDU.md` now uses
  this domain and full-resolution `--grid-stride 1` QC. QC metadata records
  contour-component counts rather than only contour presence.
- Domain validation on the reviewed v2 surface: 37,284 vertices and 73,671
  faces; all 4,914 appended repair vertices are included. Vertex-mask SHA256 is
  `07e4474306a9794733024c2c5701d1a74749c3b7e86b9d6b86723de419098534`.
- Fresh run: `/working/eduseg/prod3/surface-workflow-wall-domain-v3` completed
  successfully from 19:43:17 to 19:47:57 EDT. Fresh profiles took 2m50s,
  boundaries 18s, and the initial six-section QC 1m32s. Focused geometry,
  profile, boundary, QC, CLI, and integration validation passed 35 tests; Ruff
  and diff checks passed.
- Final full-resolution QC:
  `/working/eduseg/prod3/surface-workflow-wall-domain-v3/eomes-boundary-qc-final/boundary-qc.png`.
  All recovered inner/outer curves are single open paths at Z 354.6, 1059.2,
  1492.1, 1805.3, and 2136.9 um. No recovered curve is closed and the large
  distal return loop is gone. Direct fields remain fragmented where raw
  measurements are unsupported, as expected.
- Remaining uncertainty: recovered outer at terminal Z 2438.5 um contains one
  main path plus two tiny fragments; direct inner/outer there contain 12/4
  components. This is the previously identified terminal weak-support/
  projection ambiguity and was not hidden or cosmetically joined. Recovery at
  the proximal ends of Z 1805.3 and 2136.9 also crosses sparse direct evidence.

## 2026-09-20 - Prod3 confidence-bounded endpoint recovery

- Trigger: wall-domain v3 removed the cap loop, but recovered contours still
  made unsupported endpoint excursions at intrinsic Z 2136.9 and 2438.5 um.
- Diagnosis: the Z 2136 excursion was driven by heterogeneous rising-edge
  evidence, including a 103.5-um anchor whose peak had only three pooled
  samples and nearby profiles with no rising crossing. At Z 2438, adjacent
  image points switched about 119 um between non-rim closest-wall footpoints,
  changing the outer cutoff by 13.09 um and creating a false marching-contour
  connector. Finer section sampling did not remove either mechanism.
- Correction: raw direct crossings remain unchanged for diagnosis. Boundary
  fitting now uses crossings with at least five samples at the peak/crossing
  bracket, and recovered fields are eligible only within 20 um mesh-geodesic
  distance of those reliable anchors. QC invalidates both grid endpoints when
  adjacent closest-wall footpoints differ by more than twice the selected
  domain's 99th-percentile edge length; prod3's threshold was 51.073 um.
  Metadata now records crossing counts, fit support, recovery distance, and
  projection-continuity counts.
- Fresh output:
  `/working/eduseg/prod3/surface-workflow-confidence-v4`. Fit support is
  29,464/31,352 observed inner crossings and 30,820/32,663 observed outer
  crossings. Confidence-bounded estimates cover 33,169 inner and 34,056 outer
  vertices.
- Result: the large Z 2136 and Z 2438 excursions are removed. The result is a
  partial confidence-bounded reconstruction, not six artificially continuous
  bands. Remaining gaps and small islands mark missing/low-count evidence or
  real closest-wall branch ambiguity; they must stay unassigned rather than be
  joined. Final QC:
  `/working/eduseg/prod3/surface-workflow-confidence-v4/eomes-boundary-qc/boundary-qc.png`.
- Validation: 36 focused geometry/profile/boundary/QC/CLI/integration tests
  pass; Ruff and `git diff --check` pass.

## 2026-09-21 — Method 10 Mattes complete recovery support

- Goal: Allow the EDU four-stage `lightsheet cross-register complete-recovery` policy to consume a completed native Method 10 Mattes checkpoint for squid 638→514 registration.
- Trigger: The prior L 638→514 Method 10 Mattes run accepted 1,226/4,242 windows before recovery; the current Method 6 fit has low acceptance. The prior artifact is at `/working/eduseg/20260709/squidedu/L-514638/cross-channel-ch0-to-fused-ch1/method10-mattes400k-480cube-th50-interpolation-recovery-adjacent-good-v3-linear-filter-native-only-v1-linear-outliers-only-v1/fused_fixed_method8_summary.json`.
- Mechanism: `complete_recovery.py` rejected `method10-mattes` at input identity and hardcoded Method 6 attempt prefixes for spatial and linear recovery. `recover-native` already supports Method 10 Mattes.
- Change: Accept only Method 6 and Method 10 Mattes checkpoints; derive native source matching and recovery attempt names from the checkpoint method. Update the CLI documentation. No production fit was switched or started as part of this code change.
- Verification: Before change, a Method 10 fixture raised `Complete recovery requires native Method 6`. After change, `tests/test_complete_recovery.py` and `tests/test_fused_fixed_method10_recovery.py` passed (44 tests); targeted Ruff check passed. A Method 10 fixture traversed all four complete-recovery stages and published a manifest.
- Inputs for future run: current L native 638 positions at `/working/eduseg/20260709/squidedu/L-514638/hcr-514-run-20260920/registration-638-anchor-connected/`; fixed native 514 fusion at the sibling `fusion-514-track2-native/fused.ch1.ome.zarr`. A new Method 10 fit and native recovery checkpoint are still required before invoking complete-recovery.

## 2026-09-21 — L squid 638→514 Method 10 recovery chain

- Goal: Replace the weak L Method 6 fit with a full Method 10 Mattes fit, native recovery, and complete recovery against the current fused 514 reference.
- Stopped: `codex-squidedu-l638-crossreg` by Ctrl-C; tmux session ended. Its historical `cross-registration-638-to-514/method6` partial output was preserved.
- Exact runner: `/working/eduseg/20260709/squidedu/L-514638/hcr-514-run-20260920/cross-registration-638-to-514/method10-current-20260921/run-full-recovery.sh`, tmux `codex-squidedu-l638-method10-recovery`.
- Inputs: native 514 positions `registration-514-track2/registration.positions.json`; fixed fused 514 `fusion-514-track2-native/fused.ch1.ome.zarr`; moving 638 cross-placed positions `cross-registration-638-to-514/global-phase/global-phase.positions.json`; source geometry `registration-638-anchor-connected/registration.json` (78 tiles), channel 0.
- Threshold: 40 on fused 514 at level 2, reviewed manually at Z=150,375,600. Review PNG and JSON are in the run root (`fixed-514-threshold-40-review.*`). It covers visible tissue and excludes most background.
- Fit settings: Method 10 Mattes, 50 bins, 400,000 samples, 480³ core, 528³ context, identity starting linear matrix, level-2 Method 8 initializer, minimum corr 0.15 and gradient NCC 0.24. CUDA GPU0 UUID `GPU-5d7f9022-ea7f-32f4-71b5-e424ae96aafd`, Conda `multi`; full expected window count 3,276.
- Recovery: `lightsheet cross-register recover-native --rerun-all-eligible` with 638 adjacency, then `lightsheet cross-register complete-recovery`; each stage runs only after prior exit and output checks. Final expected output `method10-current-20260921/complete-recovery/fused_fixed_method8_summary.json`; stage logs and `run.exit` in run root.
- Launch errors corrected before fitting: Conda activation under Bash nounset failed; moved `set -u` after activation. Legacy sweep then required `CUDA_PATH` and `LD_LIBRARY_PATH`; set them to Conda CUDA target and lib as earlier squid scripts did. These failed starts created no fit output.
- Verified running: tmux session live, GPU0 worker PID 2671064 using 11.4 GiB; GPU1 remains occupied by full L/R 514 fusion. First 28 Method 10 windows included one accepted window. Final fit/recovery is not complete yet.

## 2026-09-21 — Correction: existing fit versus current 638→514 task

- User clarified that a fit already exists. Interrupted the newly launched L Method 10 fit in `codex-squidedu-l638-method10-recovery`; the tmux session ended and the GPU0 worker is gone. Preserved its partial `fit/` files, and changed `method10-current-20260921/run.exit` to 130 because the run was interrupted before publishing a fit summary or starting either recovery stage. The separate L/R 514 fusion tmux session remains running.
- Searched completed `fused_fixed_method8_summary.json` files under the squid 20260709 dataset. The completed L Method 10 summary is `L-514638/cross-channel-ch0-to-fused-ch1/method10-mattes400k-480cube-th50-phase-fallback/fused_fixed_method8_summary.json` (4,242 windows, 1,226 initially accepted). It fits 101 older L tiles' channel 0 to an older fused channel 1 image at `L-514638/fusion-ch1/fused.ch1.ome.zarr`, which no longer exists. Its moving positions are `L-514638/cross-channel-ch0-to-fused-ch1/L-514638.deconv.with-shape.positions.json`.
- The requested current L 638→514 registration has 78 moving 638 tiles and uses `hcr-514-run-20260920/fusion-514-track2-native/fused.ch1.ome.zarr` as fixed image. Its global phase placement is complete, but there is no completed per-window Method 10 summary under `hcr-514-run-20260920/cross-registration-638-to-514/`. The prior Method 6 output there is partial. Do not feed the older cross-channel fit into 638→514 complete recovery without an explicit mapping/compatible summary; its tile set and fixed image differ.
- Next: Ask for the path to the completed 638→514 fit the user means, or clarify if they instead want recovery of the older cross-channel fit.

## 2026-09-21 20:34 EDT — User authorized full current Method 10 run

- User clarified to do Method 10 fully for current L 638→514 after the older cross-channel fit proved incompatible. Resumed the existing script `L-514638/hcr-514-run-20260920/cross-registration-638-to-514/method10-current-20260921/run-full-recovery.sh` in tmux `codex-squidedu-l638-method10-recovery`; no new runner was created.
- Preflight: all required 514 and 638 position inputs, fixed fused 514 Zarr, and 638 adjacency file exist; threshold 40 review is accepted; the old partial fit had 82 cached windows and no published summary. Runner reported `start windows=3194` remaining of 3276 at 20:34 EDT.
- Verified live fit worker PID 2755549 on GPU0 UUID `GPU-5d7f9022-ea7f-32f4-71b5-e424ae96aafd` using 11,960 MiB. The independent L/R 514 fusion remains in tmux `codex-squidedu-lr-514-full` on GPU1. Existing runner gates native recovery on completed fit summary and complete recovery on completed native checkpoint.
- Expected completed output: `L-514638/hcr-514-run-20260920/cross-registration-638-to-514/method10-current-20260921/complete-recovery/fused_fixed_method8_summary.json`; status/logs in that run root. Run is not yet complete. Graceful stop: `tmux send-keys -t codex-squidedu-l638-method10-recovery C-c`.

## 2026-09-21 20:45 EDT — R 638→514 started

- User asked about R after L Method 10 launched. Current R native 514 and 638 registrations are complete: 72 and 63 tiles respectively. R has no native 514 fused image, no prior 638→514 global phase placement, and no current Method 10 per-window summary.
- Started `lightsheet cross-register global-phase` in tmux `codex-squidedu-r638-global-phase` with R 514 Track 2 positions as fixed (channel 1) and R 638 Track 1 positions as moving (channel 0), both reading `R-514638/hcr-514-run-20260920/deconvolved`, level 2, log1p for both, 32 Z-slab planes, FFT highpass 1,4,1, max residual 100 µm, orthogonal Z. Log: `R-514638/hcr-514-run-20260920/run-logs/cross-registration-638-global-phase.log`. Output: `R-514638/hcr-514-run-20260920/cross-registration-638-to-514/global-phase/global-phase.positions.json` plus summary and QC.
- Verified phase session live and reading slabs. Separate L Method 10 and combined L/R 514 fusion sessions remain live. R Method 10 awaits its own R 514 fused channel 1 reference. Avoid simultaneous GPU-heavy fusion with the two active GPU jobs; assess resource availability and launch R reference fusion after those jobs free capacity.

## 2026-09-21 20:55 EDT — Correction: combined fusion is R's 514 reference

- User clarified that the ongoing combined L/R 514 fusion uses R input. A separate R-only 514 fusion is unnecessary; the fixed fused image for R 638→514 Method 10 will be `/working/eduseg/20260709/squidedu/post-fusion-lr-ch1/final-fusion-514/fused.ch1.ome.zarr` when published. Retract the separate R fusion dependency in the preceding entry.
- R native global phase completed: 72 fixed R 514 tiles, 63 moving R 638 tiles, correlation 0.60175→0.65422, total moving shift ZYX µm [0.25155, -11.91222, 12.10461]. Summary and QC under `R-514638/hcr-514-run-20260920/cross-registration-638-to-514/global-phase/`.
- Coordinate handoff verified against all 72 R 514 tiles: `post-fusion-lr-ch1/joined-514.positions.json` translates native R positions by constant ZYX µm [20.7653001663, 700.8928886211, 1278.3383358553]. L 514 positions remain unchanged. Intensity `cross-dataset-match-514/match.json` names L as reference for intensity scaling, which is distinct from the geometry handoff.
- Derived R 638 positions in the joined 514 frame at `R-514638/hcr-514-run-20260920/cross-registration-638-to-514/joined-514/global-phase.positions.json` by adding that exact offset to each of 63 native R global-phase translations. It preserves the source phase result and records source and offset in diagnostics. The joined 514 image is still being written, so R Method 10 has not started. Use `post-fusion-lr-ch1/joined-514.positions.json` for fixed geometry, the derived R 638 positions for moving placement, and `R-514638/hcr-514-run-20260920/registration-638-track1/registration.json` for source geometry.

## 2026-09-21 21:50 EDT — Final R reference correction and R-only fusion

- User corrected the target: R 638 must register to **R-only fused 514**, not the combined L/R 514 fusion. The native R global-phase output `R-514638/hcr-514-run-20260920/cross-registration-638-to-514/global-phase/global-phase.positions.json` is the correct moving placement. The joined-frame derivative `.../joined-514/global-phase.positions.json` from the prior entry is obsolete for this task and must not be used. It remains on disk because deletion was not authorized.
- Existing `R-514638/fusion-571[-l0]/fused.ch1.ome.zarr` artifacts are from the older 571 registration (`reg-571`) and are not the current R 514 reference. No current R-only 514 fused image existed before this run.
- Dry-ran the current R fusion command against `R-514638/hcr-514-run-20260920/registration-514-track2/{registration.positions.json,registration.json}`, `grid-514-track2.ome.zarr`, and `fit-residual-514-track2/correction.json`; CLI accepted it. First tmux wrapper exited immediately before logging because the tmux default shell did not accept `set -o pipefail`; no fusion work was done. Relaunched with a direct `exec env ... lightsheet fuse` command.
- Active R fusion: tmux `codex-squidedu-r514-fusion`; log `R-514638/hcr-514-run-20260920/run-logs/fusion-514-track2-native.log`; target `R-514638/hcr-514-run-20260920/fusion-514-track2-native/fused.ch1.ome.zarr`. GPU0 UUID `GPU-5d7f9022-ea7f-32f4-71b5-e424ae96aafd`, internal GPU working-memory budget 12 GiB, raw source cache 32 GiB, staging 4 GiB; channel 1, full native level 0, 24x960x960 chunks, JPEG-XR, content-Preibisch coarse weights. Verified live session entered direct OME-Zarr write with 11,651 candidate blocks. L Method10 is also on GPU0 and combined L/R fusion is on GPU1; monitor for memory errors.
- Next after R fusion publishes: review threshold on R fused 514; run R 638→R fused 514 Method 10 using fixed position `R-514638/hcr-514-run-20260920/registration-514-track2/registration.positions.json`, moving placement native R global phase output, and moving source `R-514638/hcr-514-run-20260920/registration-638-track1/registration.json`; then native and complete recovery. Do not use joined L/R fusion or shifted R positions for this registration.

## 2026-09-22 06:10 EDT — Both 514 fusions complete; R Method 10 launched

- R-only 514 fusion completed and atomically published at `/working/eduseg/20260709/squidedu/R-514638/hcr-514-run-20260920/fusion-514-track2-native/fused.ch1.ome.zarr` at 02:18 EDT. Provenance manifest status is `complete`, channel 1, with fixed R 514 Track 2 positions and registration. Combined L/R 514 fusion independently completed at `/working/eduseg/20260709/squidedu/post-fusion-lr-ch1/final-fusion-514/fused.ch1.ome.zarr` at 06:05 EDT; it is not R 638's fit reference.
- L Method 10 fit completed 3,276 windows with 1,690 accepted, 1,586 rejected, 0 errors, median selected correlation 0.45987. Its tmux `codex-squidedu-l638-method10-recovery` is now in native recovery; complete recovery has not started as of 06:10 EDT.
- R-only fused 514 threshold review at level 2, Z=151/378/604, accepted threshold 40 after visual inspection. Image and JSON under `R-514638/hcr-514-run-20260920/cross-registration-638-to-514/method10-current-20260922/fixed-R514-threshold-40-review.*`.
- Created R runner by adapting the established L fit→native recovery→complete recovery script, with R-only fused 514, native R global-phase 638 placement, R 638 Track 1 source/adjacency, expected 2,646 windows, and GPU1 UUID `GPU-94714ab6-ba56-a7d9-6dc3-adc866b00bd6`. Added preflight that verifies R fused provenance and phase input paths/counts. Script: `R-514638/hcr-514-run-20260920/cross-registration-638-to-514/method10-current-20260922/run-full-recovery.sh`; `bash -n` and preflight passed.
- Launched in tmux `codex-squidedu-r638-method10-recovery` at 06:09 EDT. Runner reported 2,646 windows and entered fit; initial masked-background windows rejected as expected. Log `.../method10-current-20260922/run.log`; expected final `.../method10-current-20260922/complete-recovery/fused_fixed_method8_summary.json`. R fit/recovery is not yet complete. Graceful stop: `tmux send-keys -t codex-squidedu-r638-method10-recovery C-c`.

## 2026-09-22 07:15 EDT — Reuse fusion source cache for 638 registration

- Goal: Make repeated 638 moving-source reads in the L/R Method 10 fit use the existing fusion `SourceChunkCache`; preserve exact pixel values and completed-window resume identity.
- Inputs: `lightsheet/src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py` (`SourceChunkCache`), `tile_phase.read_tile_indexed_z_patch`, `channel_affine._read_level0_crop`, and `nvme/lightsheet/scripts/{try_fused_fixed_method8.py,run_fused_fixed_method8_sweep.py}`. The current deconvolved 638 source is sharded uint16 OME-Zarr with outer chunks `(1,48,960,960)` and JPEG-XR inner chunks.
- Reproducer: new sharded CZYX reader test failed with `TypeError: read_tile_indexed_z_patch() got an unexpected keyword argument 'source_cache'` before implementation. The same test passed after adding a bounded raw-shard read path; the script I/O test file passed 18 tests. Real-data benchmark and final simplification are pending.
- Active L native recovery and R fit processes remain untouched; their already loaded Python code cannot receive this change. The fusion cache was process-local and held 514 channel 1, so only its implementation can be reused for 638 channel 0.
- Implementation: `tile_phase.read_tile_indexed_z_patch` now optionally reuses fusion's `SourceChunkCache` for contiguous CZYX OME-Zarr raw shards, keyed by resolved source path, level array, channel, and shard index. `channel_affine._read_level0_crop` forwards the cache. The trial keeps one bounded cache per worker and the sweep and packaged `recover-native` CLI accept `--source-cache-max-gib`; runtime cache size is excluded from fit `cache_config`, preserving completed-window resume identity. Existing TIFF/ZYX and zero-budget reads keep the prior path. The code-simplifier pass kept one raw-crop assembly helper and one worker cache-lifetime helper; no new module or parallel registration script was added.
- Verification: `test_tile_phase.py` 33 passed; `test_fused_fixed_cli.py` and `test_fused_fixed_method10_recovery.py` 52 passed; script `test_fused_fixed_io_optimization.py` 18 passed; `ruff check --no-cache --ignore E402` passed for changed files (`E402` remains pre-existing in the trial's codec-registration import order). `git diff --check` and both runner `bash -n` checks passed.
- Real R 638 read check: source `/working/eduseg/20260709/squidedu/R-514638/hcr-514-run-20260920/deconvolved/R-514638.000.ome.zarr`, channel 0, two overlapping `528³` crops at Z/Y starts 0/0 and X starts 0/432. Cached and uncached pixels matched exactly. Cache-first reads were 2.611 s and 0.457 s; later uncached warm reads were 5.139 s and 3.540 s. Cache recorded 11 misses, then 11 hits, retaining 0.906 GiB; process max RSS was 4.02 GiB. This is one representative pair, not a full-run speed estimate.
- Rollout: gracefully stopped L and R tmux sessions, preserved all window JSON/checkpoints, added `--source-cache-max-gib 8` to the existing `run-full-recovery.sh` fit and native-recovery commands for each side, and resumed the same tmux names and output directories at 07:23 EDT. L fit resumed with 0 pending windows and native recovery with 531; R fit resumed with 1,858 pending windows. At 07:24 EDT both sessions were live and their stage logs were advancing. L log: `/working/eduseg/20260709/squidedu/L-514638/hcr-514-run-20260920/cross-registration-638-to-514/method10-current-20260921/run.log`; R log: `/working/eduseg/20260709/squidedu/R-514638/hcr-514-run-20260920/cross-registration-638-to-514/method10-current-20260922/run.log`. No final recovery or fit result has been published by the resumed jobs yet.
- Final check: the same fusion shard policy now uses `getattr(array, "shards", None) or array.chunks`, preserving unsharded Zarr compatibility; focused sharded and unsharded reader cases passed after this change. Both tmux sessions were still live, their new stage logs continued to advance, and scoped Ruff (excluding the trial's four pre-existing `E402` reports) passed.

## 2026-09-22 07:28 EDT — Increase 638 registration source cache to 64 GiB

- At user request, changed `--source-cache-max-gib` from 8 to 64 in both the fit and native-recovery commands of the existing L and R `run-full-recovery.sh` scripts. No registration inputs, output directories, or checkpoint identity changed. Host memory at restart was 251 GiB total, 198 GiB available; each worker's cache budget is a pixel-payload cap, not an RSS cap.
- Sent Ctrl-C to both tmux jobs and confirmed both sessions exited before editing. Both scripts passed `bash -n`. Restarted the same L and R tmux names with their existing scripts and `--resume` at 07:27 EDT.
- L fit skipped all completed windows and validated its 3,276-window summary; native recovery resumed with 499 pending windows (down from 531 at the prior restart). L log: `/working/eduseg/20260709/squidedu/L-514638/hcr-514-run-20260920/cross-registration-638-to-514/method10-current-20260921/run.log`.
- R fit resumed with 1,789 pending windows (down from 1,858 at the prior restart) and wrote a new accepted window after launch. R log: `/working/eduseg/20260709/squidedu/R-514638/hcr-514-run-20260920/cross-registration-638-to-514/method10-current-20260922/run.log`. Both tmux sessions were live; final recovery outputs remain pending.

## 2026-09-22 10:18 EDT — L Method 10 interpolation policy gap

- Goal: Recover the one tissue-bearing rejected L 638→514 window, `L-514638.040.qy2_qx0.z01440`, without modifying the immutable published complete-recovery artifact. Input: `/working/eduseg/20260709/squidedu/L-514638/hcr-514-run-20260920/cross-registration-638-to-514/method10-current-20260921/native-recovery/fused_fixed_method8_summary.json`.
- Diagnosis: This window had a plausible interpolated initializer from 24 good same-tile windows. Its native Method 10 attempt failed with return code 1, while an initializer-only attempt made the overall rerun `accepted`; native recovery recorded `native_rerun_attempt_mismatch`. Complete recovery promoted only `native_rerun_rejected`, so the interpolated initializer stayed rejected. The other 1,278 final rejections were fixed-image windows more than 95% masked.
- Reproducer: Added a focused Method 10 stage test in `lightsheet/tests/test_complete_recovery.py`; it failed before the fix because the mismatch row remained rejected. Updated `complete_recovery._promote_spatial_fallbacks` to also handle exactly `native_rerun_attempt_mismatch`. The test also checks that `native_rerun_grad_ncc_not_improved` remains rejected. The complete recovery test file now passes (12 tests), Ruff passes on the changed files, and `git diff --check` passes.
- Planned output: run the packaged `lightsheet cross-register complete-recovery` CLI from the existing native checkpoint into a new `complete-recovery-promote-mismatch-20260922` directory, then verify its manifest and all final non-mask statuses. The original `complete-recovery` directory remains historical.
- Result: In the `multi` environment, ran `lightsheet cross-register complete-recovery --native-summary "$BASE/native-recovery/fused_fixed_method8_summary.json" --output-dir "$BASE/complete-recovery-promote-mismatch-20260922"` with `BASE=/working/eduseg/20260709/squidedu/L-514638/hcr-514-run-20260920/cross-registration-638-to-514/method10-current-20260921`. The packaged CLI published a `complete` manifest at `$BASE/complete-recovery-promote-mismatch-20260922/manifest.json`; log: `$BASE/complete-recovery-promote-mismatch-20260922.log`. A second invocation reused the artifact and verified its identity/hash contract.
- Final L summary: 3,276 windows; 1,998 accepted, 1,278 rejected, 0 errors. Exactly one status changed versus the historical output: `L-514638.040.qy2_qx0.z01440` is now accepted as `interpolated_method10_mattes_spatial_recovery`, supported by 24 same-tile good windows. All remaining rejections are `fixed_threshold_mask_too_masked`. The spatial fallback promotion report rose from 6 to 7. A focused registered center-plane overlay is at `$BASE/complete-recovery-promote-mismatch-20260922/L-514638.040-qy2_qx0-z1440-interpolated-overlay.png`; cross-channel appearance limits visual interpretation. No fused 638 output was materialized in this step.
- Updated the existing L `run-full-recovery.sh` to use and validate the new final directory if resumed; `bash -n` passed. The historical `complete-recovery` artifact was not changed. Tests: `lightsheet/tests/test_complete_recovery.py` 12 passed; Ruff on the changed source/test passed; `git diff --check` passed.

## 2026-09-22 10:37 EDT — Focused Method 6 trial before L 638 fusion

- User deferred L 638 fusion to compare Method 6 on regions where native Method 10 was unusable. No L 638 fusion or materialization has started. This is a diagnostic trial; it does not replace the approved Method 10 complete-recovery summary.
- Target: the seven signal-bearing L windows selected as `interpolated_method10_mattes_spatial_recovery` in `/working/eduseg/20260709/squidedu/L-514638/hcr-514-run-20260920/cross-registration-638-to-514/method10-current-20260921/complete-recovery-promote-mismatch-20260922/fused_fixed_method8_summary.json`. Excludes 1,278 windows rejected by the fixed-image background mask and the 78 linear-mapping recoveries/321 accepted-transform outlier replacements. Wrote exact tile/quadrant/Z identities to `/working/eduseg/20260709/squidedu/L-514638/hcr-514-run-20260920/cross-registration-638-to-514/method6-spatial-fallback-trial-20260922/window-filter.json`.
- Method comparison: reuse each target's saved spatial recovery initializer from the Method 10 native checkpoint; keep the same fused 514 reference, L 638 source geometry, 528³ window, threshold 40, correlation/gradient gates, and linear intensity preparation; change only the native optimizer to Method 6. GPU0 had 599 MiB in use; R registration was active on GPU1. Trial output remains separate under `method6-spatial-fallback-trial-20260922`.

## 2026-09-26 — Independent review of cross-channel registration and recovery

- Goal: Assess the current image-arbitrated replacement and native-rerun promotion policies; identify why Method 6 misses or rejects windows with measurable residual shifts. Review only; no implementation or registration reruns requested.
- Inputs: WORKFLOW_EDU.md section 7; current recovery/QC modules; /home/chaichontat/nvme/lightsheet/scripts/try_fused_fixed_method8.py and native backend; /working/eduseg/prod3-2/235-4-edu-workflow/qc-interp-investigation-20260926, complete-recovery-v2 outputs, and /working/eduseg/20260709/lab-notebook.md prior field-validity findings.
- Starting state: Many pre-existing tracked and untracked changes, including all three recovery/QC modules; preserve them. A separate why-no-native-fit investigation is active.
- Validation: Trace objective, capture/phase behavior, pyramid stage transitions, candidate selection, and exact image masks; cross-check stored per-window attempts and diagnostic tables. Separate measured facts from physical/optimizer hypotheses. Notebook-only mutation.

### 2026-09-26 10:00 EDT — Review findings and focused image checks

- Scope completed: Read-only code/data review plus CPU image scoring; no implementation, native registration reruns, commits, or output rewrites. Existing uncommitted changes preserved. Active external investigation artifacts were read as snapshots.
- Native objective owner: `nvme/lightsheet/scripts/try_fused_fixed_method8.py` `_run_level2_metadata` and `_run_level0_from_level2` construct fixed masks but do not pass `fixed_mask_zyx` to `_register_native_device_quiet`. `native_reg3dgpu.register_method6_device` already supports that argument. Thus production Method 6 optimizes unmasked normalized intensity volumes, while quality gates use fixed-threshold masking and QC uses masked high-pass NCC.
- Backend trace: `/home/chaichontat/microImageLib/src/api_subfunc.cu` Method 6 uses local Powell 6-DOF then 12-DOF, rigid tolerance 0.01 and final tolerance from CLI (1e-4). `api_powell.c` starts line bracketing at 0/1 and limits metric evaluations, not a declared displacement radius; the two stages share the evaluation counter. Initializers are prewarped into same-sized crops with zero fill. The unmasked cost centers images before residual warps and includes crop/support effects; it is not the QC objective. Do not infer a fixed capture radius or a sign bug from this alone. The synthetic seed-range experiment includes zero-filled finite crops, so its x drift does not isolate transform-convention correctness.
- Stage control: `_run_level2_metadata` can exit before native optimization if both initializers fail final image gates. Sweep `_run_task` at line 820 skips level 0 when level 2 is rejected unless forced. Among v1 spatial windows: 405 168/278 skipped level 0 and 64 never ran native level 2; 514 98/286 and 16 respectively. Coarse native output can reject a window despite an accepted phase initializer. Recovery uses a field initializer at level 0 with phase skipped.
- Existing replay snapshot (`why-no-native-fit/l0_replay_405.csv`, 14 distinct windows): recorded-initializer replays reproduce stored native translations exactly at CSV precision. Starting from the NCC-estimated translation leaves 12/14 below one level-2 XY pixel, but only 4/14 pass; 8 fail gradient NCC and 2 violate affine bounds. Median translation-only error is 0.335 level-2 px. These references keep an existing matrix and are not independent full-affine ground truth. At measured-shift references the gradient gate commonly fails even when initialized near them; component data do not support a z-gradient-only explanation. Gate-eval CSV contains repeated window/candidate records from active work; deduplicate by window/level/candidate before aggregating.
- Diagnostic caveat: `why-no-native-fit/gate_eval.py` reads moving channel 0, and `l0_replay.py` sets moving_channel=0 irrespective of CH. Their current 405 runs match channel 0. A 514 run must use configured channel 2 before interpreting it. No diagnostic code edited.
- Verified v2 aggregate accepted counts unchanged: 405 1172/3072, 514 1182/3168. V2 XY counts match the prompt. The remeasurement script does not rerun z scans, so this does not establish full 3D improvement. Remaining spatial interpolations: 405 176 = 63 aligned / 35 mild / 69 ambiguous / 9 shifted; 514 233 = 92 aligned / 29 mild / 100 ambiguous / 12 shifted. The thirteenth shifted 514 window is a promoted native rerun: 235-4-561638514-3.064.qy1_qx1.z01440.json (gradient 0.04929 to 0.14641, rule gradient_ncc_margin), still +4 level-2 x pixels by v2 QC. Thus the statement that all remaining shifted windows are interpolations has one exception.
- Fix 1 assessment: Shared fixed planes and intersection support make zero-shift comparison meaningful. The 0.02 margin is conservative policy, not calibrated uncertainty or distance. Scorer accepts a single usable plane; 1000 correlated pixels are not 1000 independent observations. Plane z is determined by first candidate; scores can change on candidate reordering with z differences. Fixed level 2 reduces z by four while the moving level-2 reader samples native z. Missing/tied evidence keeps a previously accepted native, reasonable for replacement protection; re-acceptance after a physical-field rejection remains uncertain without image evidence.
- Fix 2 assessment: Baseline and native attempt gradient metrics can use different fixed crops/masks because native pretranslation re-reads and normalizes the fixed crop. Differences are not strictly paired. A +0.05 rule favors interpolation when both scores are low; it admits a relatively improved native with weak absolute evidence. The corr-only branch does not require native to beat interpolation. It also ignores other earlier candidate transforms. Promotion still accepts interpolation with only >=2 plausible support transforms and no image-confidence/model-validity condition.
- Fresh read-only image check: `PYTHONDONTWRITEBYTECODE=1 /home/chaichontat/miniforge3/envs/multi/bin/python -u -` invoked `WindowImageScorer(fusion-561/fused.ch0.ome.zarr,400)` on native-control rows, with candidates [native, x-2um, x-1um, x+1um, x+2um], converting x using 1.1512282599518298/4 um per level-0 pixel; matrix and other translation components fixed. Six controls chosen at first/middle/last sorted (moving_z,tile) entries. 405 windows: 063.qy0_qx0.z00000, 022.qy1_qx0.z01440, 082.qy0_qx0.z01972. 514 windows: 033.qy2_qx1.z00000, 062.qy1_qx1.z00960, 051.qy2_qx1.z01972. Loss at +/-1um ranged -0.00902 to +0.19196; at +/-2um +0.04574 to +0.45665. In 514.033, +1um cost only 0.000915; in 514.051, -1um improved 0.00902. This is a sensitivity probe, not full margin calibration or ground truth validation.
- Additional paired scores [interpolated, native] on original pixels: 405.096.qy1_qx0.z00000 [0.13133,0.05896]; 514.043.qy0_qx0.z00960 [0.01485,0.04985]; 514.064.qy1_qx1.z01440 [-0.01744,0.04232]. The first cautions against treating translation-vector proximity as full image alignment; the latter two demonstrate that a relative improvement can leave both candidates poorly supported.
- Execution issue: Initial sandboxed pixel-scoring probe produced no output/progress and was interrupted (exit 130). Scoped elevated read-only probe completed, with no data writes. No GPU jobs launched by this review.
- Interpretation: Evidence supports an objective/support mismatch, limited local capture after field extrapolation, and scale/content-dependent gate false negatives. X-by-tile-by-z structure is consistent with an unmodeled spatial/acquisition field and interpolation selection effects; it does not identify illumination side, stage drift, or PSF asymmetry. A smooth within-window x(z) shift is representable by affine x-z shear; abrupt switching or multiple displaced images is not.
- Recommended order: (1) correct masked/support-aware native objective and validate its landscape; (2) integrate bounded 3D structural translation capture before affine, including recovery; (3) separate coarse refinement eligibility from final geometry/image acceptance and preserve image-supported candidates; (4) use paired, scale-matched, uncertainty-aware arbitration; (5) gate field predictions by blocked-z predictive error/support and retain inferred versus verified provenance. Validate on held-out tiles/slabs, all remaining shifted and ambiguous windows, original native controls, and explicit XYZ perturbations, with fresh 3D residual checks.

## 2026-10-01 — Architecture and maintainability review

- Goal: User-requested thorough Astra subagent review of `squisher-segment` and `lightsheet`; assess current working-tree architecture, ownership, contracts, and maintainability. Review only; no implementation changes or production runs.
- Inputs: `segment/`, `lightsheet/`, shared `squisher/` interfaces, package manifests, documentation and tests. Existing tracked and untracked changes are preserved and included as the current snapshot.
- Method: Independent Astra reviews of segmentation execution, lightsheet orchestration, and data/artifact contracts; parent review of packaging, dependency structure and verification coverage, followed by consolidation of prioritized findings.
- Validation: Static code and test inspection plus bounded read-only diagnostics when useful; no claims of full GPU or production validation. Planned output: `ARCHITECTURE_REVIEW.md`.

### 2026-10-01 — Architecture review completed

- Output: `ARCHITECTURE_REVIEW.md`, consolidating three independent Astra reviews plus parent CI/package/dependency inspection. Review snapshot: base commit `75eb478` with existing uncommitted changes. No implementation edits, commits, production runs, or GPU computation performed.
- Reproduced on bounded CPU inputs in `/home/chaichontat/miniforge3/envs/multi/bin/python`: training preparation truncates three channels to two; retraining with `skip_trt=True` can leave an accepted old TensorRT plan beside a new checkpoint (training backend stubbed); registered-source loading accepts identical-shape channels separated by 1000 µm in X; rechunk at start level 0 publishes complete output with missing fusion/N4 sidecars; regionprops opens an incomplete OME source and ignores changed root provenance in its identity; actual TL/TR workflow with stage stubs forwards channel 1 everywhere except registration, whose backend selector chooses 0.
- Static high-impact findings: EDU outer resume checks can bypass stage input validation; overlap initialization publishes resumable configuration before evidence is ready; cluster lifecycle writes/deletes a shared user-global Dask config; packaged scientific operations depend on external NVMe scripts; fusion relies on process-wide third-party patches; CI omits segment tests/lint. Additional bounded limitations and staged recommendations are in the report.
- Verification: all package Python sources parsed. Installed `lightsheet --help`, `lightsheet-stitch --help`, and `squisher-segment --help` exited 0. With `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/squisher-review-mpl` and multi Python/pytest, `-q -p no:cacheprovider lightsheet/tests/test_cli.py lightsheet/tests/test_artifacts.py segment/tests/test_data_discovery.py` passed 55 tests; `segment/tests/test_train_cli.py segment/tests/test_overlap_iou.py segment/tests/test_model_cache.py` passed 45; `segment/tests/test_distributed_segmentation.py segment/tests/test_distributed_postproc.py` passed 110. Total 210 selected tests passed; no full suite or clean-install CI run.
- Runtime caveat: installed Dask is 2025.10.0 but Distributed is 2026.6.0; lockfile pins Distributed 2025.10.0. Tests describe this existing multi environment, not a clean locked install. GPU/native numerical behavior, production scientific impact, interruption cases identified statically, and performance remain unverified.
- Decision: prioritize channel/model derivation, physical-grid, artifact preservation, and stage invalidation fixes before module extraction. All fixes are recommendations only; the user's request authorized review, not implementation.

## 2026-10-01 — Implement architecture-review repairs

- User authorized Sol workers to address the architectural review. Scope: prioritized findings 1–12 in `ARCHITECTURE_REVIEW.md`, with narrowly necessary ownership improvements and focused regressions; preserve pre-existing worktree changes, no commits or production runs.
- Work split: training/model artifacts; registered input geometry and initialization; measurement input identity; rechunk artifact preservation; EDU invalidation; external backend packaging; fusion backend adaptation; cluster lifecycle. Parent owns CI, TL/TR channel forwarding, integration, documentation, and final validation.
- Validation plan: reproduce each defect with focused failing tests first; run affected CPU suites in `multi`, use temporary outputs, and report GPU/native behavior that remains unverified. No source hashing introduced; no environment files changed.

### 2026-10-02 — Architecture repairs integrated and verified

- Completed the 12 prioritized repair areas with eight Sol workers, parent integration, and a separate read-only Sol migration review. `ARCHITECTURE_REVIEW.md` now contains the repair-status table, compatibility requirements, remaining structural limits, and validation details. Existing unrelated worktree changes were preserved; no commits or production runs.
- Scientific contracts: SAM channels preserved; plans bind checkpoint/plan/build/runtime metadata and validate at driver/worker/EDU; `squisher-segment build-trt` rebuilds existing checkpoints without retraining. Shared `squisher.ome_geometry` validates registered grids and propagates crop/projection geometry into labels. Regionprops validates known completion, root identity, and available physical grids. Rechunk preserves fusion/N4 sidecars and rejects unsupported rebasing.
- Runtime/maintenance: input-bound EDU reuse including cleaned materialized stages; recoverable blank overlap initialization; checked MVS adapter/restoration; packaged sweep/trial/contact-sheet backends with exact custom-path loading; selected TL/TR channel forwarded; scoped cluster configuration with independent CPU-cluster tests; segmentation CPU/lint CI coverage. Source-code hashing replaced with maintained algorithm revision and Cellpose version; run identity schema is 7. Updated segment/lightsheet READMEs and threshold-attestation documentation.
- Compatibility: old unbound TRT plans must be rebuilt with the new CLI; schema-6 segmentation runs and EDU stages without input identities are not automatically adopted; missing registered physical geometry is rejected. Existing editable lightsheet installation needs reinstall for the new backend executables. No production environment reinstall was performed. MVS still uses private interfaces/process patches behind checked adapter; native DLL/GPU numerical equivalence, crash-atomic rechunk replacement, and concurrent writers are not established.
- Parent verification used `/home/chaichontat/miniforge3/envs/multi/bin/pytest` with `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/squisher-fix-mpl -q -p no:cacheprovider`. `-m 'not gpu' segment/tests`: 294 passed, 12 deselected. `CUDA_VISIBLE_DEVICES=-1` with `-m 'not gpu' squisher/tests`: 125 passed, 2 skipped. Lightsheet broad CPU selection: 921 passed, 18 skipped, 1 deselected, 2 failed. Fixed the contact-sheet test's removed external-script expectation and ran `lightsheet/tests/test_cli.py lightsheet/tests/test_edu_workflow.py`: 85 passed. The other failure, existing `test_yx_projection.py::test_measure_constraint_fixes_z_for_yx_projection`, needs CUDA despite lacking a marker; isolated retry without the visibility override also failed `cudaErrorNoDevice`. No GPU computation completed.
- Initial combined lightsheet+squisher collection encountered duplicate `test_cli` module names; separate package invocations matched CI and resolved collection. Initial segment aggregate had one extraction fixture without now-required physical metadata; corrected the fixture's valid-input contract, then all 294 CPU tests passed. Driver-specific bare/stale plan regressions failed before integrating shared validation and passed in the aggregate suite afterward.
- Full configured Ruff and `git diff --check` passed. `uv lock --offline` first hit sandbox cache access and then lacked cached PyTorch registry metadata after scoped escalation; `uv lock` online resolved 182 packages, and `uv lock --check` passed. Runtime packages were not installed or upgraded by lock resolution.
- Packaging worker installed a wheel into `/tmp/squisher-backend-clean-env` using the existing dependency stack, verified imports came from wheel site-packages, forbade external-script loads, and exercised both backend console `--help` commands. Scientific-function AST comparison preserved sweep/trial numerical functions except loader routes; renderer changed loader/font ownership only. Independent Sol review reported no significant introduced regression in packaging, plan/schema routing, or geometry propagation.
- Test-generated Dask lock files were preserved under `.pytest_cache/architecture-dask-omjt6_e7/.chaichontat_distributed_cellpose`. An attempted rename into `/tmp/squisher-architecture-test-scratch-i71y9ojy` failed because `/tmp` is another filesystem; the successful same-filesystem relocation removed test scratch from the untracked source-file list without deleting it.

## 2026-10-02 — Simplify architecture-repair changes

- Goal: apply the requested code-simplifier skill to the current repair diff while preserving behavior and unrelated pre-existing changes. Inspection base is merge-base `a4b8910` with `origin/main`; mutation scope remains repair hunks added over dirty HEAD `75eb478`. No staged changes or configured upstream.
- Inventory: parent reviews production contracts and enclosing functions; bounded read-only Luna agents inventory segment tests, lightsheet tests, and added backend modules. Keep channel/geometry, artifact integrity, stage reuse, and resource cleanup contracts intact.
- Initial finding: channel-preserving training preparation still returns an always-zero trimming counter, leaving unreachable logging and dead caller bookkeeping. Validation will use focused CPU tests in `multi`, Ruff, and diff checks; GPU/native numerical equivalence remains unavailable.

### Simplification results

- Removed dead training trimming counts/logging and updated the private preparation return contract; retained channel contents, singleton squeezing, and its observable count/log. Consolidated overlap evidence metadata finalization after the create/open decision. Collapsed CPU-cluster config intermediates without changing evaluation timing or override precedence.
- Inlined the one-consumer materialized-fusion input builder into its owning deferred identity callback and the runtime-report temporary binding. Preserved the exact identity payload and durable upstream inputs used after intermediate cleanup.
- Replaced backend forwarding loaders with direct imports at the same lazy call sites, relying on Python's module cache; removed the moving-source default alias and updated both parsers. Removed repeated MVS `fusion.fuse` lookup during adapter loading while retaining the `zarr.create` capability check before all writes/patches. No numerical algorithm changes.
- Test inventory retained unique regression cases. Removed a dead recovery test loader, a trivial flipped-tile loader, and an unused fixture argument. Changed explicit `--packed` coverage to start from `packed=False`, so it now distinguishes enabling an override from preserving configured `True`. Updated backend-chain assertions to exercise actual parser defaults after removing the loaders. Pre-existing unsharp-test removal and unrelated user edits remain outside this task.
- Retained added modules: `squisher/ome_geometry.py` owns CPU-only physical-grid validation shared by segmentation and measurements; placing it in lightsheet NGFF would invert dependencies. `mvs_backend.py` owns third-party capability checks and ordered patch unwinding. `_backends/__init__.py` owns selection of installed executables in the requested interpreter environment, including lexical virtual-environment symlinks. `fused_fixed_trial.py` owns per-window registration, `fused_fixed_sweep.py` owns worker scheduling and summaries, and `fused_fixed_contact_sheet.py` owns rendering. These preserve separate execution/dependency contracts; none was folded or deleted. Kept `_ImageSource` as the canonical bundle of array, root identity, and physical geometry; kept atomic artifact publication and pre/post-copy validation because they protect distinct failure stages.
- Validation: `env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/squisher-fix-mpl /home/chaichontat/miniforge3/envs/multi/bin/pytest -q -p no:cacheprovider -m 'not gpu' segment/tests/test_train_cli.py segment/tests/test_overlap_iou.py segment/tests/test_gpu_cluster.py segment/tests/test_distributed_segmentation.py` — 152 passed. Same environment/runner on `lightsheet/tests/test_edu_workflow.py`, `test_fused_fixed_flipped_tiles.py`, `test_fused_fixed_method10_recovery.py`, `test_mvs_backend.py`, `test_packaged_fused_fixed_backends.py`, and `test_cli.py` — 157 passed. Focused Ruff on all changed Python files and `git diff --check` passed.
- Result: behavior-preserving simplification complete, with 309 focused passing tests covering training, resume identity, overlap recovery, cluster/patch lifecycle, geometry consumers, and backend dispatch. GPU/native numerical equivalence remains unverified. No commits, environment reinstall, file deletions, or production runs.


## 2026-10-05 — 229Ptprz1 right 514 intensity seam diagnosis

- Scope: diagnosis only; no code, approval-gate, production-data changes, or refitting. Inspect native and final QC, owning sampling/model/rendering code, and small native plane/ROI slices.
- Inputs: `/working/229Ptprz1/acquisitions/229Ptprz1-CR-514/fit-residual-514/` (correction, manifest, QC, owner map), corresponding CL fit, and `/working/229Ptprz1/final/qc/middle-z/middle-z-514.png`. Environment: multi Python; `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/229Ptprz1-placement-review/mpl`.
- Initial evidence: large rectangles visible before and after native correction; native QC uses last-valid-source ownership without blending. Saved Z=777 horizontal boundary median absolute log2 discrepancy is 0.400732 before and 0.295093 after, while global held-out full-model pair median is 0.040251. These metrics measure different populations and statistics.
- Next diagnostic: compare co-located native overlap pixels versus distance from source edge and intensity; check original-coordinate evaluation. No fitting functions will be called.
- Diagnostic limitation: importing the sampler in the read-only sandbox failed because Matplotlib requires a writable temporary cache. Request scoped execution for read-only data access plus its designated temporary cache.

### Direct overlap evidence and diagnosis

- Read-only diagnostic interpreter: `/home/chaichontat/miniforge3/envs/multi/bin/python`, invoked with `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/229Ptprz1-placement-review/mpl`. Temporary Matplotlib cache access required scoped sandbox execution; this resolved the initial import failure. No fitting functions were called.
- Reproducer inputs: load CR `fit-residual-514/correction.json`, its `fit.registration`, and `manifest.json`; construct `post_basic._fixed_grid(Path(fit.fixed_fused), fixed_z=777, stride=4)`. Select correction source indices 3,4,6,7 by path and call `residual_correction._sample_tiles(records, grid, channel=0, level=2, workers=2)`. Each source read is at most two Z slices of 480x480 pixels (460,800 voxels); no CZYX volume was read. The four pull matrices were identity with translation offsets, and both NGFF levels had positive scales and matching origins. Correct the samples with `_corrected_tiles`, shared plus source coefficients and gain times global scale.
- Against `basic-ch0.ome.tif`, `corrected-ch0.ome.tif`, and `owner-ch0.tif`, the saved owner pixels reproduced exactly (max absolute error zero before and after) for sources 003/004/006/007: 179883/179352/180725/180200 pixels respectively. Thus native QC has real source disagreement, although its last-valid-source compositor makes unfeathered cuts.
- Pair 003/006 at fixed Z=777, stride-4 QC YX ROI `[854:909,20:466]`: every co-located valid pixel exceeds the actual fit cutoff 942.070556640625. Median corrected A/B at incoming B edge-distance bins (L2 pixels) 0:1, 1:2, 2:4, 4:8, 8:16, 16:32, 32:48, 48:55 was 1.2706,1.2521,1.2386,1.2086,1.1727,1.0737,0.9670,0.8910. Whole-strip signed log2 median was 0.073899. Last bin excludes invalid source support.
- Pair 004/007, QC ROI `[853:907,443:889]`, also entirely above cutoff: corresponding corrected ratios were 1.2239,1.2070,1.1943,1.1735,1.1274,1.0446,0.9513,0.8833 (last bin 48:54). Whole-strip signed log2 median was 0.037622. Both pairs exhibit edge-distance-dependent disagreement, crossing unity inside the overlap. This is not explained by a single tile gain or by excluding dim pixels. Intensity-quartile summaries retain edge-to-interior differences; these do not identify a unique additive versus multiplicative physical model.
- Independent native L0 check: use the same base grid with shape YX `[220,128]`, stride=1, origin shifted by `[0,3416,800] * base.scale`; sample only 003/006 at levels 0 and 2 with `_sample_tiles`, retaining Z=777. This reads approximately two native Z slices by 220 by 128 pixels per source, not complete planes/volumes. Co-located L0 ROI was 218x128; all 27,904 pixels exceeded cutoff. Incoming-edge-distance bins in L0 pixels 0:4,4:16,16:32,32:64,64:128,128:192,192:220 gave raw ratios 1.4688,1.4432,1.4076,1.3476,1.2683,1.1306,1.0187 and corrected ratios 1.3491,1.3254,1.2921,1.2366,1.1633,1.0374,0.9353. Relative correction multipliers were 0.918397,0.918224,0.917950,0.917584,0.917269,0.917570,0.918352. Level-2 checks gave nearly identical ratios (first corrected 1.3431, last 0.9354). Thus pyramid sampling cannot explain this measured seam.
- Confirmed implementation mechanism: native overlap sampler smooths and samples all eligible overlaps (`residual_correction.py:183`), while model selection/validation score absolute *signed median per pair-plane* (`post_basic.py:802`) and the QC compositor changes source at a support edge (`post_basic.py:612`). Opposite-sign spatial errors inside a pair can therefore yield a small scalar score despite a visible edge jump. Z validation reuses that metric. Shared/gain/source penalties all selected 0.3; weaker candidates did not improve the recorded selection objective. `xy_degree=1` activates only first X/Y cosines and their Z modulation (`post_basic.py:849`, `deconv/residual_field.py:20`), excluding even XY modes capable of representing symmetric edge curvature. Selected field changes the relative multiplier almost uniformly over the checked overlap and leaves the measured edge dependence.
- Native full-model held-out median is 0.0402509 and p90 0.1316077 over 180 pair-plane entries (not 180 independent source pairs); shared-only p90 was 0.1262723. Native Z=777 boundary summary reports corrected horizontal median 0.295093 log2, approximately 22.7% ratio. Parent-provided materialization pixel verification was accepted as evidence only for its sampled stores, not rerun or generalized to whole stores. Materialization summary independently binds the expected correction SHA256 and 523 tiles; downstream base correction has 1302 identity gains and zero fields. Cross-view model applies the same multiplier to all R sources at a given physical coordinate (`crop_sharpness.py:299`), so cannot change their co-located ratio.
- Diagnosis: real native edge-dependent intensity mismatch remains after residual fitting and becomes rectangular when source ownership changes. Native QC accentuates cuts but does not invent the mismatch. Final fusion/feathering may affect their presentation; compression, subtraction, final chunking, and omitted correction are not sufficient primary explanations for the checked seams. No evidence supports a raw-coordinate bug in the inspected translation-only sources; full-store verification was not performed.
- Unresolved: whether the edge profile originates in acquisition shading, deconvolution edge behavior, or an additive component; whether the strongest regularization is inappropriate under a seam-local objective; exact source ownership of every final rectangular patch. A scalar additive model or higher XY degree has not been fitted/tested. A low-degree basis and robust aggregate objective demonstrably leave this spatial pattern, but this does not prove a particular alternative model will fix it.
- Smallest discriminating next check/remedy: keep the existing correction fixed and inspect the same pair in a second small Z ROI, stratifying paired differences and ratios by both source-edge distance and matched intensity. A stable proportional edge profile supports testing the existing degree-2 native XY model (even modes) with edge-local validation; a stable absolute offset requires an additive correction at the native model owner. One narrow preview with the ownership transition moved into mutually reliable overlap can separately assess seam-placement mitigation without altering production. No refit, code/gate/data edits, or production re-render performed; only this notebook entry was appended.


## 2026-10-09 — Authorized full Prod3-2 short-window registration and fusion

The user requested “start full run and fusion.” The unchanged least-squares
local3D method is used with axial window128/core80 and XY window528/core480.
The accepted production frame has1172 parent windows on94 native405 tiles;
34 other native tiles have no accepted production parents. A full native-grid
subdivision could not preserve this frame:9340 children lacked a containing
parent,500 had two, and uniquely mapped children left200 parent domains with
source gaps. The owner now has opt-in `--split-parent-windows`, generating
children relative to each accepted parent with explicit initializer identity.
The default grid and exact-one-parent lookup remain unchanged. Invalid lineage,
bounds, XY changes and child identity collisions fail explicitly. This changes
task geometry only, with no new registration estimator or fusion weighting.

Fresh exact561 inputs were rendered at seven supported spatial/depth strata
and six cortex coordinate edges. Root manually reviewed both sheets and chose
threshold1000 at masklevel2, persisted before fitting. Metadata preflight
found7032 children, six per parent, complete parent source-domain unions,
and no invalid parents or duplicate child identities. Expected output staging
uses `/home/chaichontat/nvme/prod3-2-405-local3d-ls-z128-core80/`; final full-grid
fusion uses `/warm/prod3-2-405-local3d-ls-z128-core80/fusion/`. Available space
was806GiB NVMe and4.9TiB on /warm. No sources were moved or deleted.

The new capability first failed its tail-domain test because the parent-relative
planner was absent. After implementation,22 focused child/backend/estimator
checks passed; a subsequent36-check run covered expanded child and existing
Method6 CLI/runner contracts. All modified package Python passed `multi` Ruff.
The isolated change from pre-task dirty files is saved at
`full_run/parent_task_grid_change.patch`; unrelated work was preserved.

The candidate chain will require all7032 fits evaluated, zero operational
errors, and accepted maps before materialization. Rejected transforms are never
fused or silently replaced with old parents. Rejection counts and per-parent
unresolved source coverage are saved. Accepted-only fusion is experimental;
it is not promoted to the review app, model or embeddings. The actual multi
interpreter and RTX4090 UUID are checked in the job itself. Launch status and
verified progress will be appended after the chained-run script is ready.

Evidence root: /home/chaichontat/fishtools2/thesis-internal/thesis/output/prod3_2_brdu_dapi_subtraction/registration_check/full_run


## Launch cache correction and restart

The first launch verified real fitting on four RTX4090 workers and produced
accepted short-window maps. Root's launch audit caught a missing cache field:
the sweep recorded `split_parent_windows`, but worker serialization and the
per-window fitter cache did not propagate it. This did not change the fitted
maps; it would cause completed fits to fail cache reuse. Root stopped that
job with its scoped tmux C-c; exit130 is recorded in `full_candidate.exit`
and the original `full_candidate.log` remains intact.

A focused worker-cache parity test reproduced the failure: the default grid
passed, while the parent-relative grid missed exactly the new field. The
owner now propagates the setting through the worker payload, task namespace
and fitter cache. All48 focused child/cache, Method6CLI/runner, packaged
backend and estimator CLI checks passed. All five modified package Python
files passed multi Ruff; the launch script passed bash syntax checking.
The isolated owner diff now includes the fitter cache change.

The same7032-task candidate was restarted on a fresh completion socket
`/tmp/prod3-2-full-shortwindow-v2.sock` with the same session name,
`codex-prod3-2-full-shortwindow`. Its log/markers use `full_candidate_v2.*`.
Existing generated fit records whose cache contract lacks the field will be
recomputed in the existing output folder; no source data or previous rescue
were changed. Full fusion remains queued behind all producer checks. The
new launch's worker and per-window cache audit is recorded separately in
`launch_verified.json` after real accepted fitting is observed.


## Full run launched and verified — 2026-10-09T04:44:48.475749+00:00

The background chain is running in tmux session `codex-prod3-2-full-shortwindow` on socket
`/tmp/prod3-2-full-shortwindow-v2.sock`. The actual job log verifies the multi interpreter and
RTX4090; GPU workers were traced through their parent process IDs to this
session's panePID515354. Early native fitting produced83 terminal
window records: {'rejected': 75, 'accepted': 8}. An accepted128×528×528 child records the new parent
grid cache identity and exact published parent initializer. This verifies real
fitting progress, rather than session existence. Materialization and full-grid
fusion are queued behind their completion checks; neither is reported done.

Log: `/home/chaichontat/fishtools2/thesis-internal/thesis/output/prod3_2_brdu_dapi_subtraction/registration_check/full_run/full_candidate_v2.log`. Expected final producer output:
`/warm/prod3-2-405-local3d-ls-z128-core80/fusion/fused.ch0.ome.zarr`. No live app/model/embedding source was
changed. Graceful stop: `tmux -S /tmp/prod3-2-full-shortwindow-v2.sock send-keys -t codex-prod3-2-full-shortwindow C-c`.
The worker's owning completion smoke check passed8/8 existing materialized
stores (six128Z children and two parents), with metadata/payload manifests,
correction fingerprints and domains verified without fluorescence pixel reads.
All full-run Python passed Ruff and the launcher passed bash syntax checking.
The worker also reported three pre-existing unrelated thesis import-check
violations; those files were outside this run's change scope.

## 2026-10-11 - Closed hemisphere shells from bilateral-uv-surface

- Added `lightsheet bilateral-uv-surface --closed-shell`. It runs marching cubes on the same smoothed, cubic-resampled signed distance as the wall graphs, clips and caps each half at the split plane with PyVista `clip_closed_surface`, and writes `shell.npz` (`vertices_xyz_um`, `faces`, `split_cap_faces`). It validates one closed outward manifold per half and wall-rim agreement within 0.01 µm, and rejects masks with occupied array-face voxels. Reflected image directions flip the marching-cubes winding before clipping. `pyvista>=0.46` is now a declared dependency (lock adds pyvista 0.49.1, vtk 9.7.1 and their dependencies).
- Tests: two new shell cases (identity and reflected direction) and an array-face rejection case in `lightsheet/tests/test_bilateral_cavity_y.py`; all 19 bilateral/uv-surface tests pass; Ruff passes.
- 230Tnc check: `--cavity-axis y --split-x-index 226 --smoothing-um 19.2 --closed-shell` reproduces the previous inline shell exactly (488,618 vertices, 977,232 faces, 1,558 cap faces, rim error 8.8e-5 µm), with byte-identical wall surfaces and charts. Consumer: `thesis/analysis/ch2/tnc-lightsheet-vz/` in thesis-internal.
