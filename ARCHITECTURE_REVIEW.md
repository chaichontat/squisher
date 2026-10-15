# Squisher segmentation and lightsheet architecture review

Date: 2026-10-01. Reviewed the current working tree at base commit `75eb478`, including pre-existing uncommitted changes. Three independent Astra reviewers covered segmentation execution, lightsheet orchestration, and cross-package data contracts; the coordinating review covered packaging, CI, dependency structure, and consolidation. This is a review, not an implementation pass.

## Assessment

The repository has substantial scientific functionality and several carefully implemented persistence mechanisms. Its principal maintainability problem is inconsistent ownership of scientific and operational contracts: channel selection, physical coordinates, model derivatives, artifact publication, and resume identity are enforced differently across adjacent stages. Some of those inconsistencies already produce demonstrably incorrect behavior on small inputs.

Preserve the existing algorithms and operational commands. First repair the contracts that can silently change scientific results or reuse stale outputs. Then consolidate their owners and isolate the external backends. Splitting large modules without doing that would redistribute the same coupling.

## Architecture as implemented

```text
lightsheet / lightsheet-stitch
  -> several workflow owners: workflow, registration_workflow, edu_workflow
  -> positions / registration / recovery / QC / fusion wrappers
  -> active _legacy stitching engine + multiview-stitcher runtime patches
  -> native CUDA adapter and external NVMe Python scripts
  -> completed channel-separated OME-Zarrs + JSON artifacts
                                      |
                    registered-source manifest
                                      v
squisher-segment CLI
  -> extraction / training / N4 / region measurements
  -> distributed segmentation -> Dask GPU workers -> Cellpose / TensorRT
  -> overlap evidence / stitching -> label Zarr -> postprocessing

Shared squisher package: JPEG-XR codec, GPU retry, and compression application.
Lightsheet also imports deconvolution code; EDU invokes/imports segmentation.
```

Source inventory, excluding tests and standalone lightsheet scripts:

| Package | Python files | Source lines | Test files | Test functions |
| --- | ---: | ---: | ---: | ---: |
| `segment/squisher_segment` | 26 | 15,404 | 11 | 227 |
| `lightsheet/src/squisher_lightsheet` | 79 | 63,530 | 77 | 790 |

These are AST/file counts, not coverage measurements. The active legacy stitching module has 10,809 lines and 13 direct importing source modules. Deferred-import cycles connect it to registration, seam refinement, channel optimization, and fusion components; these are dependency cycles, not observed import failures. Segmentation likewise has a deferred cycle between extraction and distributed inference. The consequence is that nominal module boundaries do not consistently isolate a change.

## Prioritized findings

Priority means repair order: **P1** concerns silent scientific mismatch, stale-result reuse, or broken artifact contracts; **P2** concerns reproducibility, regression protection, and costly structural risks. Evidence is explicitly identified as reproduced behavior or static inspection.

### 1. P1 — Training preparation discards the third inference channel

**Reproduced.** [`train.py:218`](segment/squisher_segment/segment/train.py#L218) trims every channel-first input with more than two channels to `image[:2]`, independently of the backend. A CPU helper call with a `(3, 4, 5)` array containing channel values `[405, 561, 638]` returned `(2, 4, 5)` with `[405, 561]` and `trimmed=1`.

The documented SAM inference profile uses three channels and the iterative training contract requires preserving their number and order ([README](segment/README.md#iterative-training-loop)). The selected training tests pass because an existing test expects truncation. This is a contract mismatch that test success currently reinforces; the review did not measure its effect on a trained model.

**Owner and repair:** Make channel order/count an explicit model/training contract, shared by extraction, preparation, and inference. Apply any backend-specific selection explicitly; reject incompatible data instead of silently discarding it. Validate the three-channel SAM case and a deliberately incompatible input.

### 2. P1 — TensorRT plans can outlive the checkpoints that produced them

**Reproduced at the artifact boundary, without GPU inference.** [`train.py:452`](segment/squisher_segment/segment/train.py#L452) cleans up existing device plans only when TensorRT generation runs. Retraining the same model name with `--skip-trt` can therefore replace the checkpoint while leaving the old plan. [`model_cache.py:72`](segment/squisher_segment/segmentation/distributed/model_cache.py#L72) selects the device plan for inference; the artifact identity checks accept the checkpoint and plan independently.

A bounded probe stubbed the training backend, wrote a new checkpoint, retained the old plan, and confirmed that the real plan-identity helper accepted it. Recording both checksums detects subsequent changes but does not prove the plan was built from the named checkpoint.

**Owner and repair:** Treat a plan as a derived model artifact. Publish its generating checkpoint identity and build/runtime settings alongside it; validate that relationship before loading. Retraining must invalidate the derivative even when rebuilding is intentionally skipped. Move plan lifecycle ownership out of workflow-specific checks.

### 3. P1 — Registered-source stacking does not validate a common physical grid

**Reproduced.** The manifest loader in [`distributed_segmentation.py:1269`](segment/squisher_segment/segmentation/distributed/distributed_segmentation.py#L1269) checks storage shape, dtype, completion, and metadata identity, but does not compare NGFF physical scales/translations across channels. It applies the same index ROI and stacks the pixels.

Two completed stores with identical array shapes and a **1,000 µm difference in X origin** were accepted and returned as one `(2, 3, 4, 2)` input. Each source having an authentic metadata hash does not establish that their coordinates agree.

**Owner and repair:** Validate axes, units, physical scale, origin, and selected ROI at the registered-input boundary. This interface promises already registered inputs: reject differing grids rather than introducing implicit resampling. Share this coordinate descriptor with the manifest producer and measurement consumers.

### 4. P1 — Rechunking publishes dangling provenance references

**Reproduced.** [`ome_zarr_rechunk.py:89`](lightsheet/src/squisher_lightsheet/ome_zarr_rechunk.py#L89) copies root attributes while reconstructing the pyramid arrays; publication at [`:314`](lightsheet/src/squisher_lightsheet/ome_zarr_rechunk.py#L314) marks the result complete. It does not carry the referenced fusion provenance bundle or N4 field artifact into the output.

A CPU-only `start_level=0` repro produced a completed store retaining fusion-manifest and N4-field references with neither referenced file present. The N4 sidecar validator subsequently raised `FileNotFoundError`. Using level zero isolates this from any question about the semantics of dropping pyramid levels.

**Owner and repair:** Define the artifact bundle that accompanies an OME-Zarr rewrite. Copy and validate supported referenced members, update references when necessary, and reject unsupported extensions before publishing. N4's explicit sidecar validation offers an existing pattern to reuse.

### 5. P1 — EDU's outer resume logic can bypass input validation

**Static mechanism confirmed; stale full-volume output was not generated.** [`StageRunner.command`](lightsheet/src/squisher_lightsheet/edu_workflow.py#L290) identifies a stage from command strings and explicit environment overrides, then reuses it when output paths exist. Replacing an input at the same path does not change that identity.

The registration invocation at [`edu_workflow.py:1225`](lightsheet/src/squisher_lightsheet/edu_workflow.py#L1225) has no validator, so outer reuse can prevent the registration owner's fingerprint checks from running. The fusion validator at [`:697`](lightsheet/src/squisher_lightsheet/edu_workflow.py#L697) checks completion. The materialized-channel branch at [`:1444`](lightsheet/src/squisher_lightsheet/edu_workflow.py#L1444) and [`:1501`](lightsheet/src/squisher_lightsheet/edu_workflow.py#L1501) can reuse a complete fusion outside the stage runner even after upstream registration/recovery/correction artifacts change.

**Owner and repair:** Let each stage owner validate the input identities, semantic settings, and completed outputs that justify reuse. Outer orchestration must consume that decision rather than substituting path existence or a completion flag. Add changed-input-at-same-path cases, including recovery-to-fusion, before refactoring the runner.

### 6. P2 — Packaged scientific commands still execute external Python implementations

**Static evidence.** [`fused_fixed.py:16`](lightsheet/src/squisher_lightsheet/fused_fixed.py#L16) defaults to a sweep script under `/home/chaichontat/nvme/lightsheet/scripts`; [`:483`](lightsheet/src/squisher_lightsheet/fused_fixed.py#L483) launches that implementation. Recovery imports its parser and runner ([`fused_fixed_recovery.py:1291`](lightsheet/src/squisher_lightsheet/fused_fixed_recovery.py#L1291)). Other QC/backend paths also use external scripts. A package revision therefore does not completely identify the executed scientific implementation.

Tests load the same absolute external trial path without a missing-file guard: [`test_fused_fixed_method10_recovery.py:23`](lightsheet/tests/test_fused_fixed_method10_recovery.py#L23) and [`test_fused_fixed_flipped_tiles.py:8`](lightsheet/tests/test_fused_fixed_flipped_tiles.py#L8). A checkout without those files cannot run those tests as written; remote CI was not run in this review.

The recovery loader also imports by basename after modifying `sys.path`, so Python's module cache can retain the first implementation if another path with the same basename is selected in the same process.

**Owner and repair:** Package the scientific implementations and expose explicit fit/recovery interfaces. Keep subprocess isolation if it is operationally useful. Treat the native library as a separately configured and versioned dependency, and test backend availability and compatibility in a clean installation.

### 7. P2 — The active fusion engine relies on process-wide third-party patches

**Architectural risk, not observed numerical corruption.** The public fusion wrapper launches the legacy engine ([`fusion.py:196`](lightsheet/src/squisher_lightsheet/fusion.py#L196)). Its production path coordinates patches to source serialization, weight normalization, Zarr creation, and pyramid writing ([`stitch_20x_tl_multiview.py:10334`](lightsheet/src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py#L10334)). Worker computation also calls MVS's private `_fuse_block_zarr_backed` ([`:7613`](lightsheet/src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py#L7613)).

Correctness depends on upstream internal APIs, patch ordering, restoration, and process isolation. The subprocess wrapper and single-active-session guard mitigate concurrent invocation risk. However, the resume patch can return without installing itself when the expected MVS module shape is absent ([`:7159`](lightsheet/src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py#L7159)), even while the caller enables chunk skipping.

**Owner and repair:** First isolate an explicit MVS adapter that checks supported capabilities and fails clearly when they differ. Test its source reads, block writes, interruption behavior, and restoration. Then move storage and block execution behind that adapter; a wholesale fusion rewrite is unnecessary.

### 8. P2 — `run-tltr --channel` does not reach registration

**Reproduced at the orchestration boundary.** [`workflow.py:181`](lightsheet/src/squisher_lightsheet/workflow.py#L181) invokes `register_tiles` without the requested channel, although rough alignment, QC, and fusion receive it. The existing registration wrapper supports `channels` ([`registration.py:62`](lightsheet/src/squisher_lightsheet/registration.py#L62)); the backend defaults to registration channel zero ([`stitch_20x_tl_multiview.py:9315`](lightsheet/src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py#L9315)). A call to the actual workflow with stage stubs and `channel=1` observed rough/QC channel `1`, fusion channels `[1]`, and no registration channel argument. The actual backend selector returned `0` for track `(0, 1)` with its default registration index. No images or GPU registration were needed to establish the argument loss.

This is a concrete consequence of carrying settings through several separate workflow/wrapper/parser layers. The current workflow test uses channel zero ([`test_workflow.py:25`](lightsheet/tests/test_workflow.py#L25)). Forward the setting through the existing owner and test a nonzero channel across all stages.

### 9. P2 — CI does not protect the segmentation package

**Verified configuration gap.** [`.github/workflows/ci.yml:32`](.github/workflows/ci.yml#L32) lints only `squisher` and `lightsheet`; its test steps likewise omit `segment`. The repository contains 227 segmentation test functions, including tests for resume, label stitching, training, and artifact contracts, but this workflow never invokes them.

**Owner and repair:** Add segmentation's CPU contract suite to required CI and define explicit GPU/native integration jobs or separately reported checks. A GPU marker alone is not an execution policy. Use a clean package installation to catch accidental workspace/external-script dependencies. Keep CLI forwarding tests, but complement them with the small producer-to-consumer contract cases identified above.

### 10. P2 — Region measurements ignore OME-level completion and provenance

**Reproduced acceptance and identity gaps; stale measurement mixing was not exercised.** [`regionprops.py:416`](segment/squisher_segment/segment/regionprops.py#L416) selects an OME dataset without checking completion, axes, or physical geometry. Its identity at [`:398`](segment/squisher_segment/segment/regionprops.py#L398) covers the selected array metadata and label identity attributes, omitting the OME root; this identity is also used for intensity sources ([`:615`](segment/squisher_segment/segment/regionprops.py#L615)).

A synthetic source explicitly marked incomplete opened successfully. Changing its root N4 provenance and completion state did not change the measurement input identity. This allows partial inputs and leaves a gap in resume invalidation; the documented requirement that source chunks remain immutable still applies.

**Owner and repair:** Retain the grouped source descriptor when selecting its array. Validate known producer completion markers, include root producer identity in reuse checks, and compare available physical coordinate metadata before combining labels and intensities.

### 11. P2 — Overlap setup has an unrecoverable initialization interval

**Static state-transition finding; process interruption was not injected.** [`distributed_segmentation.py:2314`](segment/squisher_segment/segmentation/distributed/distributed_segmentation.py#L2314) publishes the run configuration before initializing overlap evidence at [`:2322`](segment/squisher_segment/segmentation/distributed/distributed_segmentation.py#L2322). Resume uses configuration existence to select the existing-run path, which requires complete evidence metadata ([`:2267`](segment/squisher_segment/segmentation/distributed/distributed_segmentation.py#L2267)). `--overwrite` preserves matching configuration and temporary state ([`:2905`](segment/squisher_segment/segmentation/distributed/distributed_segmentation.py#L2905)).

An interruption after publishing configuration but before completing evidence initialization therefore leaves state that both ordinary resume and overwrite retain but cannot use.

**Owner and repair:** Represent initializing versus ready state explicitly. Publish readiness only after empty evidence initialization, or permit safe reconstruction when no blocks have checkpointed. Test failures at both sides of that transition; do not weaken identity checks for already computed blocks.

### 12. P2 — Cluster configuration is written into a shared user-global file

**Static lifecycle finding; the user's configuration was not modified.** [`gpu_cluster.py:33`](segment/squisher_segment/segmentation/distributed/gpu_cluster.py#L33) writes effective Dask configuration to a fixed `~/.config/dask/distributed_cellpose_dask_config.yaml`. Shutdown deletes that file rather than restoring any previous contents ([`:63`](segment/squisher_segment/segmentation/distributed/gpu_cluster.py#L63), [`:254`](segment/squisher_segment/segmentation/distributed/gpu_cluster.py#L254)). The write precedes device validation ([`:138`](segment/squisher_segment/segmentation/distributed/gpu_cluster.py#L138)), and multiple clusters share the same path. Shutdown also changes global logger levels without restoring them ([`:228`](segment/squisher_segment/segmentation/distributed/gpu_cluster.py#L228)).

**Owner and repair:** Pass configuration explicitly to the scheduler/workers or scope it to a per-run lifecycle. Preserve process logging state. Verify successful shutdown, startup failure, and two independent cluster instances without touching a real home-directory configuration.

## Additional bounded observations

These are secondary design limitations or static findings, not demonstrated production failures.

| Area | Evidence and consequence | Narrow next step |
| --- | --- | --- |
| Label portability | Final label attributes contain run/schema keys, while source/ROI identity resides in a sibling `.done` file ([`distributed_segmentation.py:2582`](segment/squisher_segment/segmentation/distributed/distributed_segmentation.py#L2582), [`:804`](segment/squisher_segment/segmentation/distributed/distributed_segmentation.py#L804)). Copying only the Zarr loses spatial provenance. | Embed the source grid, crop-origin adjustment, and projection descriptor in labels and preserve them through postprocessing. |
| Replacement semantics | Rechunk removes an existing destination before copying ([`ome_zarr_rechunk.py:82`](lightsheet/src/squisher_lightsheet/ome_zarr_rechunk.py#L82)); fused subtraction does similarly ([`fused_subtraction.py:287`](lightsheet/src/squisher_lightsheet/fused_subtraction.py#L287)). A failure can lose the last complete output. | Reuse staged replacement where preservation of the prior result is required, following N4's established policy. |
| Review-artifact provenance | `WORKFLOW.md` promises a validated reviewed TIFF, but the registration interface accepts a numeric threshold and writes a `human_reviewed_threshold` record ([`stitch_cli.py:72`](lightsheet/src/squisher_lightsheet/stitch_cli.py#L72), [`registration_workflow.py:572`](lightsheet/src/squisher_lightsheet/registration_workflow.py#L572)). | State whether the number is caller attestation; if artifact validation is promised, persist the reviewed-render identity and its input association. Human inspection remains procedural. |
| Training option precedence | The CLI's default `packed=False` unconditionally replaces the model configuration ([`cli.py:20`](segment/squisher_segment/cli.py#L20), [`:45`](segment/squisher_segment/cli.py#L45)). | Distinguish absent override from explicit enable/disable. |
| Concurrent run ownership | Default segmentation outputs, caches, temporary directories, and staging names are shared by inputs in one directory ([`distributed_segmentation.py:2697`](segment/squisher_segment/segmentation/distributed/distributed_segmentation.py#L2697), [`:2212`](segment/squisher_segment/segmentation/distributed/distributed_segmentation.py#L2212), [`:2506`](segment/squisher_segment/segmentation/distributed/distributed_segmentation.py#L2506)). Sequential identity checks do not provide exclusion. | Keep same-workspace concurrency unsupported until workspace selection and exclusive ownership are explicit. |
| Postprocessing recovery and seams | Postprocessing uses a temporary directory cleaned on exception ([`distributed_postproc.py:990`](segment/squisher_segment/segmentation/distributed/distributed_postproc.py#L990)); chunk-local donation is approximate and reconstruction uses face contact ([`:48`](segment/squisher_segment/segmentation/distributed/distributed_postproc.py#L48), [`:258`](segment/squisher_segment/segmentation/distributed/distributed_postproc.py#L258)). | Document the weaker resume/approximation contract and characterize tiled versus whole-volume results before changing algorithms. |
| Resume sensitivity to source edits | Existing source fingerprints include the orchestration/CLI-bearing inference file ([`model_artifacts.py:54`](segment/squisher_segment/segment/model_artifacts.py#L54)); exact identity checks invalidate resume after nonsemantic edits. This existing mechanism also conflicts with the supplied operating rule against hashing source. | Use an explicit algorithm/release identity with documented compatibility decisions; retain fail-closed numerical compatibility checks. |

## Structural improvements after the correctness repairs

1. **Share narrowly defined contracts.** A registered physical grid, a model-plus-derived-plan identity, and an OME artifact bundle each deserve one explicit owner. Reuse existing codec and publication utilities; do not create a generic workflow framework first.
2. **Keep CLIs as adapters.** Typed requests should pass to stage operations without serializing settings into CLI strings and parsing them back. Recovery currently crosses private helper APIs and `SimpleNamespace`/parser-shaped objects ([`complete_recovery.py:632`](lightsheet/src/squisher_lightsheet/complete_recovery.py#L632), [`fused_fixed_recovery.py:1299`](lightsheet/src/squisher_lightsheet/fused_fixed_recovery.py#L1299)).
3. **Extract around ownership, not size.** The distributed inference owner mixes planning, identity, evidence initialization, execution, checkpoints, and finalization. The active legacy engine mixes source geometry, scheduling, backend adaptation, and publication. Extract a responsibility when its contract and characterization tests are clear; retain public command and output compatibility.
4. **Make workflow dependencies explicit.** EDU imports segmentation and TensorRT-build helpers ([`edu_workflow.py:894`](lightsheet/src/squisher_lightsheet/edu_workflow.py#L894)), but `lightsheet/pyproject.toml` does not declare segmentation. A workflow installation/profile and an early dependency preflight would express this optional operational requirement without forcing every lightsheet user to install the GPU training stack.
5. **Distinguish portable guidance from run records.** Preserve acquisition records, but give contributors one current architectural map, supported runtime/dependency profile, and CPU/GPU verification commands. Currently many operational assumptions live in long workflow documents and machine-specific paths.

## Strengths to preserve

- Segmentation binds resume state to model, input, evaluation, and stitching settings, and separates disjoint core writes from overlap evidence. Overlap-IoU stitching changes identity without intentionally changing foreground support.
- Registration has input fingerprints, settings checks, and explicit connectivity policy ([`registration_workflow.py:183`](lightsheet/src/squisher_lightsheet/registration_workflow.py#L183)).
- Complete recovery validates artifacts and publishes through staging; N4 validates geometry and sidecars and uses staged atomic publication. These are useful precedents for weaker owners.
- Scientific stages are separately invocable and their intermediate artifacts are inspectable. This supports bounded reruns and diagnosis.
- The tests include meaningful geometry, overlap, persistence, corruption, and reduction cases. The main gaps are inconsistent execution in CI and missing cross-stage invalidation/contract cases, not a lack of test files.

## Validation and limits

- Parsed every package source file in the inventory successfully and inspected internal imports; no source changes were made.
- Installed `multi` entry points `lightsheet --help`, `lightsheet-stitch --help`, and `squisher-segment --help` all exited zero.
- Coordinating check: `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/squisher-review-mpl /home/chaichontat/miniforge3/envs/multi/bin/pytest -q -p no:cacheprovider lightsheet/tests/test_cli.py lightsheet/tests/test_artifacts.py segment/tests/test_data_discovery.py` — **55 passed**.
- Segmentation reviewer: `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/squisher-review-mpl /home/chaichontat/miniforge3/envs/multi/bin/python -m pytest -q -p no:cacheprovider segment/tests/test_train_cli.py segment/tests/test_overlap_iou.py segment/tests/test_model_cache.py` — **45 passed**.
- Segmentation reviewer: same interpreter/environment with `-m pytest -q -p no:cacheprovider segment/tests/test_distributed_segmentation.py segment/tests/test_distributed_postproc.py` — **110 passed**. Total across these three selected test runs: **210 passed**.
- Bounded CPU probes reproduced channel truncation, acceptance of an old plan after checkpoint replacement, incompatible-grid stacking, missing sidecars after rechunking, regionprops' completion/identity omissions, and loss of the nonzero registration channel. These establish the named contracts; they do not measure production numerical impact.
- Runtime snapshot: Dask `2025.10.0`, Distributed `2026.6.0`, Zarr `3.2.1`, Cellpose `4.0.8.dev47+gcd3eaf139.d20260826`. The lockfile pins Distributed `2025.10.0`, so the inspected `multi` environment is not an exact locked installation. No scheduler failure is inferred from that difference.
- No full suite, clean-install CI run, production data processing, GPU inference, native registration, or numerical-equivalence benchmark was performed. All implementation recommendations remain proposed.

## Repair status — 2026-10-02

The user authorized implementation after the review. Eight Sol workers addressed
separate owners; the parent integrated the changes and another Sol reviewer
checked the backend, model-identity, and geometry migrations. The findings and
line references above describe the original review snapshot.

| Finding | Implemented repair |
| --- | --- |
| 1. Training channels | Preserve SAM's one-to-three channels; reject excess channels; retain UNet's explicit selection. Absent `--packed` preserves configuration. |
| 2. Derived plans | Bind each plan to checkpoint, plan bytes, profile, device and runtime versions in `.plan.json`; validate on driver, worker and EDU reuse. Add `squisher-segment build-trt` for rebuilding without retraining. |
| 3. Registered grids | Shared CPU geometry reader validates matching physical grids; crop-origin and projection provenance reach final labels and survive existing postprocessing metadata copying. |
| 4. Rechunk bundles | Validate/copy fusion and N4 sidecars, reject unsupported members and provenance rebasing, preserve prior output during ordinary copy/publication failures. |
| 5. EDU reuse | Bind stages and completed materialized fusions to durable upstream identities; reject unbound or changed markers, including after intermediate cleanup. |
| 6. External Python backends | Package sweep, trial and contact-sheet implementations; default to installed console entrypoints; custom loaders use exact paths. Native shared libraries remain external configured dependencies. |
| 7. MVS adapter | Check required capabilities/signatures before patching, centralize patch ordering, reject unsupported resume targets, restore globals before fallible cleanup. Private APIs and process-wide patches remain isolated behind this checked adapter. |
| 8. TL/TR channel | Forward the selected channel to registration; cover zero and nonzero channels. |
| 9. Segmentation CI | Add segmentation CPU tests and lint to the workflow, declare development dependencies, refresh and verify the lockfile. |
| 10. Measurements | Validate known completion markers, retain root provenance in resume identity, compare physical grids when both inputs provide them. |
| 11. Initialization | Recover only blank, uncheckpointed overlap initialization; reject existing labels, evidence, markers or conflicting identities. |
| 12. Cluster lifecycle | Remove user-global config files and logger mutations; scope startup configuration, avoid default-client mutation, clean up failed startup/shutdown, verify independent CPU clusters. |

Source-file hashing was also replaced by an explicit maintained algorithm
revision plus installed Cellpose version, with run identity schema 7. The
threshold-review documentation now accurately distinguishes caller attestation
from technical validation of a render association.

### Compatibility and remaining limits

- Existing plans need verified sidecars. Use `squisher-segment build-trt MODEL_PATH`
  on the intended GPU to rebuild; stale or unverified plans are not adopted.
- Schema-6 segmentation state and EDU markers without input identities are not
  silently migrated. Preserve historical runs and use separate output directories.
- Registered source metadata must explicitly contain physical axes/units/scale;
  existing manifests themselves require no new fields.
- Rechunking fusion/N4 bundles requires level zero and unchanged numeric dataset
  paths. Replacement rollback is not crash-atomic or safe for concurrent writers.
- Existing editable lightsheet installations must be reinstalled to create
  `lightsheet-fused-fixed-sweep` and `lightsheet-fused-fixed-contact-sheet`.
  Installed-wheel smoke checks exercised both entrypoints in a temporary venv
  using the existing dependency stack; the production Conda environment was not
  reinstalled.
- Broad module extraction, a new postprocessing algorithm, same-workspace
  concurrent segmentation, and replacing all MVS patches remain outside this
  bounded repair. No production data or native/GPU numerical results were changed
  or validated.

### Implementation validation

- Segmentation CPU suite: **294 passed, 12 GPU tests deselected**.
- Shared `squisher` suite: **125 passed, 2 skipped**.
- Broad lightsheet CPU selection initially returned **921 passed, 18 skipped,
  1 deselected, 2 failed**. The contact-sheet test still expected the removed
  external-script constant; its fixture now follows the installed-entrypoint
  contract. The complete CLI/EDU subset then passed **85 tests**. The remaining
  existing projection test invokes CUDA despite lacking a GPU marker; running it
  separately also failed with `cudaErrorNoDevice`. It was not disabled or rewritten
  to hide the unavailable GPU capability.
- Full configured Ruff check, `git diff --check`, and `uv lock --check` passed.
  Offline lock resolution initially lacked PyTorch registry metadata; online
  resolution completed without installing runtime packages.
- Installed-wheel smoke and scientific-function AST comparison validated backend
  packaging; the latter found only loader changes for sweep/trial and loader/font
  changes for rendering. This is not a numerical-equivalence benchmark.
- An independent Sol integration review reported no significant actionable
  regressions in backend packaging, model provenance, or geometry propagation.
