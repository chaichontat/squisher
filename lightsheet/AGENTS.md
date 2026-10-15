# Squisher Lightsheet Agent Guide

## Registration Thresholds

- For every image-registration run, choose the signal threshold by manually inspecting a threshold-review render generated from that run's exact processed inputs and registration channel. Persist the reviewed value with the review artifact before launching registration; never infer or reuse it from percentiles, defaults, filenames, or another dataset.
- In the packaged registration workflow, retain disconnected tiles by default only when every tile outside the anchor-connected component is an acquisition-edge tile in the complete screened face-adjacency graph. Continue to fail on interior disconnections unless the user explicitly supplies `--allow-disconnected`; preserve `--exclude-disconnected` as an explicit request to omit all tiles outside the anchor component.
- For CL/CR datasets, use the `lr_z_endview_flip_xz` placement mode and retain the CR negative Z/X orientation through joined positions. Fuse each side internally with `content-preibisch-coarse`; reserve seam-weighted fusion for the final between-dataset CL/CR combination.

## HCR QC Approval

- Before approving a future HCR QC gate, consult Astra on its actual images and supporting metrics; reuse the dataset’s persistent Astra conversation and record its recommendation in the notebook.
