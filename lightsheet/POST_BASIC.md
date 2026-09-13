# Residual Calibration Acceptance

- Before accepting a residual fit, inspect tile interiors and boundaries in shared-scale deconvolved/corrected images at low, middle, and high sampled Z, together with the applied after/before multiplier map and camera-X/Y field profiles. Assess seam improvement alongside these views; a lower seam score and valid artifacts alone are insufficient.
- If the correction introduces or worsens tile-periodic bands, bright-edge/dark-center waves, or loss of within-tile contrast, reject that candidate and continue the correction task; do not stop at diagnosis. Compare lower spatial order and stronger regularization through the packaged CLI, refit, and repeat the same checks. Accept only a candidate that passes; otherwise report the concrete blocker and unsuccessful comparisons.
- Preserve rejected runs and record the selected parameters, seam scores, and visual comparisons in the notebook.
