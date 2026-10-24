# Bounded photo-alignment feedback

Runs by default after the initial GLB export. Set reconstruction option
`selfCorrection: false` to disable. Existing saved rooms are not rewritten.

1. Load exported object meshes and corresponding SAM masks with calibrated camera.
2. Rasterize convex silhouettes of individual mesh parts at 160-pixel width;
   approximate occlusion by part-depth ordering. Compare visible instance masks.
3. Try bounded pose and scale proposals (10-second search, at most 100 trials,
   35 cm displacement, 15 degrees yaw, 15 percent size change).
4. Restore support/wall constraints. Reject increases in measured physical
   penalties. Require mean IoU improvement, allowing at most .01 per-object IoU
   decrease for rasterization tolerance.
5. Re-export real geometry to a candidate directory and measure it again.
   Promote only a verified improvement; otherwise retain the original GLB.

`metrics.extra_details.self_correction` reports before/after IoU, proposals,
duration, final export verification and weak-object flags. ProcessingView shows
the summary. Alignment PNGs are diagnostic proposals, not photoreal renders.

## Limits

This is a first silhouette-feedback implementation, not full differentiable
rendering or photometric scene recovery. Convex part silhouettes and median-depth
ordering approximate concavity and occlusion. It does not optimize camera pose,
lighting, hidden geometry or mesh topology. Existing semantic verification runs
before geometry; weak matches are flagged for re-detection but do not yet trigger
another automatic post-render AI call. Objects missed by both detection passes
cannot be measured against masks that do not exist. Contact/collision checks do
not cover every possible physical relationship. A high IoU is not 3D accuracy.

Checks: `test_self_correction.py` covers improvement, immutability, absent
evidence, bounded trials and collisions. `check_self_correction.py` runs saved
evidence in a temporary directory and rechecks rebuilt geometry without changing
saved rooms. No reference-photo lookup or image-specific placement rules are used.
