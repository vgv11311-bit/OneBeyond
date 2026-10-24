# Photo-detail reconstruction fixes

## General implementation

- Upright pillows use an orientation-preserving rotation. Exported cushion
  meshes have outward winding, including when the browser culls back faces.
- Pillow placement clears the actual headboard thickness.
- Wall shelves retain detected depth and slab thickness. The builder no longer
  quarters the thickness or invents brackets extending outside the fitted bounds.
- Shelf support matching compares the supported item's bottom to the slab top;
  a thin slab does not invalidate the association with a tall vase or book.
- Verification receives bounded, labelled detail crops from the current upload.
  All returned boxes must remain in original-photo coordinates.
- A separate missing-detail audit can add visible pillows, coverings, books,
  vessels and plants without replacing the accepted furniture layout. It
  validates additions, suppresses duplicates and permits an empty result.
- Chair association can use unambiguous nearby image evidence when an initial
  depth estimate incorrectly separated it from a wall-attached desk.
- Cloth hems use the inferred hanging extent, with a smooth transition at the
  mattress edge rather than a fixed horizontal-overhang-to-drop multiplier.
- Broad, front-facing coverings that visibly cross the projected mattress foot
  retain coverage of that edge. Narrow throws are not moved by this check.

No filename, image hash, supplied-photo coordinates, or reference-room layout is
used to select a reconstruction template. Reference uploads are test inputs only.

## Verification

`../.venv-moge/Scripts/python.exe -m unittest discover -p 'test_*.py'`

83 tests passed, including exported mesh winding/volume, shelf bounds, support
placement, audit deduplication, cloth drop and wrong-depth chair association.

Two independent fresh uploads completed during verification:

- `eb98d3d2-e29`: desk-room photo; 10 detected objects, right-wall desk,
  room-side chair facing the desk, primary covering and separate throw.
- `aff7d510-71a`: shelf-room photo; 21 detected objects versus 11 in the initial
  comparison run, including both pillows and supported small objects.
- `6ced9888-41c`: detail-only replay of the latter's detections to verify the
  final mattress-foot coverage fix. This is not a fresh AI analysis, and its
  original full-scene optimizer score is deliberately invalidated.

Diagnostic renders are stored as `geometry-review.png` in each job directory.
These use Blender diagnostic lighting, not the browser's lighting pipeline.
Existing saved GLBs are not silently overwritten; reconstruct to use new geometry.

## Remaining limits

Detection and metric dimensions remain estimates from one image. Detail crops
do not recover information absent from the photograph. The geometry tests and
headless renders do not verify browser-specific flicker or live WASD behavior.
