# Book, color and reliability update

## Implemented

- Book-only instance audit with enlarged upload-derived detail views. Multiple
  detected spines/covers can replace a group; non-book objects remain unchanged.
  Bounds, confidence, duplicate, coverage and total-object-count checks apply.
  Unresolved groups are retained rather than inventing a book count.
- Flat books with image evidence of stacking get separate support heights.
- Photo-color sampling now includes tables, side tables, seating and accessories.
  Tabletops exclude floor gaps; thin masks survive erosion; empty masks do not
  silently sample background. Accessories tint bodies but retain hardware/screens.
- Photographed sRGB is converted to linear material values, avoiding pale colors.
- Added saved upright-hardback and paperback assets; corrected the original
  hardback's excessive thickness. Library has 36 GLBs, plus the separate blanket
  and two external sample models: 39 saved reusable models in total.
- Request validation, safe job IDs, pre-launch job reservation against duplicate
  workers, atomic metadata writes and isolated state snapshots.
- Bounded audit transport retries, handling dropped connections, and preserving
  valid detail objects when another returned object fails validation.

## Validation and limits

The final full backend test run passed 96 tests. An earlier run hit two Blender
timeouts; the isolated rerun passed. Live detection checks encountered HTTP 429
and 503, so successful book separation on a fresh user photo is NOT established.
No extra provider requests were left running. Existing provider-failure telemetry
now also includes an analysis warning for unavailable detail verification.

`24f4eeba-9fd` is a saved-evidence color/geometry review, not a fresh detection or
proof of book separation. Its source job is recorded, its source GLB is preserved,
and the old optimizer score is invalidated. The diagnostic render uses Blender
lighting, not the browser lighting pipeline.

No supplied photo names, hashes, object coordinates or preset layouts are used
as reconstruction branches. The book audit cannot restore details the detector
omits when the provider is unavailable; those cases remain a limitation.
