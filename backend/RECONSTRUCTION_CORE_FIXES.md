# Reconstruction core fixes — 2026-10-04

## Exhausted-provider fallback follow-up

The later `a531bda3-f41` attempt exhausted Groq's daily allowance and received
invalid JSON from Gemini while checking positions. The Gemini adapter had copied
Groq's small visible-output cap into Gemini's total generation allowance. It now
uses a separate 8192-token allowance, checks MAX_TOKENS explicitly, and reports
invalid alternate responses instead of disguising them as missing object evidence.
This does not enlarge Groq reservations or bypass any reconstruction gate.

Independent live Gemini checks passed both position and bedding validation on
the five-meter upload and on the different six-meter upload. These focused runs
are not proof of completed export or visual fidelity. A stale localization marker
can no longer certify an object omitted from a new response.

End-to-end follow-up: the five-meter upload completed analysis, SAM 2, MoGe,
Blender GLB validation and photo-alignment checking in 293.02 seconds, with no
error. QA output: `storage/spaces/qa-ztxjxphf`; no Recent metadata was created.
The first-pass room inventory was reused, while position/bedding/fixture checks
made live provider calls. This was not a completely uncached primary-generation
run. Uncertain visual geometry remains flagged; completion is not exact 3D truth.
The final full regression suite passed 356 tests.

## Aspect-ratio regression follow-up

The later five-meter export had vertically compressed detection boxes: the
chair box covered wall above the chair, the bed box missed its base, and the
window box was too short. This contaminated material samples, silhouette masks
and metric refinement. Groq position, inventory and bedding transports now use
an aspect-preserving, top-left-anchored square canvas. Coordinates are explicitly
mapped back to the original rectangle. Padding detections are rejected rather
than clamped into the photo. Cache versions and operation digests invalidate
incompatible earlier evidence without deleting uploads or saved rooms.

Inventory no longer inserts ceiling/wall fixtures whose outlines are contained
inside a window opening: exterior/reflected brightness is not an indoor fixture.
This does not remove table/floor lamps standing in front of windows.

Validation: 350 full-suite tests passed before the final fixture guard; 57
focused tests passed after it, including portrait/landscape frame round trips,
padding rejection and window-fixture tests. Fresh full-room validation is still
blocked by the account daily allowance. No exact future-photo accuracy guarantee
is made, and existing exported GLBs are not silently rewritten.

## Confirmed failure

A live position response declared a 750-by-750 pixel frame but returned a bed
box ending at 882, 915. Strict validation correctly rejected that evidence.
The request now explicitly uses normalized 0–1000 XYXY coordinates with stable
object IDs. Rejections retain useful diagnostics; accepted position groups are
cached independently for retry. Unsupported evidence is not marked verified.

## General pipeline changes

- Audit results merge changed fields by object ID instead of replacing a newer
  layout with an older cached scene snapshot.
- Inventory checks are combined; verified bedding is not inventoried twice.
  Oversized component groups split only when necessary, retaining valid work.
- Groq uses one shared budget coordinator, measured token usage and right-sized
  detail output reservations rather than a second per-instance token wait.
- Depth estimation overlaps photo analysis. A server-owned local model host
  reuses model weights; `ONE_BEYOND_WARM_WORKERS=0` disables this optimization.
- With a configured Gemini key, independent fixture/component checks can run
  alongside Groq. `ONE_BEYOND_HYBRID_AUDITS=0` disables this. One provider owns
  layout; independent audit deltas cannot overwrite unchanged poses.
- Window rendering no longer embeds source-photo crops, including legacy cached
  window textures. Bed headboard inference accepts associated pillows outside
  the old upper-image band. Flat pillow fitting evaluates orientation before
  stretching dimensions and preserves thickness.

## Validation boundaries

The five-meter saved-scene replay exported and rendered without provider calls
or adding a Recent entry. Bed orientation, flat-pillow proportions and opaque
window were visually checked. This is not fresh-provider acceptance testing.

The live investigation exhausted Groq's daily account allowance. Its recorded
coordinate-frame failure is confirmed, but a complete new Groq reconstruction
and the under-300-second target remain unverified. Software cannot eliminate
provider quotas, outages or guarantee exact hidden geometry from one photograph.

Warm inference checks cancellation before and after inference; active numerical
inference may finish before its result is discarded. Browser cancellation during
provider/Blender execution still needs end-to-end acceptance testing.

## Window evidence protection (October 4 follow-up)

The saved five-meter scene showed a later silhouette correction moving the
window away from its fitted corners. Its normalized corner error increased
from approximately 0.000128 to 0.000390. Window corrections now must preserve
or improve the outline/box projection error as well as mask overlap. Outline
fitting also seeds from camera rays intersecting the wall, including translated
and rotated cameras. Multiple-window extraction no longer overwrites the depth
point map with the first outline's vertices.

Focused regression coverage includes misleading masks, both side walls with a
translated camera, and multiple window outlines. The saved-room preview uses
zero provider calls and is not added to Recent. The latest failed upload records
Groq daily exhaustion followed by Google daily exhaustion; swapping their roles
cannot bypass those account quotas. Fresh-provider visual acceptance remains
blocked until account availability is restored.

## Flat-pillow posture consistency

A saved detection declared a flat pillow but estimated its longest dimension
vertically. The old fitter could retain it because the tall detection box gave
the invalid shape a lower projection error. Flat pillows now normalize swapped
axes and constrain thickness before projection fitting, with mattress support
height restored. This is driven by posture and dimensions, not room identity.
Explicit upright cushions are not flattened. Regression tests cover rotated
beds, misleading tall boxes, absent camera calibration, and upright cushions.

## Pillow localization/material and foreground floor crop

Flat-pillow fitting retains the localized horizontal photo bounds rather than
rescaling them through the entire bed/headboard proxy. Vertical support-relative
mapping remains necessary when camera/mattress height is uncertain; fitting
absolute vertical pixels alone pushed the cushion back and shrank its footprint.
Pillow material samples exclude seam/underside shadows and clipped highlights,
using observed body midtones without forcing white or copying photo textures.

Thin floor-supported rugs cut by the bottom photograph edge can span the camera
plane. Bounded repair preserves their size/yaw and moves that cropped footprint
inside the entrance. Elevated, uncropped or fully behind-camera objects still
fail validation. The actual new upload initially reproduced the camera-plane
failure; a subsequent fresh provider analysis completed after the repair. This
analysis check did not export a GLB or add a Recent entry. Local saved-room
renders are separate from fresh reconstruction acceptance.
