# Mixed-coordinate audit regression

Saved result 25678c08-3e7 contains an added window whose image box is over the
floor, a second blanket whose box spans unrelated room regions, and vessels
with support warnings. The verifier received a full image plus enlarged crops;
some returned boxes were inconsistent with full-image coordinates. Those boxes
then corrupted masks, colors and inferred placement.

Changes: use only the full photo in the active detail audit; reject additions
whose boxes conflict with their declared supporting object; deduplicate
overlapping cloth aliases and contained fixture detections. Include ceiling
lights in the inventory and compile new detected lights as flush fixtures.
No fixed light count or source-photo coordinates are used.

Desk-wall audit corrections are allowed, but a fresh desk-room test showed
that semantic verification still chose the back wall. Added a calibrated
projection check for wall-attached work desks beyond rear-wall corners,
followed by existing contact and chair fitting. Saved-evidence review
df930280-e69 verifies right-wall contact and chair orientation; source job
87921155-2c8 is a fresh full reconstruction and contains three ceiling lights.

Photo materials retain sampled white balance. Removed the second whole-room
warm/cool tint in the browser photo lighting rig; local fixture colors remain.
No general brightness increase or photo-specific palettes were introduced.

Validation: 111 backend tests and frontend build passed. Desk review GLB was
visually inspected with diagnostic Blender lighting, not browser screenshot
color comparison. Exact source appearance remains approximate.

Fresh stock-room job 83e9eb78-5da completed with 30 objects: one window,
one curtain, one blanket, 12 books, and supported vessels. No unresolved
support warnings were present. Its exported GLB review was visually inspected:
the floor-level extra window and floating vase/cloth artifacts from the reported
result are absent. No existing saved result was overwritten.
