# Inventory coverage and material regression

The reported saved job `553b81e3-679` contained only eight scene objects.
Its missing books, opening, curtain and covering were absent from scene data,
not merely invisible meshes. The detail audit was unavailable, while the UI
still presented the output as complete.

Changes:

- Use one photo-only detail inventory after the primary spatial analysis,
  instead of three additional geometry-generating audit requests. Enlarged
  crops are derived from the current upload and retain full-image coordinates.
- Merge individually detected books and supported accessories into the scene.
  Initial geometry comes from each detected supporting object; downstream
  masks, depth and support fitting refine it. Do not replace accepted furniture
  poses or erase existing cropped openings when auditing small objects.
- Sample raised slat faces for periodic cabinet fronts, rather than treating
  dark gaps as the timber color. Remove additional photo-material darkening.
- Display partial-analysis warnings in both the completion screen and viewer.

No filename, image hash, reference-photo layout, fixed book count or saved
reference colors are used to select these behaviors. Provider omissions and
single-view scale ambiguity remain limitations; this is not exact recovery.

Validation: 101 backend tests passed; frontend production build passed.
A live detector check retained window/curtain/blanket and accepted nine books,
five vases and two cups. This checks detections, not final rendering quality.

Fresh end-to-end job `dc3bb4a7-af1` subsequently completed with 27 objects,
including 11 book instances, a window, curtain and blanket. Its exported GLB
was rendered and visually inspected with `render_geometry_review.py`: books
are separate visible meshes and both wall fixtures are visible. This diagnostic
uses Blender review lights, not the browser lighting. Exact photo color matching,
all accessory shapes, and browser movement were not verified by that render.
Backend was restarted; the frontend responds on localhost:5173.
