# Bed direction and accessory review

Evidence: saved scene 85943bbb-617 has bed yaw 180 and back-wall contact.
The bed asset places its headboard on local -Z; this input turns its headboard
toward the source camera, hiding the mattress and pillows. Resolve only an
opposite-direction ambiguity on the already detected wall-normal axis. Do not
rotate beds whose long side contacts a wall.

Independent books can have overlapping fitted widths. Preserve their inferred
centres and limit upright spine widths against neighbours on the same support.
Do not create extra books or use a predetermined book count.

Color samples are retained per object. Conservative gains from nearby neutral
unoccupied source-image regions reduce baked illumination differences. This
is an approximation, not intrinsic-material recovery or a forced shared color.

Replace ribbed cylinder-bundle vessel bodies with continuous curved hollow
surfaces. Preserve photo-derived per-instance material colors and cup handles.

The original source does contain a rectangular leaning board above the left
cabinet. Image-evidenced support now grounds such artwork on the cabinet top,
constrains its width by relative image bounds, and samples its actual color.
No photo, filename, hash, palette, layout or fixed object count is hardcoded.

Validation: 105 backend tests passed, including exported accessory bounds,
photo material transfer, reversed headboards, separated spines and art support.
Saved-evidence review 0b84e21a-d0e was exported and visually inspected with
Blender diagnostic lights: mattress/pillows are visible, upright books have
clearance, vessels have continuous surfaces, and the board rests on its cabinet.
This was not a new AI detection run or browser movement test. Exact source
appearance, especially illumination and small-object classification, remains
approximate. Backend restarted; frontend returned HTTP 200 on port 5173.
