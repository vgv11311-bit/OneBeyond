# Photo texture transfer correction

Confirmed from exported GLB textures: photographed RGB colors had been converted
to linear material values, then written directly as sRGB PNG texture bytes.
The glTF viewer decoded those bytes again, substantially darkening textured
pillows, wood, fabric and books. Non-textured ceramics correctly used linear
baseColorFactor, making the two material paths inconsistent.

Encode linear photo texture pixels to sRGB exactly once before packing. Keep
normal maps and legacy non-photo palettes unchanged. No exposure boost, shared
cabinet color, photo-specific RGB values or geometry changes are involved.

`test_texture_color_space.py` checks exported texture medians against three
distinct source RGB colors, including a dark color. All 106 backend tests pass.
`rebuild_detail_review.py --materials-only` preserves the entire saved object
list and recompiles materials without pose fitting or another detector call.
Single-image shading versus material-color ambiguity remains: this fixes the
transfer bug, not an assurance of pixel-identical appearance under new lights.
