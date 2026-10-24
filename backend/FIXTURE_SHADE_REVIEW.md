# Fixture warmth and cabinet recess shadows

Cabinet meshes cast shadows but previously only the floor received them. Enable
shadow receiving on reconstructed cabinet/wardrobe/dresser/bookshelf descendants,
while leaving room-shell self-shadowing disabled. Reduce photo key-light normal
bias to 0.008 so small recess shadows are not offset away. Do not darken timber
albedo or change dimensions, geometry, object positions, exposure or light count.

Fixture chromaticity used the brightest ten percent of pixels, often clipped
white. Prefer bright non-clipped pixels inside each detected fixture crop;
normalize chromaticity separately from intensity. Neutral/unlit fixture tests
prevent a universal yellow tint or invented illumination. The actual uploaded
desk-room fixture samples now produce warm RGB ratios near [1,.8,.6].

Reduce photo-fabric environment reflection intensity to 0.15; retain local
fixture illumination and neutral broad fill. Sample-room lighting is unchanged.

114 backend tests and frontend build passed. Review f5067767-554 resamples only
fixture lighting from ac524949-633; existing geometry and material colors are
preserved. Browser visual shadow strength and final color appearance have not
been screenshot-verified. No filename/hash/layout-specific behavior was added.
