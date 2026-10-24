"""Convert photographed sRGB bytes to Blender's scene-linear material input."""
def photo_linear_rgba(rgb):
    channels=[max(0.,min(1.,float(v)/255)) for v in rgb[:3]]
    return [v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in channels]+[1.]
