"""Shared asset/support dimensions so cloth cannot drift from its mattress."""
def bed_layers(height,variant='default'):
    legged=variant in {'wooden_rails','painted_slatted'}
    frame=min(.4,height*.35) if legged else min(.22,height*.24)
    mattress=min(.28,max(.16,height*.27))
    return frame,mattress
