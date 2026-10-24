"""Shared, original accessory recipes. Coordinates and dimensions are Y-up XYZ."""
import math

ACCESSORIES={'laptop','keyboard','mouse','bottle','cup','vase','pen_holder','door','blinds','roller_shade',
             'curtain_rod','ceiling_fan','picture_frame','desk_lamp','cushion'}

def parts(label):
    result=[]
    def add(name,shape,size,center,color): result.append((name,shape,size,center,color))
    dark=[42,46,51,255]; metal=[115,120,125,255]; wood=[135,84,48,255]
    white=[218,213,201,255]; glass=[78,128,152,255]
    if label=='laptop':
        add('Laptop_Base','box',[1,.06,.65],[0,-.47,.06],metal)
        add('Laptop_Lid','box',[1,.94,.04],[0,0,-.30],dark)
        add('Laptop_Display','box',[.91,.80,.008],[0,.03,-.275],glass)
    elif label=='keyboard':
        add('Keyboard_Base','box',[1,.4,.95],[0,-.3,0],dark)
        for r in range(5):
            for c in range(14): add('Key','box',[.057,.35,.13],[(c-6.5)*.069,.1,(r-2)*.18],metal)
    elif label=='mouse':
        add('Mouse_Body','sphere',[.5,.5,.5],[0,0,0],dark)
        add('Mouse_Wheel','box',[.08,.06,.18],[0,.47,-.08],metal)
    elif label in {'bottle','vase','cup','pen_holder'}:
        # Open vessels use wall segments, not a solid disk over the mouth.
        for i in range(32):
            a=i*math.tau/32
            add('Vessel_Wall','cylinder',[.052,.86,.052],[math.cos(a)*.43,0,math.sin(a)*.43],white if label!='bottle' else glass)
        add('Vessel_Base','cylinder',[.47,.07,.47],[0,-.465,0],white)
        if label=='bottle':
            add('Bottle_Neck','cylinder',[.19,.18,.19],[0,.41,0],glass)
            add('Bottle_Cap','cylinder',[.20,.06,.20],[0,.53,0],dark)
        if label=='cup':
            for y in (-.24,.24): add('Cup_Handle','box',[.24,.09,.12],[.52,y,0],white)
            add('Cup_Handle','box',[.08,.55,.12],[.64,0,0],white)
    elif label=='door':
        add('Door_Panel','box',[.89,.96,.45],[0,-.02,0],wood)
        for x in (-.48,.48): add('Door_Jamb','box',[.04,1,.9],[x,0,0],dark)
        add('Door_Lintel','box',[1,.04,.9],[0,.48,0],dark)
        add('Door_Handle','sphere',[.035,.02,.13],[.32,-.03,.38],metal)
    elif label=='blinds':
        add('Blind_Header','box',[1,.04,1],[0,.48,0],white)
        for i in range(18): add('Blind_Slat','box',[1,.038,.6],[0,.44-i*.052,0],white)
    elif label=='roller_shade':
        # One continuous opaque fabric plane, never a repeated slat pattern.
        # Fabric overlaps its header and bottom rail, rather than floating
        # below the roller with a visible band of wall between them.
        add('Roller_Fabric','box',[1,.93,.05],[0,-.025,0],white)
        add('Roller_Header','box',[1,.065,.18],[0,.465,0],metal)
        add('Roller_Bottom_Rail','box',[1,.025,.08],[0,-.49,0],metal)
    elif label=='curtain_rod':
        add('Curtain_Rod','box',[.94,.20,.20],[0,0,0],metal)
        for x in (-.46,.46): add('Rod_Finial','sphere',[.04,.5,.5],[x,0,0],metal)
    elif label=='ceiling_fan':
        add('Fan_Stem','cylinder',[.03,.55,.03],[0,.21,0],metal)
        add('Fan_Motor','cylinder',[.12,.25,.12],[0,-.13,0],dark)
        add('Fan_Blades_X','box',[1,.025,.11],[0,-.22,0],wood)
        add('Fan_Blades_Z','box',[.11,.025,1],[0,-.22,0],wood)
    elif label=='picture_frame':
        add('Picture_Inset','box',[.90,.90,.25],[0,0,0],white)
        for x in (-.47,.47): add('Frame_Edge','box',[.06,1,1],[x,0,0],wood)
        for y in (-.47,.47): add('Frame_Edge','box',[1,.06,1],[0,y,0],wood)
    elif label=='desk_lamp':
        add('Lamp_Base','cylinder',[.46,.06,.46],[0,-.47,0],metal)
        add('Lamp_Stem','cylinder',[.045,.75,.045],[0,-.07,0],metal)
        add('Lamp_Shade','cylinder',[.4,.25,.4],[0,.375,0],white)
    elif label=='cushion':
        add('Cushion','sphere',[.5,.5,.5],[0,0,0],white)
    return result
