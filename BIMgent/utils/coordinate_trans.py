# Compute length and midpoint
import math

# GUI panel corners (x, y)
gui_tl = (471, 250)    # top‑left in GUI
gui_br = (979, 974)    # bottom‑right in GUI

# Real‑world corners (x, y)
real_tl = (-5036,  7197)   # maps from gui_tl
real_br = ( 5036, -7197)   # maps from gui_br

def gui_to_real(gui_pt):
    x, y = gui_pt

    # Linear interpolation for X
    real_x = real_tl[0] + (x - gui_tl[0]) * (real_br[0] - real_tl[0]) / (gui_br[0] - gui_tl[0])

    # Linear interpolation for Y (note the inverted direction)
    real_y = real_tl[1] + (y - gui_tl[1]) * (real_br[1] - real_tl[1]) / (gui_br[1] - gui_tl[1])

    return (real_x, real_y)

def map_gui_to_ifc(x1, y1, x2, y2):
    # Your wall endpoints in GUI coords
    gui_endpoints = [(x1, y1), (x2, y2)]

    # Transform to real world
    real_endpoints = [gui_to_real(pt) for pt in gui_endpoints]


    length     = math.dist(real_endpoints[0], real_endpoints[1])
    midpoint   = (
        (real_endpoints[0][0] + real_endpoints[1][0]) / 2,
        (real_endpoints[0][1] + real_endpoints[1][1]) / 2,
    )
    
    return real_endpoints, length, midpoint

