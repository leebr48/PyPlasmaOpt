'''
This script calculates the average (minor) radius of a list of curves (obj_to_find_avg_rad_of). 
'''
from pyplasmaopt import * 
import numpy as np

(coils, mas, currents) = get_ncsx_data(Nt_ma=6, Nt_coils=6, ppp=20, copies=1, contNum=0, contRad=1)

obj_to_find_avg_rad_of = coils

rall = np.array([])
for coil in obj_to_find_avg_rad_of:
    c = np.average(coil.gamma, axis=0)
    rcoil = np.linalg.norm(coil.gamma-c,axis=1)
    rall = np.append(rall,rcoil)

minor_radius = np.average(rall)

print('MINOR RADIUS: ', minor_radius)
