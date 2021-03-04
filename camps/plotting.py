#!/usr/bin/env python3
import numpy as np

def center_align_bar_thickness(a):
    ''' helper for creating bar thicknesses when bar thicknesses are variable
        width for bar thicknesses are 90% of the distance to the nearest other bar
    '''
    first_diff = np.fabs(a[1]) - np.fabs(a[0])
    last_diff = np.fabs(a[-1]) - np.fabs(a[-2])
    dist_to_next = np.fabs(np.ediff1d(a, to_end=last_diff))
    dist_from_prev = np.fabs(np.ediff1d(a, to_begin=first_diff))

    bar_thickness = np.minimum(dist_to_next, dist_from_prev) * 0.9
    return bar_thickness

