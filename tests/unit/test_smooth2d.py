import xarray as xr
from camps.actors import smooth2d, smooth2d_array, smooth2d_block
import numpy as np

def test_smooth2d_array():
    a = [[0, 0, 9, 9],
         [0, 0, 9, 9],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]

    smoothed = smooth2d_array(a)
    expected = np.array([[0., 3., 6., 9.],
                         [0., 2., 4., 6.],
                         [0., 1., 2., 3.],
                         [0., 0., 0., 0.]])

    np.testing.assert_array_equal(smoothed, expected)

def test_smooth2d_block():
    a = np.array([[[0, 0, 9, 9],
                   [0, 0, 9, 9],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]],

                  [[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [9, 9, 0, 0],
                   [9, 9, 0, 0]]])

    a_expected = np.array([[[0, 3, 6, 9],
                            [0, 2, 4, 6],
                            [0, 1, 2, 3],
                            [0, 0, 0, 0]],

                           [[0, 0, 0, 0],
                            [3, 2, 1, 0],
                            [6, 4, 2, 0],
                            [9, 6, 3, 0]]])

    coords = [('time', range(a.shape[0])),
              ('x', range(a.shape[1])),
              ('y', range(a.shape[2]))]

    a = xr.DataArray(a, coords=coords)
    a_expected = xr.DataArray(a_expected, coords=coords)

    a_smoothed = smooth2d_block(a, dims=('y','x'))

    xr.testing.assert_equal(a_smoothed, a_expected)
