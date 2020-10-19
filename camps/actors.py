from functools import wraps, partial
import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
from scipy import signal
from copy import copy
import camps


actors = dict()


def actor(method=None, **kwargs):

    if not callable(method):
        return partial(actor, **kwargs)

    @wraps(method)
    def wrapper(*args, **kwargs):
        return method(*args, **kwargs)

    if 'actor_name' in kwargs:
        name = kwargs['actor_name']
    else:
        name = method.__name__
    actors[name] = wrapper

    return wrapper

# Actors take primary input data as first argument (a) (Actors that don't consume data can expect None as input)
# Actors may take options and perform a technique based on settings; IE different amount of smoothing
# Actors may access data that is not provided as 'data' so long as the data accessed is determined by 'data'
# Camps.Variables are actors that don't consume data and don't take options
# Data handles may be provided via options when other data is needed
# Actors are responsible for modifying metadata appropriately
# Actors should perform heavy work lazily and return a lazy xr.DataArray whenever possible
# The actor decorator registers actors in the module level dict actors so they may be accessed by name


def smooth2d_array(a: np.array) -> np.array:
    '''
    Return two dimensional array with nine point smother applied.
    '''
    kernel_9point = np.ones((3, 3)) / 3**2
    a = signal.convolve2d(a, kernel_9point, mode='same', boundary='symm')
    return a


def smooth2d_block(a: xr.DataArray, dims=('lat', 'lon')) -> xr.DataArray:
    '''
    Return an xr.DataArray with a block of data smoothed along two dimensions.
    Input must be xr.DataArray object that is not lazy and not chunked.
    Will smooth chunked data with xr.map_blacks.
    Lazy/Dasky xr.DataArray objects do not support data assignment with .loc
    like xr.DataArrays with numpy data.
    '''

    two_d_axis, two_d_dims = zip(*[(i, dim) for i, dim in enumerate(a.dims) if dim in dims])

    other_dims = [dim for dim in a.dims if dim not in two_d_dims]
    other_dim_arrays = [a[dim].data for dim in other_dims]
    for dim_values in product(*other_dim_arrays):
        loc = {k: v for k, v in zip(other_dims, dim_values)}
        a.loc[loc] = smooth2d_array(a.loc[loc])
    return a


@actor
def smooth2d(a: xr.DataArray, dims=('lat', 'lon'), **kwargs) -> xr.DataArray:
    '''
    Return an xr.DataArray smoothed along two dimensions.
    Works with both chunked(dask) and unchunked(numpy) data.
    Metadata attrs are adjusted according to camps metadata conventions.
    '''

    if len(dims) != 2:
        raise ValueError(f'dims {dims} is not length two')

    two_d_axis, two_d_dims = zip(*[(i, dim) for i, dim in enumerate(a.dims) if dim in dims])

    if len(two_d_axis) != 2:
        raise ValueError(f'dims {dims} were not both in array {a}')

    if 'chunk' in kwargs:
        a = a.chunk(kwargs['chunk'])
        del kwargs['chunk']

    if a.chunks:
        for axis, dim in zip(two_d_axis, two_d_dims):
            len_dim = len(a.chunks[axis])
            if len_dim > 1:
                raise ValueError(f'Expected chunks spanning dim "{dim}" to be one, but was {len_dim};'
                                 f' chunked data may not span dim "{dim}" with multiple chunks')

    kwargs['dims'] = dims
    a = xr.map_blocks(smooth2d_block, a, kwargs=kwargs, template=a)

    a.attrs['smooth'] = 'smooth_9point'

    return a


@actor
def wind_speed_from_uv(a: None, *, u: camps.Variable, v: camps.Variable, datastore=None, **kwargs) -> xr.DataArray:
    if a:
        raise ValueError('actor wind_speed_from_uv cannot consume data')

    if not isinstance(u, camps.Variable):
        raise ValueError('u argument must be passed as a camps.Variable')
    if not isinstance(v, camps.Variable):
        raise ValueError('v argument must be passed as a camps.Variable')

    if 'U_wind' not in u.name:
        raise ValueError(f'variable passed as u {u} is not u wind speed')

    if not datastore:
        raise ValueError('no datastore provided; so cannot get U or V wind components')

    u = u(datastore=datastore)
    v = v(datastore=datastore)
    if 'chunk' in kwargs:
        u = u.chunk(kwargs['chunk'])
        v = v.chunk(kwargs['chunk'])

    speed = xr.apply_ufunc(lambda u, v: np.sqrt(u**2 + v**2), u, v, dask='allowed')
    speed.name = 'wind_speed'
    return speed

@actor
def to_netcdf(a: xr.DataArray, *, datastore=None, **kwargs):
    if not datastore:
        raise ValueError('no datastore provided; so cannot determine where to write data')
    out_handle = datastore.out_handle(a)
    a.to_netcdf(out_handle, compute=False, **kwargs)
    return a


@actor
def to_stations(data: xr.DataArray = None, stations: pd.DataFrame = None, **kwargs) -> xr.DataArray:
    if not data:
        raise ValueError('no data provided')
    if not stations:
        raise ValueError('no stations provided')
    # transform gridded array to station array
