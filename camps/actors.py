from functools import wraps, partial
import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
from scipy import signal
from copy import copy
import camps
from scipy.spatial import cKDTree
from camps.datastore import DataStore


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


# Actors take primary input data as first argument (a) (Actors that don't consume data may have optional arg input)
# Actors may take options and perform a technique based on settings; IE different amount of smoothing
# Actors may access data that is not provided as 'data' so long as the data accessed is determined by 'data' or is passed directly as a kwarg
# Camps.Variables are actors that don't consume data. They may take options related to how to access... IE datastore, chunk
# Actors are responsible for modifying metadata appropriately
# Actors should perform heavy work lazily and return a lazy xr.DataArray (otherwise they would initialize computation)
# The actor decorator registers actors in the module level dict actors so they may be accessed by name (or however we want to arganize these general computational routines)


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
def wind_speed_from_uv(a=None, *, u: camps.Variable, v: camps.Variable, datastore: DataStore, **kwargs) -> xr.DataArray:
    if a is not None:
        raise ValueError('actor wind_speed_from_uv cannot consume data')

    if not isinstance(u, camps.Variable):
        raise ValueError('u argument must be passed as a camps.Variable')
    if not isinstance(v, camps.Variable):
        raise ValueError('v argument must be passed as a camps.Variable')

    if 'Uwind' not in u.name:
        raise ValueError(f'variable passed as u {u} is not u wind speed')

    if not datastore:
        raise ValueError('no datastore provided; so cannot get U or V wind components')

    if 'chunk' in kwargs:
        chunk = kwargs['chunk']
    else:
        chunk = dict()

    u = u(datastore=datastore, chunk=chunk)
    v = v(datastore=datastore, chunk=chunk)
#    if 'chunk' in kwargs:
#        u = u.chunk(kwargs['chunk'])
#        v = v.chunk(kwargs['chunk'])

    speed = xr.apply_ufunc(lambda u, v: np.sqrt(u**2 + v**2), u, v, dask='allowed')
    speed.name = 'wind_speed'
    return speed


@actor
def to_netcdf(a: xr.DataArray, *, datastore: DataStore, **kwargs):
    out_handle = datastore.out_handle(a)
    a = a.to_netcdf(out_handle, compute=False, **kwargs)
    return a


@actor
def to_stations(data: xr.DataArray, *, stations: pd.DataFrame) -> xr.DataArray:

    xdim='x'
    ydim='y'

    ###################
    # Prep the template
    ###################

    da = data.copy()

    # reshape action
    da = da.stack(station=(xdim, ydim))
    da = da.isel(station=np.arange(len(stations)))  # select appropriate stations
    # end reshape action

    def config_meta(da, xdim, ydim, stations):
        da = da.reset_index('station') # remove the temorary multiindex

        da = da.reset_coords([xdim, ydim], drop=True) # remove x and y dimensions (may be lat/lon)
        if 'latitude' in da.coords:
            da = da.reset_coords(['latitude','longitude'], drop=True) # remove the lats and lons of the grid cells

        # prep station coord with 'station' numeric index
        stations = stations.reset_index()
        stations.index.set_names('station', inplace=True)
        # assign the new coords
        da = da.assign_coords({'station': stations.call})

        # prep lat/lon coords with 'station' call index
        stations = stations.set_index('call')
        stations.index.set_names('station', inplace=True)
        # assign the new coords
        da = da.assign_coords({'latitude': stations.lat})
        da = da.assign_coords({'longitude': stations.lon})
        return da

    da = config_meta(da, xdim, ydim, stations)
    # end preping the template

    def nearest_worker(da: xr.DataArray,* ,xdim ,ydim, stations) -> xr.DataArray:
        da = da.stack(station=(xdim, ydim))  # squash the horizontal space dims into one
        gridlonlat = np.column_stack((da['longitude'].data, da['latitude'].data))  # array of lon/lat pairs of gridpoints # not a lazy version of column_stack
        stationlonlat = np.column_stack((stations.lon, stations.lat))  # array of lon/lat pairs of stations

        tree = cKDTree(gridlonlat)  # kdtree is fast search of nearest neighbor (lat/lon value-wise)
        dist_ix = tree.query(stationlonlat)  # find the distance (degrees) and index of the nearest gridpoint to each station

        da = da.isel(station=dist_ix[1])
        da = config_meta(da, xdim, ydim, stations)
        return da

    kwargs = dict()
    kwargs['xdim'] = xdim
    kwargs['ydim'] = ydim
    kwargs['stations'] = stations
    data = xr.map_blocks(nearest_worker, data, kwargs=kwargs, template=da)

    return data
