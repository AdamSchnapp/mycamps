#!/usr/bin/env python3
from functools import wraps, partial
import numpy as np
import pandas as pd
import xarray as xr
import dask.delayed
from itertools import product
from scipy import signal
from copy import copy
import camps
from scipy.spatial import cKDTree
from camps.datastore import DataStore
from typing import Union


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


def smooth2d_array(a: np.array, **kwargs) -> np.array:
    '''
    Return two dimensional array with nine point smother applied.
    '''
    kernel_9point = np.ones((3, 3)) / 3**2
    a = signal.convolve2d(a, kernel_9point, mode='same', boundary='symm')
    return a


def smooth2d_block(da: xr.DataArray, dims, **kwargs) -> xr.DataArray:
    '''
    Return an xr.DataArray with a block of data smoothed along two dimensions.
    Input must be xr.DataArray object that is not lazy and not chunked.
    Will smooth chunked data with xr.map_blacks.
    Lazy/Dasky xr.DataArray objects do not support data assignment with .loc
    like xr.DataArrays with numpy data.
    '''

    other_dims = [dim for dim in da.dims if dim not in dims]
    other_dim_arrays = [da[dim].data for dim in other_dims]
    # iterate over all the 2d sections applying smoother
    for dim_values in product(*other_dim_arrays):
        loc = {k: v for k, v in zip(other_dims, dim_values)}
        da.loc[loc] = smooth2d_array(da.loc[loc], **kwargs)
    return da


@actor
def smooth2d(da: Union[camps.Variable, xr.DataArray],
             dims=('x', 'y'),
             datastore: DataStore = None, chunks: dict = None,
             **kwargs) -> xr.DataArray:
    '''
    Return an xr.DataArray smoothed along two dimensions.
    Works with both chunked(dask) and unchunked(numpy) data.
    Metadata attrs are adjusted according to camps metadata conventions.
    '''

    if len(dims) != 2:
        raise ValueError(f'dims {dims} is not length two')

    if isinstance(da, camps.Variable):
        da = da(datastore=datastore, chunks=chunks, **kwargs)
    else:
        if chunks:
            da = da.camps.chunk(chunks)

    dim0, _ = da.camps.dim_ax(dims[0])
    dim1, _ = da.camps.dim_ax(dims[1])
    dims = (dim0, dim1)
    kwargs['dims'] = dims  # kwargs are passed to smooth2d_block

    for dim in dims:
        n_chunks = da.camps.nchunks_spanning_dim(dim)
        if n_chunks > 1:
            raise ValueError(f'Expected chunks spanning dim "{dim}" to be one, but was {n_chunks};'
                             f' chunked data may not span dim "{dim}" with multiple chunks')

    da = xr.map_blocks(smooth2d_block, da, kwargs=kwargs, template=da)

    da.attrs['smooth'] = 'smooth_9point'

    return da


@actor
def wind_speed_from_uv(da=None, *, u: Union[camps.Variable, xr.DataArray],
                       v: Union[camps.Variable, xr.DataArray],
                       datastore: DataStore = None, chunks: dict = None, **kwargs) -> xr.DataArray:
    ''' return wind speed data array created from U and V wind speed components'''

    if da is not None:
        raise ValueError('wind_speed_from_uv cannot consume data except for u and v')

    if not datastore:
        raise ValueError('no datastore provided; so cannot get U or V wind components')

    uv_options = dict(datastore=datastore)
    if chunks:
        uv_options['chunks'] = chunks

    if isinstance(u, camps.Variable):
        u = u(**uv_options)
    elif isinstance(u, xr.DataArray):
        if chunks:
            u = u.camps.chunk(chunks)
    else:
        raise ValueError('v argument must be passed as a camps.Variable or xr.DataArray')

    if isinstance(v, camps.Variable):
        v = v(**uv_options)
    elif isinstance(u, xr.DataArray):
        if chunks:
            v = v.camps.chunk(chunks)
    else:
        raise ValueError('v argument must be passed as a camps.Variable or xr.DataArray')

    # check that the passed u component is expected
    if u.attrs['standard_name'] != 'eastward_wind':
       raise ValueError(f'variable passed as u {u} is not eastward_wind')
    # check that the passed v component is expected and agrees with u
    if v.attrs['standard_name'] != 'northward_wind':
       raise ValueError(f'variable passed as v {v} is not northward_wind')

    speed = xr.apply_ufunc(lambda u, v: np.sqrt(u**2 + v**2), u, v, dask='allowed')
    # Create metadata accordingly
    speed.name = 'wind_speed'
    speed.attrs['observed_property'] = 'wind_speed'
    return speed


#@actor
#def to_netcdf(da: Union[camps.Variable, xr.DataArray], *, datastore: DataStore = None,
#              chunks: dict = None, **kwargs):
#
#    if isinstance(da, camps.Variable):
#        da = da(datastore=datastore, chunks=chunks, **kwargs)
#    elif isinstance(da, xr.DataArray):
#        if chunks:
#            da = da.camps.chunk(chunks)
#
#    out_handle = datastore.out_handle(da)
#
#    da = da.to_netcdf(out_handle, compute=False, **kwargs)
#    return da


@actor
def to_stations(da: Union[camps.Variable, xr.DataArray], *, stations: pd.DataFrame,
                datastore: DataStore = None, chunks: dict = None, **kwargs) -> xr.DataArray:

    if isinstance(da, camps.Variable):
        da = da(datastore=datastore, chunks=chunks, **kwargs)
    elif isinstance(da, xr.DataArray):
        if chunks:
            da = da.camps.chunk(chunks)

    try:
        xdim = da.camps.x.name
        ydim = da.camps.y.name
    except ValueError:
        xdim = da.camps.longitude.name
        ydim = da.camps.latitude.name

    # need to know the lat and lon coord names, they would only be the same as xdim,ydim for mercator grid
    lon = da.camps.longitude.name
    lat = da.camps.latitude.name

    ###################
    # Prep the template
    ###################

    da_template = da.copy()

    # reshape action
    da_template = da_template.stack(station=(xdim, ydim))
    da_template = da_template.isel(station=np.arange(len(stations)))  # select appropriate stations
    # end reshape action

    def config_meta(da, xdim, ydim, stations):
        da = da.reset_index('station') # remove the temorary multiindex

        da = da.reset_coords([xdim, ydim], drop=True) # remove x and y dimensions (may be lat/lon)
        if lat in da.coords:
            da = da.reset_coords([lat, lon], drop=True) # remove the lats and lons of the grid cells

        # prep station coord with 'station' numeric index
        stations = stations.reset_index()
        stations.index.set_names('station', inplace=True)
        # assign the new coords
        da = da.assign_coords({'platform_id': stations.platform_id})

        # prep lat/lon coords with 'station' call index  ## previously I was indexing by platform_id, but platform id is not strictly a cf "coordinate variable" based on NUG because it is not numeric
        #stations = stations.set_index('platform_id')
        #stations.index.set_names('station', inplace=True)

        # assign the new coords with arbitrary integer index
        da = da.assign_coords({'lat': stations.lat})
        da.lat.attrs['standard_name'] = 'latitude'
        da = da.assign_coords({'lon': stations.lon})
        da.lon.attrs['standard_name'] = 'longitude'

        # drop the arbitrary station dim/index coordinate (for cf NUG conventions... station is an arbitrary coordinate described by auxiliarry coordinate vraiable platform_id)
        da = da.reset_index('station', drop=True)

        return da

    da_template = config_meta(da_template, xdim, ydim, stations)
    # end preping the template

    def nearest_worker(da: xr.DataArray,* ,xdim ,ydim, stations) -> xr.DataArray:
        da = da.stack(station=(xdim, ydim))  # squash the horizontal space dims into one
        gridlonlat = np.column_stack((da[lon].data, da[lat].data))  # array of lon/lat pairs of gridpoints # not a lazy version of column_stack
        stationlonlat = np.column_stack((stations.lon, stations.lat))  # array of lon/lat pairs of stations

        tree = cKDTree(gridlonlat)  # kdtree is fast search of nearest neighbor (lat/lon value-wise)
        dist_ix = tree.query(stationlonlat)  # find the distance (degrees) and index of the nearest gridpoint to each station

        da = da.isel(station=dist_ix[1])
        # still need to determine which stations should be missing as aren't covered by grid
        da = config_meta(da, xdim, ydim, stations)
        return da

    kwargs['xdim'] = xdim
    kwargs['ydim'] = ydim
    kwargs['stations'] = stations
    data = xr.map_blocks(nearest_worker, da, kwargs=kwargs, template=da_template)

    return data
