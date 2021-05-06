#!/usr/bin/env python3
from functools import wraps, partial
import numpy as np
import pandas as pd
import xarray as xr
import dask.delayed
from itertools import product
from scipy import signal
from copy import copy
import metpy
from metpy.plots.mapping import CFProjection
from pyproj import Proj
from scipy.spatial import cKDTree
import camps
from camps.datastore import DataStore
from typing import Union, List
from numbers import Number
from wrf import interplevel


actors = dict()

Var = Union[camps.Variable, xr.DataArray]




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
def smooth2d(da: Var,
             datastore: DataStore = None, chunks: dict = None,
             **kwargs) -> xr.DataArray:
    '''
    Return an xr.DataArray smoothed along two dimensions.
    Works with both chunked(dask) and unchunked(numpy) data.
    Metadata attrs are adjusted according to camps metadata conventions.
    '''

    if isinstance(da, camps.Variable):
        da = da(datastore=datastore, chunks=chunks, **kwargs)
    else:
        if chunks:
            da = da.camps.chunk(chunks)

    x = da.camps.x.name
    y = da.camps.y.name

    # rechunk so that multiple chunks don't span x and y dims
    if da.chunks is not None:
        da = da.chunk({x:-1, y:-1})

    dims = (x, y)
    kwargs['dims'] = dims  # kwargs are passed to smooth2d_block

    da = xr.map_blocks(smooth2d_block, da, kwargs=kwargs, template=da)

    da.attrs['smooth'] = 'smooth_9point'

    return da


@actor
def wind_speed_from_uv(da=None, *, u: Var,
                       v: Var,
                       datastore: DataStore = None, chunks: dict = None, **kwargs) -> xr.DataArray:
    ''' return wind speed data array created from U and V wind speed components'''

    if da is not None:
        raise ValueError('wind_speed_from_uv cannot consume data except for u and v')

    uv_options = dict(datastore=datastore)
    if chunks:
        uv_options['chunks'] = chunks

    if isinstance(u, camps.Variable):
        u = u(datastore=datastore, chunks=chunks, **kwargs)
    elif isinstance(u, xr.Dataset):
        if chunks:
            u = u.camps.chunk(chunks)
    else:
        raise ValueError('v argument must be passed as a camps.Variable or xr.DataArray')

    if isinstance(v, camps.Variable):
        v = v(datastore=datastore, chunks=chunks, **kwargs)
    elif isinstance(u, xr.Dataset):
        if chunks:
            v = v.camps.chunk(chunks)
    else:
        raise ValueError('v argument must be passed as a camps.Variable or xr.DataArray')

    # check that the passed u component is expected
    if u.data_var.attrs['standard_name'] != 'eastward_wind':
       raise ValueError(f'variable passed as u {u} is not eastward_wind')
    # check that the passed v component is expected and agrees with u
    if v.data_var.attrs['standard_name'] != 'northward_wind':
       raise ValueError(f'variable passed as v {v} is not northward_wind')

    speed = xr.apply_ufunc(lambda u, v: np.sqrt(u**2 + v**2), u, v, dask='allowed')
    # Create metadata accordingly
    speed.name = 'wind_speed'
    speed.attrs['observed_property'] = 'wind_speed'
    return speed


def edge(a):
    '''
    Return 1-dimensional array consisting of the perimiter points cyclicly
    >>> a = np.array([[1, 2, 3],
    ...               [4, 5, 6],
    ...               [7, 8, 9]])
    >>> edge(a)
        array([1, 4, 7, 8, 9, 6, 3, 2])
    '''

    if a.ndim != 2:
        raise ValueError("requires two dimensional array")
    s1 = a[:a.shape[0]-1,0]
    s2 = a[-1,:a.shape[1]-1]
    s3 = a[::-1,-1][:-1]
    s4 = a[0,::-1][:-1]
    return np.concatenate((s1,s2,s3,s4))

def to_stations(da: Var, *, stations: pd.DataFrame,
                datastore: DataStore = None, chunks: dict = None, **kwargs) -> xr.DataArray:

    if isinstance(da, camps.Variable):
        da = da(datastore=datastore, chunks=chunks, **kwargs)
    elif isinstance(da, xr.DataArray):
        if chunks:
            da = da.camps.chunk(chunks)

    # Determine x and y
    # Use Projected crs as common crs for grid and station

    try:
        x = da.camps.projx.name
        y = da.camps.projy.name
    except KeyError:
        # projected coordinates do not exist
        if da.camps.grid_mapping:
            # try to make them from grid_mapping and lat/lons
            da = da.metpy.assign_crs(da.camps.grid_mapping)
            da = da.metpy.assign_y_x()
            da = da.drop('metpy_crs')
            x = da.camps.projx.name
            y = da.camps.projy.name
        else:
            # exception for mercator data without grid_mapping
            lat = da.camps.latitude
            if lat.ndim == 1:
                y = lat.name
            lon = da.camps.longitude
            if lon.ndim == 1:
                x = lon.name

            # ensure longitude expressed as degrees east from prime meridian with -179 and 180 bounds
            lon_attrs = lon.attrs
            da[lon.name] = xr.where(lon > 180, lon - 360, lon)
            da[lon.name].attrs = lon_attrs

            # Add the lat lon grid mapping since it didn't have one
            # Latitude and longitude on the WGS 1984 datum
            lat_lon_wgs84 = {'grid_mapping_name' : "latitude_longitude",
                'longitude_of_prime_meridian' : 0.0,
                'semi_major_axis' : 6378137.0,
                'inverse_flattening' : 298.257223563}
            gm = xr.DataArray()
            gm.attrs.update(lat_lon_wgs84)
            da = da.assign_coords({'lat_lon_wgs84': gm})
            da.attrs['grid_mapping'] = 'lat_lon_wgs84'

    pyproj_crs = CFProjection(da.camps.grid_mapping).to_pyproj()
    stations['x'], stations['y'] = Proj(pyproj_crs)(stations.lon.values, stations.lat.values)

    # rechunk so that multiple chunks don't span x and y dims
    if da.chunks is not None:
        da = da.chunk({x:-1, y:-1})
        da = da.unify_chunks()

    # load x,y data
    da[x].load()
    da[y].load()

    # make horizonontal space 1-D by stacking x and y
    stacked = da.stack(xy=(x,y))
    gridxy = np.column_stack((stacked[x].data, stacked[y].data))

    stationxy = np.column_stack((stations.x, stations.y))

    tree = cKDTree(gridxy) # fast nearest neighbor search algorith
    dist_ix = tree.query(stationxy) # find distance to nearest and index of nearest

    # make a grid polygon
    from shapely.geometry import Polygon, MultiPoint, Point
    unit_y = xr.DataArray(np.ones(da[y].shape), dims=da[y].dims)
    unit_x = xr.DataArray(np.ones(da[x].shape), dims=da[x].dims)
    edge_x = edge(da[x] * unit_y)  # broadcast constant x along the y dim
    edge_y = edge(unit_x * da[y])  # broadcast constant y along the x dim

    xy = zip(edge_x, edge_y)
    grid_polygon = Polygon(xy)

    # make station points
    xy = zip(stations.x.values, stations.y.values)
    station_points = MultiPoint(list(xy))

    # determine which stations lie outside grid domain (polygon)
    stations['point_str'] = pd.Series([str(p) for p in station_points])
    stations['ix'] = stations.index
    points_outside_grid = station_points - station_points.intersection(grid_polygon)
    if not points_outside_grid.is_empty:
        if isinstance(points_outside_grid, Point):
            points_outside_grid = [points_outside_grid]
        ix_stations_outside_grid = stations.set_index('point_str').loc[[str(p) for p in points_outside_grid]].ix
    else:
        ix_stations_outside_grid = list() # let be empty list

    print(len(ix_stations_outside_grid))
    def nearest_worker(da: xr.DataArray, *, x, y, ix, ix_nan) -> xr.DataArray:
        da = da.stack(station=(x, y))  # squash the horizontal space dims into one

        da = da.isel(station=ix)
        da = da.drop_vars('station')  # remove station coord
        da.loc[{'station': ix_nan}] = np.nan  # use integer index location to set stations outside grid to missing

        return da
    # make template for expressing the change in shape in map_blocks
    template = da.copy()
    # reshape action
    template = template.stack(station=(x, y))  # combine the lat/lon dims into one dim called station
    template = template.isel(station=[0]*len(stations))  # select only the first; this removes the dim station
    template = template.drop('station')  # drop the multiindex lat/lon coord associated with 'station' from the 0th grid point

    mb_kwargs = dict(x=x, y=y, ix=dist_ix[1], ix_nan=ix_stations_outside_grid)
    da = xr.map_blocks(nearest_worker, da, kwargs=mb_kwargs, template=template)

    # remove any metadata that may be leftover from the grid
    da = da.drop_vars([x, y], errors='ignore')

    # configure station metadata
    # prep station coord with numeric index called 'station'
    stations = stations.reset_index()
    stations.index.set_names('station', inplace=True)
    # assign the new coords
    da = da.assign_coords({'platform_id': stations.call})
    da.platform_id.attrs['standard_name'] = 'platform_id'

    # assign the new coords with numeric index
    da = da.assign_coords({'lat': stations.lat})
    da.lat.attrs['standard_name'] = 'latitude'
    da.lat.attrs['units'] = 'degrees_north'
    da = da.assign_coords({'lon': stations.lon})
    da.lon.attrs['standard_name'] = 'longitude'
    da.lon.attrs['units'] = 'degrees_east'
    # drop the numeric index;
    da = da.reset_index('station', drop=True)

    return da

@actor
def interp_to_isosurfaces(da: Var, *,
                level_data: Var,
                isovalues: Union[Number, List[Number]],
                datastore: DataStore = None, chunks: dict = None, **kwargs) -> xr.DataArray:

    if isinstance(da, camps.Variable):
        da = da(datastore=datastore, chunks=chunks, **kwargs)
    elif isinstance(da, xr.DataArray):
        if chunks:
            da = da.camps.chunk(chunks)

    if isinstance(level_data, camps.Variable):
        level_data = level_data(datastore=datastore, chunks=chunks, **kwargs)
    elif isinstance(level_data, xr.DataArray):
        if chunks:
            level_data = level_data.camps.chunk(chunks)


    # rechunk so that multiple chunks don't span z dim
    if da.chunks is not None:
        da = da.camps.chunk({'z':-1})

    if da.coords.keys() != level_data.coords.keys():
        raise ValueError('data and level variable coords do not match')

    # detach and re-attach non-NUG coords
    z = da.camps.z.name
    saved_coords = dict()
    for coord in da.coords:
        if len(da[coord].dims) > 1:
            if z in da[coord].dims:
                raise ValueError('non-NUG coords spanning z axis are not allowed')
            saved_coords[coord] = da[coord]
            da = da.drop(coord)
            level_data = level_data.drop(coord)


    # prep for computation via map_blocks
    kwargs['isovalues'] = isovalues

    # inputs prepped as dataset as map_blocks does not take multiple chunked arrays
    ds = xr.Dataset()
    ds['variable'] = da
    ds['level_variable'] = level_data

    # prep output template (the output will have this meta structure)
    template = interp_to_isosurface_meta_template(ds, isovalues)

    # horizontal dims will be either x,y grid or stations, for now don't worry about handling stations
    # rename input z axis dim name to output z axis dim name (determined by meta template; only z is touched)
    x = da.camps.x.name
    y = da.camps.y.name
    z_in = da.camps.z.name
    z_final = template.camps.z.name
    ds = ds.rename({z_in: z_final})
    kwargs['space_dims'] = [ z_final, x, y]

    # perform work on each chunked block individually with the interp_to_isosurface_worker
    #da = xr.map_blocks(interp_to_isosurface_worker, ds, kwargs=kwargs, template=template)
    da = xr.map_blocks(interp_to_isosurface_worker, ds, kwargs=kwargs, template=template)

    # re-assign coords that spanned more than one dimension
    da = da.assign_coords(saved_coords)

    return da

def interp_to_isosurface_meta_template(ds, isovalues):
    ''' ds is a dataset with variables named variable and level_variable '''

    if not isinstance(isovalues, list):
        isovalues = [isovalues]

    z = ds.variable.camps.z.name
    da = ds.variable.drop(z) # drop the coordinate z (metadata array) # this metadata array will be replaced with isovalues
    da = da.isel({z:range(len(isovalues))})
    da = da.assign_coords({z: isovalues})
    da[z].attrs = ds.level_variable.attrs  # need to get only relevant attrs from level_data; for now just get all
    da[z].attrs['axis'] = 'Z'
    da = da.rename({z: 'z'})
    return da



def interp_to_isosurface_worker(ds: xr.Dataset, *, isovalues, space_dims, **kwargs) -> xr.DataArray:
    '''
    Return an xr.DataArray with a block of data interpolated to an isosurface.
    Input must be xr.Dataset object that is not lazy and not chunked.
    Input dataset will have two variables with names "variable" and "level_variable"
    the two arrays needed are passed as a dataset as xr.map_blocks cannot take a datarray as kwarg
    Lazy/Dasky xr.DataArray objects do not support data assignment with .loc
    like xr.DataArrays with numpy data.
    '''

    # make final data structure to populate with data
    final = interp_to_isosurface_meta_template(ds, isovalues)

    # record initial axis order; for reshaping back to initial before return
    final_dims = final.variable.dims
    ordered_dims = list(final_dims)
    z = final.camps.z.name
    ordered_dims.remove(z)
    ordered_dims.insert(0, z) # ordered dims now has z as left-most axis for use with wrf interplevel

    # go to working space shape
    ds = ds.transpose(*ordered_dims) # input
    final = final.transpose(*ordered_dims) # output



    # order dims for work with interplevel
    other_dims = [dim for dim in ds.dims if dim not in space_dims]
    other_dim_arrays = [ds[dim].data for dim in other_dims]
    # iterate over all the space sections applying interplevel
    for dim_values in product(*other_dim_arrays):
        loc = {k: v for k, v in zip(other_dims, dim_values)}
        final.loc[loc] = interplevel(ds.variable.loc[loc], ds.level_variable.loc[loc], isovalues, meta=False, **kwargs)


    # go back to initial shape and axis order
    final = final.transpose(*final_dims)
    return final


