import xarray as xr
# temporary utilities for putting camps nc files into more friendly metadata formats

def friendly_obs(ds: xr.Dataset) -> xr.Dataset:
    station = ds.station.sum(dim='num_characters').to_series().rename_axis('station')
    ds = ds.drop_dims('num_characters')
    ds = ds.rename_dims({'default_time_coordinate_size': 'time', 'number_of_stations': 'station'})
    ds = ds.assign_coords({'station': station})
    ds = ds.set_index({'time':'OM__phenomenonTimeInstant'})
    return ds

def friendly_gfs(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.set_coords(['longitude', 'latitude'])
    ds = ds.rename_dims({'default_time_coordinate_size': 'reference_time', 'lead_times': 'lead_time'})
    ds = ds.set_index({'reference_time':'FcstRefTime', 'lead_time':'LeadTime' })
