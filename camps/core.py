#!/usr/bin/env python3
import cf_xarray
import xarray as xr
from camps.variables import Variable
from camps.names import name_from_var_and_scheme
import camps
from collections.abc import Iterable

@xr.register_dataset_accessor("camps")
class CampsDataset:
    ''' extend xarray Dataset functionality '''

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def var_name(self, var):
        if isinstance(var, camps.Variable):
            print(f'producing variable name based on variable name scheme on dataset: {var.name}')
            return name_from_var_and_scheme(var, None)  #  change None to be the scheme on the file once implemented
            # create Variable name from Dataset name scheme

    def __contains__(self, obj):
        if isinstance(obj, camps.Variable):
            name = self.var_name(obj)
            return name in self._obj
        else:
            return obj in self._obj

    def __getitem__(self, var) -> xr.DataArray:
        if isinstance(var, camps.Variable):
            name = self.var_name(var)
            return self._obj.cf[name].camps.select(var)
        else:
            return self._obj.cf[var]

@xr.register_dataarray_accessor("camps")
class CampsDataarray:
    ''' extend xarray DataArray functionality '''

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

#    def update_meta_attrs(self, var):
#        ''' conform metadata to Variable '''
#        # The Variable class will define the relevant metadata attrs and values
#        # populate self._obj.attrs with the relevent metadata here
#        pass

    def select(self, var):
        ''' Return selection of variable from DataArray as DataArray '''
        a = filters[0](self._obj, var)
        for filter in filters[1:]:
            a = filter(a, var)
        return a

    def dim_ax_from_standard_name(self, standard_name) -> tuple:
        # return a tuple with dimension name (str) and axis (int) of the array

        for ix, dim_name in enumerate(self._obj.dims):

            if self._obj[dim_name].attrs['standard_name'] == standard_name:  # let lack of standard name be error
                return dim_name, ix

        raise ValueError(f'No dimension exists with standard_name: {standard_name}')

#   cf_xarray has access by standard name for some names
    def coord_name_from_standard_name(self, standard_name) -> str:
        for coord in self._obj.coords:
            try:
                if self._obj[coord].attrs['standard_name'] == standard_name:
                    return coord
            except KeyError:
                pass  # forgive coords that don't have 'standard_name' attribute

        raise ValueError(f'No coordinate exists with standard_name: {standard_name}')


    def chunks_spanning_dim(self, dim_name) -> int:
        # return number of chunks that span the array in the dim_name direction
        if self._obj.chunks is None:
            return 1   # the dataset is not chunked therefore the whole array can be considered as one chunk
        ax = self._obj.dims.index(dim_name)
        return len(self._obj.chunks[ax])

    def chunk_dict_from_std(self, chunk: dict) -> dict:
        # return chunk dictionary
        # if dims expressed in chunk dict are not dims, attempt to map them to dim names based on standard_name
        new_chunk = dict()
        for dim, per_chunk in chunk.items():
            if dim not in self._obj.dims:
                dim, _ = self.dim_ax_from_standard_name(dim)
            new_chunk[dim] = per_chunk
        return new_chunk

#   cf_xarray has similar functionality, but looks limited
    def chunk(self, chunk: dict) -> xr.DataArray:
        # return chunked DataArray
        # if dims expressed in chunk are not dims, attempt to map them to dim names based on standard_name
        return self._obj.chunk(self.chunk_dict_from_std(chunk))

filters = list()

def filter(f):
    filters.append(f)
    return f

@filter
def forecast_reference_time(data: xr.DataArray, var: camps.Variable) -> xr.DataArray:
    if var.forecast_reference_time is None:
        return data
    else:
        #reference_time, _ = data.camps.dim_ax_from_standard_name('forecast_reference_time')
        reference_time = 'reference_time'
        return data.loc[{reference_time: var.forecast_reference_time}]

@filter
def time(data: xr.DataArray, var: camps.Variable) -> xr.DataArray:
    if var.time is None:
        return data
    else:
        return data.sel(time=var.time)

@filter
def lead_time(data: xr.DataArray, var: camps.Variable) -> xr.DataArray:
    if var.lead_time is None:
        return data
    else:
        return data.sel(lead_time=var.lead_time)
