#!/usr/bin/env python3
import cf_xarray
import xarray as xr
import pandas as pd
from camps.variables import Variable
from camps.names import name_from_var_and_scheme, VarNameScheme
import camps
from camps.meta import meta, meta_pieces
import os
from collections.abc import Iterable
from dask.base import tokenize

@xr.register_dataset_accessor("camps")
class CampsDataset:
    ''' extend xarray Dataset functionality '''

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def var_name(self, var):
        if isinstance(var, camps.Variable):
            print(f'producing variable name based on variable name scheme on dataset: {var.name}')
            scheme = VarNameScheme.from_dataset(self._obj)
            print(f'scheme: {scheme}')
            name = name_from_var_and_scheme(var, scheme)  #  change None to be the scheme on the file once implemented
            print(f'name based on scheme: {name}')
            return name

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

    @property
    def name_scheme(self):
        return VarNameScheme.from_dataset(self._obj)

#class Time:
#    def __get__(self, obj, objtype=None):
#        meta_piece = meta_pieces['time']
#        try:
#            return obj._obj[meta_piece.coord_name(obj._obj)]
#        except MayBeDerivedError:
#            print('handle trying to compute via meta piece')

class Coord:
    def __set_name__(self, owner, name):
        #self.private_name = f'_{name}'
        self.name = name

    def __get__(self, obj, objtype=None):
        coord_name = meta_pieces[self.name].coord_name(obj._obj)
        if coord_name:
            return obj._obj[coord_name]
        else:
            raise KeyError(f'no metadata coord for "{name}"')


@xr.register_dataarray_accessor("camps")
class CampsDataarray:
    ''' extend xarray DataArray functionality '''
    #time = Time()
    reference_time = Coord()
    lead_time = Coord()
    time = Coord()
    latitude = Coord()
    longitude = Coord()
    x = Coord()
    y = Coord()
    z = Coord()


    def __init__(self, xarray_obj):
        self._obj = xarray_obj

#    def update_meta_attrs(self, var):
#        ''' conform metadata to Variable '''
#        # The Variable class will define the relevant metadata attrs and values
#        # populate self._obj.attrs with the relevent metadata here
#        pass

    def select(self, var):
        ''' Return selection of variable from DataArray as DataArray '''
        #filters = list(meta_pieces.values())
        filters = [meta_pieces[meta] for meta in var.meta_filters_]
        a = filters[0].select(self._obj, var)
        for filter in filters[1:]:
            a = filter.select(a, var)
        return a

    def dim_ax(self, camps_name) -> tuple:
        # return a tuple with dimension name (str) and axis (int) of the array

        name = getattr(self._obj.camps, camps_name).name
        for ix, dim_name in enumerate(self._obj.dims):

            if dim_name == name:
                return dim_name, ix

        raise ValueError(f'No dimension exists with standard_name: {standard_name}')

#   cf_xarray has access by standard name for some names
    def coord_name_from_attr_(self, attr_key, attr_val) -> str:
        ''' return the name of array coordinate based on attribute key and val '''
        coord_return = None
        for coord in self._obj.coords:
            try:
                if self._obj[coord].attrs[attr_key] == attr_val:
                    if coord_return:
                        raise ValueError('Multiple coords exist with {attr_key}: {attr_val}')
                    coord_return = coord
            except KeyError:
                pass  # pass by coords that don't have attr_key

        if coord_return:
            return coord_return

        raise KeyError(f'No coordinate exists with {attr_key}: {attr_val}')


    def to_netcdf(self, datastore, **kwargs):
        ''' write data according to name scheme
            writes are append only, cannot modify existing variables
            if file does not exist, it is created by datastore
        '''
        f = datastore.out_handle(self._obj)
        out_file_ds = xr.open_dataset(f) # read only meta arrays
        scheme = out_file_ds.camps.name_scheme
        arrays = self._obj.camps.split_based_on_scheme(scheme)
        ds = xr.Dataset()
        for array in arrays:
            array.name = name_from_var_and_scheme(array, scheme)
            if array.name in out_file_ds:
                raise ValueError(f'no support for modifying variables that are already on output file {f}')
            else:
                # this is a new variable, ensure no coords conflict
                for coord_name in array.coords:
                    if coord_name in out_file_ds:
                        if not array[coord_name].equals(out_file_ds[coord_name]):
                            new_coord_name = coord_name + str(tokenize(array[coord_name]))
                            array = array.rename({coord_name:new_coord_name})

                    if coord_name in ds:
                        if not array[coord_name].equals(ds[coord_name]):
                            new_coord_name = coord_name + str(tokenize(array[coord_name]))
                            array = array.rename({coord_name:new_coord_name})

            ds = ds.assign({array.name: array})

        out_file_ds.close()
        save = ds.to_netcdf(f, mode='a', **kwargs)
        return save

    def split_based_on_scheme(self, scheme) -> list:
        ''' split self array based on var naming scheme '''
        arrays = [self._obj]
        temp_arrays = list()
        for meta_piece_name in scheme.pieces:
            meta_piece = meta_pieces[meta_piece_name]
            for arr in arrays:
                arr = meta_piece.split_array(arr)
                temp_arrays.extend(arr)
            arrays = temp_arrays
            temp_arrays = list()
        return arrays
