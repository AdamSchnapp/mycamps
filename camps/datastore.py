#!/usr/bin/env python3
import xarray as xr
from pathlib import Path
import os
import camps
from camps.names import scheme
from abc import ABC, abstractmethod

#datastore = None
#DEFAULT_NAME_SCHEME = '1'

class DataStoreError(Exception):
    pass


class DataStore(ABC):
    '''
        Imlplement access to i/o handles
        Let a datastore be an  optional tool for managing i/o
        Let subclassing of Datastore be the method for implementing datastore strategies for an application

        camps variable calls and computational routine calls may take datastores and access data via them

        API methods
            in_handle(self, var: camps.Variable): -> file or collection of files that form multi-file dataset to read from
                'in handle is responsible for providing confugration to xr.open_mfdataset for access to data'

            out_handle(self, var: camps.Variable, scheme=None): -> single file to write to
                'out handle is responsible for providing the file to write to'


    '''


    @abstractmethod
    def in_handle(self, var: camps.Variable):
        '''in handle is responsible for providing files that can be opened via xr.open_mfdataset based on variable metadata'''
        pass

    @abstractmethod
    def out_handle(self, var: camps.Variable):
        '''out handle is responsible for providing the single file to write to based on variable metadata'''
        pass

class SimpleDataStore(DataStore):

    def __init__(self, in_handle=None, out_handle=None):
        ''' initialize datastore from configuration '''
        self._init_kwargs = dict(in_handle=in_handle, out_handle=out_handle)
        self._in_handle = in_handle
        self._out_handle = out_handle

    def in_handle(self, var: camps.Variable):
        return self._in_handle

    def out_handle(self, var: camps.Variable):
        return self._out_handle

    def __repr__(self):
        ''' show how datastore was initialized '''
        return f'{self.__class__.__name__}(**{self._init_kwargs})'
#        self.var_name_version = DEFAULT_NAME_SCHEME
#        self.init_kwargs = kwargs
#
#        if 'netcdf' in kwargs:
#            if isinstance(kwargs['netcdf'], str):
#                self.ds = xr.open_dataset(kwargs['cf_files'])
#        self.ds = xr.tutorial.open_dataset('air_temperature.nc')
#        self.config = dict()
#        self.config.update(kwargs)
#        self.init_kwargs = kwargs
#        self.scheme = scheme
#        if in_handle:
#            self.in_handle_ = os.path.abspath(os.path.expanduser(in_handle))
#        if out_handle:
#            self.out_handle_ = os.path.abspath(os.path.expanduser(out_handle))
#        print(self.scheme)
#
#    @abstarctmethod
#    def in_handle(self, var: camps.Variable):
#        ''' Discover variable data lives based on configuration '''
##        from urllib.request import urlretrieve
##        urlretrieve('https://github.com/pydata/xarray-data/raw/master/air_temperature.nc', 'data.nc')
#        if self.in_handle_:
#            f = self.in_handle_
#        else:
#            f = 'out_data1.nc'
#
#        return f
#
#    @abstarctmethod
#    def out_handle(self, var: camps.Variable):
#
#        #f = self.file_from_var(self, var: camps.Variable)
#        if self.out_handle_:
#            f = self.out_handle_
#        else:
#            f = 'out_data1.nc'
#        # if out_file does not yet exist, create base camps cf file
#        if not os.path.exists(f):
#            print(f'creating to file {f}')
#            # create camps cf file with variable naming scheme
#            from camps.names import scheme # for now just use a scheme.
#            self.scheme.to_netcdf(f)
#        #f = self.file_from_var(self, var: camps.Variable)
#        # if out_file does not yet exist, create base camps cf file
#
#        return f
#
#    def scheme(self, var: camps.Variable):
#        ''' based on variable and datastore configuration, return a name scheme
#        '''
#        return scheme
