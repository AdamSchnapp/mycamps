#!/usr/bin/env python3
import xarray as xr
from pathlib import Path
import os
import camps
from camps.names import scheme

#datastore = None
#DEFAULT_NAME_SCHEME = '1'

class DataStoreError(Exception):
    pass


class DataStore:
    ''' Imlplement access to i/o handles
        Let a datastore be an  optional tool for managing i/o
        Unsure about how much datastore will read from files on disk?
            does it validate files?
            does it search for data?
    '''

    def __init__(self, scheme=scheme, out_handle=None, in_handle=None, **kwargs):
        ''' initialize datastore from configuration '''
        ''' let configuration inform how/where new data is stored
                how:
                    file  (data split across multiple cf files)
                        location_path
                    group  (data split across multiple cf groups)
                        file_path
                dividers:
                    type:
                        station:
                        grid:
                            grid_spec:
                    time:
                    source:
            let configuration determine where to find input data
            let datastore look at contents of file for determing disk handles ?

        '''

#        self.var_name_version = DEFAULT_NAME_SCHEME
#        self.init_kwargs = kwargs
#
#        if 'netcdf' in kwargs:
#            if isinstance(kwargs['netcdf'], str):
#                self.ds = xr.open_dataset(kwargs['cf_files'])
#        self.ds = xr.tutorial.open_dataset('air_temperature.nc')
        self.config = dict()
        self.config.update(kwargs)
        self.init_kwargs = kwargs
        self.scheme = scheme
        if in_handle:
            self.in_handle_ = os.path.abspath(os.path.expanduser(in_handle))
        if out_handle:
            self.out_handle_ = os.path.abspath(os.path.expanduser(out_handle))
        print(self.scheme)

    def in_handle(self, var: camps.Variable):
        ''' Discover variable data lives based on configuration '''
#        from urllib.request import urlretrieve
#        urlretrieve('https://github.com/pydata/xarray-data/raw/master/air_temperature.nc', 'data.nc')
        if self.in_handle_:
            f = self.in_handle_
        else:
            f = 'out_data1.nc'

        return f

    def out_handle(self, var: camps.Variable):

        #f = self.file_from_var(self, var: camps.Variable)
        if self.out_handle_:
            f = self.out_handle_
        else:
            f = 'out_data1.nc'
        # if out_file does not yet exist, create base camps cf file
        if not os.path.exists(f):
            print(f'creating to file {f}')
            # create camps cf file with variable naming scheme
            from camps.names import scheme # for now just use a scheme.
            self.scheme.to_netcdf(f)
        #f = self.file_from_var(self, var: camps.Variable)
        # if out_file does not yet exist, create base camps cf file

        return f

    def scheme(self, var: camps.Variable):
        ''' based on variable and datastore configuration, return a name scheme
        '''
        return scheme

    def __repr__(self):
        ''' show how datastore was initialized '''
        return str(self.init_kwargs)
#
#    def __enter__(self):
#        ''' store self on camps module '''
#        camps.datastore = self
#
#    def __exit__(self, exc_type, exc_value, traceback):
#        ''' shutdown the datastore; possible cleanup '''
#        #self.shutdown()
#        pass
#
#    def put(self, data):
#        ''' put data to internal datastore; would only be accesible in this same process '''
#        # raise error if data does not have unique variable name
#        self.ds[data.name] = data
#        pass
#
#    def extend(self, data):
#        ''' update data in internal datastore; would only be accesible in this same process '''
#        # for extending data that is already in datastore; this will ussually be extending coordinates
#        pass
#
#    def get(self, data):
#        ''' return xarray dataarray from datastore '''
#        pass
#
#    def save(self, data):
#        ''' mark that should be put to datastore; and persisted '''
#        #self.put(data)
#        pass

#    def _create_camps_cf(path, name_scheme=DEFAULT_NAME_SCHEME):
#        ''' for creating new files with camps conventions
#             who should have file creation responsibility? should data to_disk ops always be appends?
#        '''
#        pass
#        if os.path.exists(path):
#            raise ValueError(f'{path} already exists and cannot be created new')
#        ds=xr.Dataset()
#        ds.atrrs['camps_var_name_scheme_version'] = name_scheme
#        ds['camps_var_name_scheme'] = camps.names.var_name_schemes[name_scheme]
#        return ds

def create_new_cf(f, scheme):
    ''' create a new camps cf file on disk '''
    scheme.var.to_netcdf(f)
