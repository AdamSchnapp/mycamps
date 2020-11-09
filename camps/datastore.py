#!/usr/bin/env python3
import xarray as xr
from pathlib import Path
import os
import camps

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

    def __init__(self, **kwargs):
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

    def in_handle(self, var: camps.Variable):
        ''' Discover variable data lives based on configuration '''
#        from urllib.request import urlretrieve
#        urlretrieve('https://github.com/pydata/xarray-data/raw/master/air_temperature.nc', 'data.nc')
        return self.config['in_handle']

    def out_handle(self, var: camps.Variable):
        return 'out_data.nc'

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
