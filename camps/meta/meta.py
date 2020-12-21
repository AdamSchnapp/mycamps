import pathlib
import yaml
import xarray as xr
import camps
from camps.helpers import UniqueValDict, ClassRegistry, removeprefix
from abc import ABC, abstractmethod
import datetime
import numpy as np
import pandas as pd

META = pathlib.Path(__file__).parent.absolute()

class MetaConfig(dict):
    ''' Implements access to camps core metadata configuration
    '''
    def __init__(self, **kwargs):
        ''' Load meta.yaml file '''
        super().__init__()
        f = META / 'meta.yaml'
        with open(f) as f:
            data = yaml.safe_load(f)
        self.update(data)

#    def update(self, other):
#        ''' update registry with data from other, accomodate URLs to codes registries '''
#        super().update(other)

meta_config = MetaConfig()

#class MayBeDerivedError(Exception):
#    ''' raise when meta coord_attr cannot be found and derived_from exists '''
#    pass

class MetaPiece(ABC, ClassRegistry):
    ''' Metadata coders translate metadata into strings that can be used in the variable name.
        They must be able to encode the native metadata to a string with known max_length
        They must also be able to introspect and divide by metadata based on configuration
    '''

    @abstractmethod
    def encode(self, decoded):
        pass

    @property
    @abstractmethod
    def max_len(self):
        pass

    @property
    @abstractmethod
    def meta_name(self):
        pass

    @classmethod
    def coord_name(self, array: xr.DataArray):
        meta = meta_config[self.meta_name]
        if 'coord_attr' in meta:
            for k, v in meta['coord_attr'].items():
                try:
                    coord_name = array.camps.coord_name_from_attr_(k, v)
                    return coord_name
                except KeyError:  # occurrs when attr does not correspond to a coord
                    pass


#        if 'derived_from' in meta:
#            raise MayBeDerivedError('No multi-length coord found based on coord_attr; derived_from does exist; maybe it can be derived.')


    @classmethod
    def val_from_array_attr(self, array: xr.DataArray):
        ''' return the value from this piece of metadata if it is expressed via an attr;
            if it is not expressed as a value via an attribute, return None;
            this does not exclude this piece of metadata being expressed as a coord.
        '''
        meta = meta_config[self.meta_name]
        if 'non_coord_attr' in meta:
            if meta['non_coord_attr'] in array.attrs:
                return array.attrs[meta['non_coord_attr']]
        if 'depreciated_non_coord_attrs' in meta:
            attrs = meta['depreciated_non_coord_attrs']
            if isinstance(attrs, str):
                if attrs in array.attrs:
                    return array.attrs[attrs]
            else:
                for attr in attrs:
                    if attr in array.attrs:
                        return array.attrs[attrs]
        return None

    @classmethod
    def decoded_value(self, var) -> str:
        if isinstance(var, xr.DataArray):
            decoded_value = self.val_from_array_attr(var)
            if decoded_value is None:
                coord_name = self.coord_name(var)
                if not coord_name:
                    #raise ValueError('metadata issue with variable, it has no metadata for {self.meta_name}')
                    return None
                if var[coord_name].size != 1:
                    raise ValueError('coord array is not length one')
                decoded_value = var[coord_name].data[0]
            return decoded_value
        elif isinstance(var, camps.Variable):
            decoded_value = getattr(var, self.meta_name)
            print(f'decoded value: {decoded_value}')
            if decoded_value is None:
                return None
            if len(decoded_value) > 1:
                raise ValueError(f'Variable has more than one {self.meta_name}')
            return decoded_value[0]



#    def derive(self, array: xr.DataArray) -> xr.DataArray:
#        ''' optional method to create new DataArray with derived coords.
#            The derive method is needed when 'derived_from' is included for makeing meta
#        '''
#        pass

    @classmethod
    def select_one(self, array, coord_name, value) -> xr.DataArray:
        if 'non_coord_attr' in meta_config[self.meta_name]:
            # non coord attr is a switch to ommit this metapiece as a unit dimension
            non_coord_attr = meta_config[self.meta_name]['non_coord_attr']
            a = array.loc[{coord_name: value}].drop(coord_name)
            a.attrs[non_coord_attr] = str(value)
        else:
            # ommision of 'non_coord_attr' is switch to retain this metapiece as a unit dimension
            if self.meta_type == 'datetime':
                a = array.loc[{coord_name: pd.DatetimeIndex([value])}]
            else:
                a = array.loc[{coord_name: pd.Index([value])}]
        return a

    @classmethod
    def select(self, data: xr.DataArray, var: camps.Variable) -> xr.DataArray:
        dim_name = self.coord_name(data)
        if dim_name is None or dim_name not in data.dims:
            return data
        else:
            return data.loc[{dim_name: getattr(var,self.meta_name)}]

#    @abstractmethod
#    def array_has_more_than_one(self):
#        pass


meta_pieces = MetaPiece._class_registry


class Property(MetaPiece, default_name='observed_property'):
    # max len is relative for making name schemes, encoded string should not exceed max_len
    max_len = 40
    meta_name = 'observed_property'

    @classmethod
    def encode(self, decoded_str):
        print(decoded_str)
        # creating variable name relies on decoded components
        if len(decoded_str) > self.max_len:
            msg = '{self.meta_name} name may not exceed length {max_len}'
            raise ValueError(msg)
        return decoded_str



#class Smoothing(MetaPiece):
#    max_len = 4  # up to three characters after the prefix
#    prefix = 's' # the prefix is to assist human readability and identifying which parts of the name
#    meta_attr = smoothing
#    encode_mapping = UniqueValDict()
#    encode_mapping.update({'5_point':'1',
#                           '25_point':'2'})
#
#    @classmethod
#    def encode(self, decoded_str):
#        encoded = self.encode_mapping[decoded_str]
#        self.encoded.add(decoded_str)
#        return f'{self.prefix}{encoded}'
#
#
#class Duration(MetaPiece):
#    max_len = 4
#    prefix = 'd'
#    meta_attr = duration
#    encode_mapping = UniqueValDict()
#    encode_mapping.update({'1_hour':'1',
#                           '3_hour':'3'})
#
#    @classmethod
#    def encode(self, decoded_str):
#        return self.prefix + self.encode_mapping[decoded_str]

class ReferenceTime(MetaPiece, default_name='reference_time'):
    max_len = 12
    meta_name = 'reference_time'

    @classmethod
    def encode(self, time: datetime.time):
        time = pd.to_datetime(time)
        # store encoded as YYYYMMDDHHMMSS
        format = '%Y%m%d%H%M%S'
        return time.strftime(format)

#    def select_one(self, array, coord_name, value) -> xr.DataArray:
#        if 'non_coord_attr' in meta[self.meta_name]:
#            # non coord attr is a switch to ommit this metapiece as a unit dimension
#            non_coord_attr = meta[self.meta_name]
#            a = a.loc[{coord_name: value}]
#            a.attrs[non_coord_attr] = value
#        else:
#            # ommision of 'non_coord_attr' is switch to retain this metapiece as a unit dimension
#            if self.meta_type == 'datetime':
#                a = a.loc[{coord_name: pd.DatetimeIndex([value])}]
#            else:
#                a = a.loc[{coord_name: pd.Index([value])}]



class ReferenceTimeOfDay(MetaPiece, default_name='reference_time_of_day'):
    max_len = 6
    meta_name = 'reference_time_of_day'

    @classmethod
    def encode(self, time: datetime.time):
        # store encoded as seconds after 00 time
        td = timedelta(hours=time.hour, seconds=time.second)
        return self.prefix + str(td.seconds)

class Time(MetaPiece, default_name='time'):
    max_len = 12
    meta_name = 'time'

    @classmethod
    def encode(self, time: datetime.time):
        time = pd.to_datetime(time)
        # store encoded as YYYYMMDDHHMMSS
        format = '%Y%m%d%H%M%S'
        return time.strftime(format)

    # time could implement it's own special select method overiding the default one
    # I.E if time is a 2d coord, and thus not a dim, select it appropriately
    @classmethod
    def select(self, data: xr.DataArray, var: camps.Variable) -> xr.DataArray:
        dim_name = self.coord_name(data)
        if dim_name is not None and dim_name in data.dims:
            return data.loc[{dim_name: getattr(var,self.meta_name)}]

        reference_time_name = meta_pieces['reference_time'].coord_name(data)
        lead_time_name = meta_pieces['lead_time'].coord_name(data)
        if reference_time_name and lead_time_name:
            if 'time' not in data.coords:
                # trust that if time is already a coord that it is ref + lead
                data['time'] = data[reference_time_name] + data[lead_time_name]
                meta = meta_config[self.meta_name]
                data.time.attrs.update(meta['time']['coord_attr'])
            return data.where(data.time.isin(var.time), drop=True)

        return data

class LeadTime(MetaPiece, default_name='lead_time'):
    max_len = 12
    meta_name = 'lead_time'

    @classmethod
    def encode(self, time: datetime.timedelta):
        td = pd.to_timedelta(time)
        return str(td.seconds)

class Latitude(MetaPiece, default_name='latitude'):
    max_len = 6
    meta_name = 'latitude'

    @classmethod
    def encode(self, lat):
        return str(lat)

class Longitude(MetaPiece, default_name='longitude'):
    max_len = 7
    meta_name = 'longitude'

    @classmethod
    def encode(self, lon):
        return str(lon)

class X(MetaPiece, default_name='x'):
    max_len = 6
    meta_name = 'x'

    @classmethod
    def encode(self, x):
        raise ValueError('Do not use meta {self.meta_name} in name scheme')

class Y(MetaPiece, default_name='y'):
    max_len = 6
    meta_name = 'y'

    @classmethod
    def encode(self, y):
        raise ValueError('Do not use meta {self.meta_name} in name scheme')

class Z(MetaPiece, default_name='z'):
    max_len = 6
    meta_name = 'z'

    @classmethod
    def encode(self, y):
        raise NotImplementedError

#class TimeOfDay(MetaPiece, default_name='time_of_day'):
#    max_len = 6
#    prefix = 't'
#
#    @classmethod
#    def encode(self, time: datetime.time):
#        # store encoded as seconds after 00 time
#        td = timedelta(hours=time.hour, seconds=time.second)
#        return self.prefix + str(td.seconds)
