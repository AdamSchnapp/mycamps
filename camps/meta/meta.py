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
        ''' return the name of the coord describing this type of metadata '''
        meta = meta_config[self.meta_name]
        if 'coord_attr' in meta:
            for k, v in meta['coord_attr'].items():
                try:
                    coord_name = array.camps.coord_name_from_attr_(k, v)
                    return coord_name
                except KeyError:  # occurrs when attr does not correspond to a coord
                    pass  # let return be None when there isn't one


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
        ''' given a blob of metadata as an xr.DataArray or camps.Variable determine the value of this piece of metadata
            Raise an error if the metablob expresses more than one of this type. (this meta type is a coord of len > 1)'''
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


    @classmethod
    def select_one(self, array, coord_name, value) -> xr.DataArray:
        ''' select a unit slice of the metadata type and arrange attrs/coords based on meta config '''
        if 'non_coord_attr' in meta_config[self.meta_name]:
            # non coord attr is a switch to ommit this metapiece as a unit dimension
            non_coord_attr = meta_config[self.meta_name]['non_coord_attr']
            a = array.loc[{coord_name: value}].drop(coord_name)
            a.attrs[non_coord_attr] = str(value)
        else:
            # ommision of 'non_coord_attr' is switch to retain this metapiece as a unit dimension
            a = array.loc[{coord_name: pd.Index([value])}]
        return a

    @classmethod
    def select(self, data: xr.DataArray, var: camps.Variable) -> xr.DataArray:
        ''' select this metadata types values from data '''
        dim_name = self.coord_name(data)
        if dim_name is None or dim_name not in data.dims:
            return data
        else:
            return data.loc[{dim_name: getattr(var,self.meta_name)}]

    @classmethod
    def split_array(self, array) -> list:
        ''' split array based on meta piece '''
        coord_name = self.coord_name(array)
        return_arrays = list()
        if coord_name:
            for v in array[coord_name].data:
                a = self.select_one(array, coord_name, v)
                return_arrays.append(a)
        else:
            return_arrays.append(array)
        return return_arrays


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
    max_len = 8
    meta_name = 'reference_time_of_day'

    @classmethod
    def encode(self, timedelta: datetime.timedelta) -> str:
        timedelta = pd.Timedelta(timedelta)
        return str(timedelta.seconds) + 's'

    # reference time of day is cycle time; it is not explicitly a dimension of an array,
    # It may be a coordinate variable that decribes the times that exist in reference_time; and is thus derived metadata
    # It may be used to filter reference_time
    @classmethod
    def select(self, data: xr.DataArray, var: camps.Variable) -> xr.DataArray:
        reference_time_name = meta_pieces['reference_time'].coord_name(data)
        if reference_time_name is None:
            raise ValueError('No reference_time metadata, therefore cannot filter by {self.meta_name}')
        reference_time = pd.DatetimeIndex(data.camps.reference_time.to_series())
        ref_day = reference_time.floor('d')
        ref_time_as_td = reference_time - ref_day
        refs_select = reference_time[ref_time_as_td.isin(var.reference_time_of_day)]
        return data.loc[{reference_time_name: refs_select}]

    @classmethod
    def select_one(self, array, coord_name, value) -> xr.DataArray:
        ''' selecting one reference time of day does not eleminate an actual dim;
            just add the non_coord_attr if it is provided
        '''
        a = array.loc[{coord_name: value}]
        if 'non_coord_attr' in meta_config[self.meta_name]:
            non_coord_attr = meta_config[self.meta_name]['non_coord_attr']
            print(non_coord_attr)
            a.attrs[non_coord_attr] = str(value)
        return a

    @classmethod
    def val_from_array_attr(self, array: xr.DataArray):
        ''' for forecast_reference_time_of_day return timedelta derived from reference_time if only one reference_time exists
            do not actually reference variable attributes as they do not support
            return the value from this piece of metadata if it is expressed via an attr;
            if it is not expressed as a value via an attribute, return None;
            this does not exclude this piece of metadata being expressed as a coord.
        '''
        ref_times_of_day = array.camps.reference_time - array.camps.reference_time.dt.floor('d')
        n_unique_time = len(np.unique(ref_times_of_day))
        if n_unique_time > 1:
            raise ValueError('More than one reference_time_of_day in array')
        elif n_unique_time == 1:
            return ref_times_of_day.data[0]
        return None

    @classmethod
    def split_array(self, array) -> list:
        ''' split array based on meta piece '''
        reference_time_name = meta_pieces['reference_time'].coord_name(array)
        if reference_time_name is None:
            return [array]
        reference_time = pd.DatetimeIndex(array.camps.reference_time.to_series())
        ref_day = reference_time.floor('d')
        ref_times_as_td = reference_time - ref_day
        unique_times = ref_times_as_td.unique()
        return_arrays = list()
        for td in unique_times:
            print(td)
            #sel = reference_time[ref_times_as_td == td]
            time = (datetime.datetime.min + td).time()  # this converts a timedelta to a datetime.time
            a = self.select_one(array, reference_time_name, time)
            return_arrays.append(a)
        return return_arrays


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
