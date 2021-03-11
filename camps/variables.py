#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np
import xarray as xr
from numbers import Number
import pandas as pd
import datetime
import camps
from camps.registry import registry
import logging

logger = logging.getLogger(__name__)


class Validator:
    def __set_name__(self, owner, name):
        self.private_name = f'_{name}'
        self.name = name

    def __get__(self, obj, objtype=None):
        try:
            value = getattr(obj, self.private_name)
        except AttributeError:
            value = None
        return value


class DatetimeIndex(Validator):

    def __set__(self, obj, value):
        '''
        >>> class Thing:
        ...     time = DatetimeIndex()
        >>> obj = Thing()
        >>> obj.time = '2020-01-01'
        >>> obj.time
        DatetimeIndex(['2020-01-01'], dtype='datetime64[ns]', freq=None)
        >>> obj.time = datetime.datetime(year=2020, month=5, day=5)
        >>> obj.time
        DatetimeIndex(['2020-05-05'], dtype='datetime64[ns]', freq=None)
        >>> obj.time = ["2020-01-01", "2020-01-02"]
        >>> obj.time
        DatetimeIndex(['2020-01-01', '2020-01-02'], dtype='datetime64[ns]', freq=None)
        >>> obj.time = [datetime.datetime(2020, 1, 1, 12), datetime.datetime(2020, 1, 2, 12)]
        >>> obj.time
        DatetimeIndex(['2020-01-01 12:00:00', '2020-01-02 12:00:00'], dtype='datetime64[ns]', freq=None)
        >>> obj.time = pd.date_range(start='2020-01-01', freq='1D', periods=3)
        >>> obj.time
        DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64[ns]', freq='D')
        >>> obj.time = {'start': '2020-01-01', 'end': '2020-01-01 06' , 'freq': '6H' }
        >>> obj.time
        DatetimeIndex(['2020-01-01 00:00:00', '2020-01-01 06:00:00'], dtype='datetime64[ns]', freq='6H')
        '''

        try:
            value = pd.DatetimeIndex(value)
        except TypeError:
            value = pd.DatetimeIndex([pd.to_datetime(value)])
        except ValueError:
            value = pd.date_range(**value)
        setattr(obj, self.private_name, value)
        obj.meta_filters_.append(self.name)


class TimedeltaIndex(Validator):

    def __set__(self, obj, value):
        '''
        >>> class Thing:
        ...     lead_time = TimedeltaIndex()
        >>> obj = Thing()
        >>> obj.lead_time = '1 day'
        >>> obj.lead_time
        Timedelta('1 days 00:00:00')
        >>> obj.lead_time = datetime.timedelta(minutes=15)
        >>> obj.lead_time
        Timedelta('0 days 00:15:00')
        >>> obj.lead_time = ['1 hours', '2 hour']
        >>> obj.lead_time
        TimedeltaIndex(['0 days 01:00:00', '0 days 02:00:00'], dtype='timedelta64[ns]', freq=None)
        >>> obj.lead_time = pd.timedelta_range(start=0, end='2 hour', freq='1H')
        >>> obj.lead_time
        TimedeltaIndex(['0 days 00:00:00', '0 days 01:00:00', '0 days 02:00:00'], dtype='timedelta64[ns]', freq='H')
        >>> obj.lead_time = {'start':'1 day', 'freq':'1D', 'periods':5}
        >>> obj.lead_time
        TimedeltaIndex(['1 days', '2 days', '3 days', '4 days', '5 days'], dtype='timedelta64[ns]', freq='D')
        '''

        try:
            value = pd.TimedeltaIndex(value)
        except TypeError:
            value = pd.TimedeltaIndex([pd.Timedelta(value)])
        except ValueError:
            value = pd.timedelta_range(**value)
        setattr(obj, self.private_name, value)
        obj.meta_filters_.append(self.name)


class PdIndex(Validator):

    def __set__(self, obj, value):
        try:
            value = pd.Index(value)
        except TypeError:
            value = pd.Index([value])
        value = pd.Index(value)
        setattr(obj, self.private_name, value)
        obj.meta_filters_.append(self.name)


@dataclass(init=False)
class Variable(dict):
    '''
    Variable instances are containers for variable metadata.
    '''

    name: str = None
    long_name: str = None
    standard_name: str = None
    data_type: str = None
    units: str = None
    valid_min: Number = None
    valid_max: Number = None

    x = PdIndex()
    y = PdIndex()
    z = PdIndex()
    reference_time: pd.DatetimeIndex = DatetimeIndex()
    reference_time_of_day: pd.TimedeltaIndex = TimedeltaIndex()
    lead_time: pd.TimedeltaIndex = TimedeltaIndex()
    time: pd.DatetimeIndex = DatetimeIndex()
    time_of_day: pd.TimedeltaIndex = TimedeltaIndex()

    duration: pd.TimedeltaIndex = TimedeltaIndex()
    pressure = PdIndex()
    height = PdIndex()
    x = PdIndex()
    y = PdIndex()
    z = PdIndex()
    latitude: pd.Index = PdIndex()
    longitude: pd.Index = PdIndex()
    threshold = PdIndex()
    observed_property: pd.Index = PdIndex()


    def __init__(self, meta_dict=None, *args, **kwargs):
        super().__init__()
        self.__dict__ = self
        self.meta_filters_ = list()  # keep track of which metadata types would have their filters used
        print(meta_dict)
        if meta_dict is not None:
            self.update(meta_dict)
        #self.name = name

    @classmethod
    def from_registry(cls, reg_name: str):
        v = cls(reg_name)
        v.update(registry['variables'][reg_name])
        return v

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        setattr(self, key, value)

    def update(self, other=None):
        if other is not None:
            for k, v in other.items():
                setattr(self, k, v)



    def __call__(self, data=None, *, in_handle=None, datastore=None, **kwargs):
        ''' Variable behaves like an actor; call will yield lazy data, but does not consume data like other actors '''
        if data:
            raise ValueError('Variable calls do not take data')  # Variables behave as actors that don't take inputs

        if not in_handle:
            if datastore:
                in_handle = datastore.in_handle(self)
                print(f'in_handle: {in_handle}')
            else:
                # case where no input provided; maybe can be created out of thin air!
                raise ValueError("in_handle can't be discovered")
                pass  # invoke instruction to try creating without inputs
                # return data/access to lazy data

        options = dict()
        if 'chunks' in in_handle:
            # pre open dataset to determine dims that are to be chunked
            #if isinstance(in_handle[files], str):
            #    one_file = in_handle[files]
            #pre = xr.open_dataset(one_file)
            #var_name = pre.camps.var_name(self)
            #in_handle['chunks'] = pre[var_name].camps.chunks_dict(chunks)
            #pre.close()
            # always use open_mfdataset when chunks included  # mf dataset seems to incur some overhead when accessing single files
            #if len(options['chunks']) == 0:
            #    logger.warning('Opening up dataset(s) as a dask array without chunking; things may go wrong or be slow')
            #ds = xr.open_mfdataset(**in_handle)
            raise NotImplementedError('turned off capability to chunk on variable call for now')

        else:
            # always use open_dataset when chunk ommited
            logger.info('Opening up dataset without chunking; computation might perfporm actively from here as data will be numpy arrays until chunked')
            ds = xr.open_dataset(**in_handle)
        a = ds.camps[self]  # select array of interest ; this applies filters/selection based on metadata attached to self

        return a


if __name__ == "__main__":
    import doctest
    doctest.testmod()
