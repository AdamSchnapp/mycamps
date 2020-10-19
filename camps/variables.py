from dataclasses import dataclass
import numpy as np
import xarray as xr
from numbers import Number
import pandas as pd
import datetime
import camps
from camps.registry import registry


class Validator:
    def __set_name__(self, owner, name):
        self.private_name = f'_{name}'

    def __get__(self, obj, objtype=None):
        try:
            value = getattr(obj, self.private_name)
        except AttributeError:
            value = None
        return value


class DatetimeIndex(Validator):

    def __set__(self, obj, value):
        if isinstance(value, pd.DatetimeIndex):
            pass
        elif isinstance(value, datetime.datetime):
            value = pd.DatetimeIndex(value)
        else:
            value = pd.date_range(*value)
        setattr(obj, self.private_name, value)


class TimedeltaIndex(Validator):

    def __set__(self, obj, value):
        if isinstance(value, pd.TimedeltaIndex):
            pass
        elif isinstance(value, datetime.timedelta):
            value = pd.TimedeltaIndex(value)
        else:
            value = pd.timedelta_range(*value)
        setattr(obj, self.private_name, value)

class OneDArray(Validator):

    def __set__(self, obj, value):
        if isinstance(value, np.ndarray):
            pass
        else:
            value = np.array(value)
        if value.ndim != 1:
            raise ValueError(f'{value} is not one dimension')
        setattr(obj, self.private_name, value)

@dataclass(init=False)
class Variable(dict):
    '''
    Variable instances are containers for variable metadata.
    '''

    name: str
    long_name: str = None
    standard_name: str = None
    data_type: str = None
    units: str = None
    valid_min: Number = None
    valid_max: Number = None
#    coordinate_variables: list = None
#    OM__observedProperty: str = None
#    SOSA__usedProcedure: list = None
    # possible dim variables are array dimensions when not part of variable name
    dims_possible = ['reference_time',
                    'lead_time',
                    'pressure',
                    'height',
                    'duration',
                    'threshold']
    reference_time: pd.DatetimeIndex = DatetimeIndex()
    cycle: Number = None
    time: pd.DatetimeIndex = DatetimeIndex()
    lead_time: pd.TimedeltaIndex = TimedeltaIndex()
    duration: pd.TimedeltaIndex = TimedeltaIndex()
    pressure: np.array = OneDArray()
    height: np.array = OneDArray()
    threshold: np.array = OneDArray()

    camps_multistep: camps.MultiStep = None  # this is the instruction set

    def __init__(self, name):
        super().__init__()
        self.__dict__ = self
        self.name = name

    @classmethod
    def from_registry(cls, reg_name: str):
        v = cls(reg_name)
        v.update(registry['variables'][reg_name])
        return v

    def __call__(self, data=None, *, datastore=None, in_handle=None, out_handle=None, out=False, **kwargs):
        ''' Variable behaves like an actor; call will yield lazy data, but does not consume data like other actors '''
        if data:
            raise ValueError('Variable calls do not take data')  # Variables behave as actors that don't take inputs

        if out:
            if not out_handle:
                out_handle = datastore.out_handle(self)
        if not in_handle:
            if datastore:
                in_handle = datastore.in_handle(self)
                print(in_handle)
            else:
                # case where no input provided; maybe can be created out of this air!
                pass  # invoke instruction to try creating without inputs
                # return data/access to lazy data

        options = dict()
        if 'chunk' in kwargs:
            options['chunks'] = kwargs['chunk']

        ds = xr.open_mfdataset(in_handle, concat_dim='default_time_coordinate_size', **options)
        return ds.camps[self]
