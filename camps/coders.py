from camps.helpers import UniqueValDict, ClassRegistry, removeprefix
from abc import ABC, abstractmethod
import datetime


class MetaCoder(ABC, ClassRegistry):
    ''' Metadata coders translate metadata into strings that can be used in the variable name.
        They must be able to encode and decode the native metadata type to a string with known max_length
        and unique character prefix'''

    @abstractmethod
    def encode(self, decoded_str):
        pass

    @property
    @abstractmethod
    def max_len(self):
        pass

    @property
    @abstractmethod
    def prefix(self):
        pass

coders = MetaCoder._class_registry

class Smoothing(MetaCoder):
    max_len = 4  # up to three characters after the prefix, a letter cannot be used in the second position to maintain prefix
    prefix = 's'
    encode_mapping = UniqueValDict()
    encode_mapping.update({'5_point':'1',
                           '25_point':'2'})
    encoded = set()

    @classmethod
    def encode(self, decoded_str):
        encoded = self.encode_mapping[decoded_str]
        self.encoded.add(decoded_str)
        return f'{self.prefix}{encoded}'


class Duration(MetaCoder):
    max_len = 4
    prefix = 'd'
    encode_mapping = UniqueValDict()
    encode_mapping.update({'1_hour':'1',
                           '3_hour':'3'})
    decode_mapping = encode_mapping.flip_key_val()

    @classmethod
    def encode(self, decoded_str):
        return self.prefix + self.encode_mapping[decoded_str]

class ReferenceTimeOfDay(MetaCoder):
    max_len = 6
    prefix = 't'

    @classmethod
    def encode(self, time: datetime.time):
        # store encoded as seconds after 00 time
        td = timedelta(hours=time.hour, seconds=time.second)
        return self.prefix + str(td.seconds)
