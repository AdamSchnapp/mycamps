from dataclasses import dataclass
from camps.coders import coders
''' names will implement name schemes.
    - Map data contruct and a scheme to  name: str
    - Implementations of schemes must be static
'''

def name(data, scheme) -> str:
    ''' return name based on data and scheme '''
    pass

MAX_NC_NAME = 256
MAX_BASE_NAME = 40
var_name_schemes = dict()

@dataclass
class VarNameScheme:
    version: str
    pieces: list

    def __post_init__(self):
        length = MAX_BASE_NAME
        for meta in self.pieces:
            if meta not in coders:
                raise ValueError(f'coder {meta} does not exist')
            length += coders[meta].max_len
            length += 1  # Seperator
        length += MAX_BASE_NAME
        if length > MAX_NC_NAME:
            raise ValueError(f'Cannot create {self.__class__.__name__} as '
                f'it would have max length of {length}, but cannot exceed {MAX_NC_NAME}')

        if self.version in var_name_schemes:
            raise ValueError(f'version "{self.version}" already exists')
        var_name_schemes[self.version] = self

VarNameScheme('1',['ReferenceTimeOfDay',
                   'Smoothing',
                   'Duration',])

def name_from_var_and_scheme(var, scheme):
    if var.name == 'U_wind_speed_instant': return 'Uwind_instant_500mb_00Z'
    if var.name == 'V_wind_speed_instant': return 'Vwind_instant_500mb_00Z'
    return var.name
