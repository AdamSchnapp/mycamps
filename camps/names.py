from dataclasses import dataclass, field
import camps
from camps.meta import meta, meta_pieces
import xarray as xr
from itertools import product
''' names will implement name schemes.
    - Map data contruct and a scheme to  name: str
    - Implementations of schemes must be static
'''

def name(data, scheme) -> str:
    ''' return name based on data and scheme '''
    pass

MAX_NC_NAME = 256
SPACE = ' '
SEPARATOR = '_'
var_name_schemes = dict()

@dataclass
class VarNameScheme:
    '''
    Store ordered pieces of name scheme as strings.
    Validate that associated pieces (encoded string returned from associated "meta_pieces") is <= the maximum.
    If a variable name piece does not have an associated "coder" raise an error
    If the total max length of all the pieces + the separators exceeds the total max length, raise an error
    pieces and variable name scheme are encoded as a xr.DataArray (self.var) for encoding to nc files.
    '''
    pieces: list
    sep: str = field(init=False, default=SEPARATOR)
    var: xr.DataArray = field(repr=False, init=False, default=None)

    def __post_init__(self):
        length = 0
        scheme = list()
        max_len = list()
        for meta in self.pieces:
            if meta not in meta_pieces:
                raise ValueError(f'coder {meta} does not exist')
            length += meta_pieces[meta].max_len
            max_len.append(meta_pieces[meta].max_len)
            scheme.append(meta)
            length += len(self.sep)  # Seperator

        if length > MAX_NC_NAME:
            raise ValueError(f'Cannot create {self.__class__.__name__} as '
                f'it would have max length of {length}, but cannot exceed {MAX_NC_NAME}')

        max_len_var = {'camps_name_scheme_component_max_length': max_len}
        self.var = xr.DataArray(scheme, name='camps_name_scheme', dims=max_len_var, coords=max_len_var)
        self.var.attrs['separator'] = self.sep

    @classmethod
    def from_dataset(cls, ds):
        if 'camps_name_scheme' not in ds:
            raise KeyError('dataset does not have a camps name scheme')
        return cls(list(ds['camps_name_scheme'].load().data))


    def to_netcdf(self, nc):
        self.var.to_netcdf(nc)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.pieces == other.pieces and self.sep == self.sep:
            print('schemes equall')
            return True

scheme = VarNameScheme(['observed_property'])

def name_from_var_and_scheme(var, scheme) -> str:
        name_pieces = list()
        for piece in scheme.pieces:
            meta_piece = meta_pieces[piece]
            decoded_value = meta_piece.decoded_value(var)
            if decoded_value is None:  # the data doesn't have this piece of meta, let be empty string
                encoded_value = '#'
            else:
                encoded_value = meta_piece.encode(decoded_value)
            encoded_value = encoded_value.replace(SEPARATOR, '') # we can modify the encoded value deterministically so long as we don't increase it's length
            name_pieces.append(encoded_value)
        name = SEPARATOR.join(name_pieces)
        print(f'name: {name}')
        return name
