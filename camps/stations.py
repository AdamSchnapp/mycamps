import logging
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

def south_to_negative(lat):
    if 'N' in lat:
        lat = lat.strip('N')
        return float(lat)
    elif 'S' in lat:
        lat = lat.strip('S')
        return float(lat) * -1

def west_to_negative(lon):
    if 'E' in lon:
        lon = lon.strip('E')
        return float(lon)
    elif 'W' in lon:
        lon = lon.strip('W')
        return float(lon) * -1

def to_degrees_east(lon):
    if 'E' in lon:
        lon = lon.strip('E')
        return float(lon)
    elif 'W' in lon:
        lon = lon.strip('W')
        return 360.0 - (float(lon))

def from_mos2ktbl(tbl):
    df = pd.read_csv(tbl, sep=':', usecols=[0,1,2,3,6,7,18],
            names=['call','link','name','state','lat','lon','comment'], quoting=3)
            # quoting = 3 prevents unclosed quotes from blocking parse on sep and \n
    df['call'] = df['call'].str.strip()
    df['lat'] = df['lat'].apply(south_to_negative)
    df['lon'] = df['lon'].apply(west_to_negative)
    return df

core30 = ['KATL','KLAX','KORD','KDFW','KDEN','KJFK','KSFO','KSEA','KLAS','KMCO',
          'KEWR','KCLT','KPHX','KIAH','KMIA','KBOS','KMSP','KFLL','KDTW','KPHL',
          'KLGA','KBWI','KSLC','KSAN','KIAD','KDCA','KMDW','KTPA','KPDX','PHNL']


#class Stations:
##    ''' station/point data container '''
##    all : pd.DataFrame
#
##    def __init__(self, df):
##        if ('call' not in df) or ('lat' not in df) or ('lon' not in df):
##            raise ValueError(f'call lat or lon are missing from {df}')
##        self.all = df.set_index('call')
#
#    @classmethod
#    def from_mos2ktbl(cls, tbl):
#        df = pd.read_csv(tbl, sep=':', usecols=[0,1,2,3,6,7,18],
#                names=['call','link','name','state','lat','lon','comment'], quoting=3)
#                # quoting = 3 prevents unclosed quotes from blocking parse on sep and \n
#        df['call'] = df['call'].str.strip()
#        df['lat'] = df['lat'].apply(south_to_negative)
#        df['lon'] = df['lon'].apply(west_to_negative)
#        return df


