#!/usr/bin/env python3
import re
from io import StringIO
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

def date_from_sfctbl(sfctbl):
    yyyymmdd = re.compile(r'\d{8}')

    try:
        with open(sfctbl) as f:
            line0 = f.readline()
    except TypeError:
        line0 = sfctbl.readline()

    date = pd.to_datetime(yyyymmdd.search(line0).group(0))
    return date


def read_sfctbl(sfctbl):
    ''' return dataframe of observational data from mos2k sfctbl'''

    columns_and_names = '''
    col_ix, col_name
    0,      call
    1,      type
    2,      lat
    3,      lon
    4,      time
    5,      tmp
    6,      dew
    10,     vis
    11,     wdr
    12,     wsp
    13,     gst
    14,     msl
    16,     ca1
    17,     ch1
    18,     ca2
    19,     ch2
    20,     ca3
    21,     ch3
    22,     ca4
    23,     ch4
    24,     ca5
    25,     ch5
    '''
    cols_and_names = pd.read_csv(StringIO(columns_and_names), skipinitialspace=True)
    usecols = cols_and_names['col_ix']
    names = cols_and_names['col_name']
    date = date_from_sfctbl(sfctbl)
    df = pd.read_csv(sfctbl, sep=':', skiprows=2, usecols=usecols, names=names)
    df = df.dropna()

    # orient time into hour and minute timedeltas from previous day and hour
    hour = pd.to_timedelta(df.time // 100, unit='h')
    minute = pd.to_timedelta(df.time % 100, unit='m')
    # make datetime field
    df['time']  = date + hour + minute

    # convert numeric fields to numbers
    to_numeric = ['tmp','dew','vis','wdr','wsp','gst','msl','ch1','ch2','ch3','ch4','ch5']
    for elm in to_numeric:
        df[elm] = pd.to_numeric(df[elm], errors='coerce')

    # strip white space from string fields and make empty string nan
    strip = ['call','type','ca1','ca2','ca3','ca4','ca5']
    for field in strip:
        df[field] = df[field].str.strip()
    df = df.replace('', np.nan)

    # type categorical fields as categories
    df["type"] = df["type"].astype("category")

    sky_cat = CategoricalDtype(categories=["SKC", "CLR", "FEW", "SCT", "BKN", "OVC", "VV"], ordered=True)
    sky_mapping = {'SKC':0,'CLR':0, 'FEW':0.1875, 'SCT':0.4375, 'BKN':0.75, 'OVC':1, 'VV':1}
    cloud_layers = ['ca1', 'ca2', 'ca3', 'ca4', 'ca5']
    for layer in cloud_layers:
        df[layer] = df[layer].astype(sky_cat)
        df[f"{layer}_numeric"] = pd.to_numeric(df[layer].replace(sky_mapping))

    return df







