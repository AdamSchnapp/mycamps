import xarray as xr
from camps.actors import to_stations
import numpy as np
import pandas as pd
from io import StringIO

def test_basic():
    ds = xr.tutorial.open_dataset('air_temperature')

    stations = """
    platform_id lat        lon
    KBWI        39.1833    -76.6667
    KAVL        35.4333    -82.5500
    KDAB        29.1833    -81.0500
    KSEA        47.4500    -122.300
    PADK        51.8833    -176.6500
    """
    stations = pd.read_csv(StringIO(stations), delim_whitespace=True)

    station_data = to_stations(ds.air, stations=stations)
    assert station_data.dims == ('time', 'station')
    assert station_data.shape == (2920, 5)

    expected_at_2013010100 = np.array([274.9, 279. , 290.5, 273.4, np.nan], dtype=np.float32)
    np.testing.assert_array_equal(station_data.sel(time='2013-1-1 00').data, expected_at_2013010100)

    expected_lon = stations.lon.to_numpy()
    np.testing.assert_array_equal(station_data.camps.longitude.data, expected_lon)




