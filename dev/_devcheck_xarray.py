def _devcheck_xarray():

    import xarray as xr
    import numpy as np
    data1 = xr.DataArray(np.random.rand(1, 10, 10, 3), dims=('t', 'y', 'x', 'c'), coords={'c': ['b1', 'b2', 'b3']})
    data2 = xr.DataArray(np.random.rand(1, 10, 10, 4), dims=('t', 'y', 'x', 'c'), coords={'c': ['b3', 'b4', 'b5', 'b6']})
    combo = xr.concat([data1, data2], dim='t')

    combo.coords
    # isinstance(combo, np.ndarray)
    # combo = xr.concat([data1.data, data2.data], dim=3)

    data1 = xr.DataArray(np.random.rand(1, 10, 10, 2), dims=('t', 'y', 'x', 'c'), coords={'c': ['b1', 'b3'], 't': [1]})
    data2 = xr.DataArray(np.random.rand(1, 10, 10, 2), dims=('t', 'y', 'x', 'c'), coords={'c': ['b5', 'b6'], 't': [2]})
    data3 = xr.DataArray(np.random.rand(1, 10, 10, 2), dims=('t', 'y', 'x', 'c'), coords={'c': ['b1', 'b7'], 't': [3]})
    combo = xr.concat([data1, data2, data3], dim='t')

    datas = [
        xr.DataArray(np.random.rand(*(1, 10, 10, 5)), dims=('t', 'y', 'x', 'c'), coords={'t': [1], 'c': ['wv1', 'wv2', 'wv3', 'wv4', 'wv5']}),
        xr.DataArray(np.random.rand(*(1, 10, 10, 1)), dims=('t', 'y', 'x', 'c'), coords={'t': [2], 'c': ['gray']}),
        xr.DataArray(np.random.rand(*(1, 10, 10, 0)), dims=('t', 'y', 'x', 'c'), coords={'t': [3], 'c': []}),
        xr.DataArray(np.random.rand(*(1, 10, 10, 5)), dims=('t', 'y', 'x', 'c'), coords={'t': [4], 'c': ['wv1', 'wv2', 'wv3', 'wv4', 'wv5']}),
        xr.DataArray(np.random.rand(*(1, 10, 10, 1)), dims=('t', 'y', 'x', 'c'), coords={'t': [5], 'c': ['gray']}),
    ]
    combo = xr.concat(datas, dim='t')
    assert np.all(np.isnan(combo.values[2]))

    a = xr.DataArray(np.random.rand(5, 8, 3, 2), dims=('a', 'b', 'c', 'd'), coords={'c': ['z', 'x', 'y']})
    a.sel(c=['x', 'y', 'z'])
