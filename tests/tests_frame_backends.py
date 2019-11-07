import ndsampler
import numpy as np
import ubelt as ub


def test_None_backend():
    dset = ndsampler.CocoSampler.demo('shapes27', backend=None)
    assert dset.frames._backend['type'] is None
    raw_img = dset.load_image(1)
    assert isinstance(raw_img, np.ndarray)
    assert dset.frames.cache_dpath is None


def test_npy_backend():
    dset = ndsampler.CocoSampler.demo('shapes11', backend='npy')
    assert dset.frames._backend['type'] == 'npy'
    assert dset.frames.cache_dpath is not None
    ub.delete(dset.frames.cache_dpath)
    ub.ensuredir(dset.frames.cache_dpath)
    raw_img = dset.load_image(1)
    assert isinstance(raw_img, np.memmap)


def test_cog_backend():
    try:
        import gdal  # NOQA
    except ImportError:
        import pytest
        pytest.skip('cog requires gdal')

    dset = ndsampler.CocoSampler.demo('shapes13', backend='cog')
    assert dset.frames._backend['type'] == 'cog'
    assert dset.frames.cache_dpath is not None
    ub.delete(dset.frames.cache_dpath)
    ub.ensuredir(dset.frames.cache_dpath)
    raw_img = dset.load_image(5)
    assert not isinstance(raw_img, np.ndarray)
