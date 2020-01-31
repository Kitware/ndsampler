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


def test_variable_backend():
    try:
        import gdal  # NOQA
    except ImportError:
        import pytest
        pytest.skip('cog requires gdal')
    import ndsampler
    import ubelt as ub
    dpath = ub.ensure_app_cache_dir('ndsampler/tests/test_variable_backend')
    ub.delete(dpath)
    ub.ensuredir(dpath)

    fnames = [
        'test_rgb_255.jpg',
        'test_gray_255.jpg',

        'test_rgb_255.png',
        'test_rgba_255.png',
        'test_gray_255.png',

        'test_rgb_01.tif', 'test_rgb_255.tif',
        'test_rgba_01.tif', 'test_rgba_255.tif',
        'test_gray_01.tif', 'test_gray_255.tif',
    ]
    import kwarray
    import kwimage
    fpaths = []
    rng = kwarray.ensure_rng(0)
    for fname in fnames:
        if 'rgb_' in fname:
            data = rng.rand(32, 32, 3)
        elif 'rgba_' in fname:
            data = rng.rand(32, 32, 4)
        elif 'gray_' in fname:
            data = rng.rand(32, 32, 1)

        if '01' in fname:
            pass
        elif '255' in fname:
            data = (data * 255).astype(np.uint8)

        fpath = join(dpath, fname)
        fpaths.append(fpath)
        kwimage.imwrite(fpath, data)

    for fpath in fpaths:
        data = kwimage.imread(fpath)
        print(ub.repr2({
            'fname': basename(fpath),
            'data.shape': data.shape,
            'data.dtype': data.dtype,
        }, nl=0))

    dset = ndsampler.CocoDataset.from_image_paths(fpaths)
    sampler = ndsampler.CocoSampler(dset, backend='cog')
    frames = sampler.frames
