from os.path import basename
from os.path import join
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
    """
    CommandLine:
        xdoctest -m $HOME/code/ndsampler/tests/tests_frame_backends.py test_variable_backend
        --debug-validate-cog --debug-load-cog
    """
    try:
        import gdal  # NOQA
    except ImportError:
        import pytest
        pytest.skip('cog requires gdal')
    import ndsampler
    import kwcoco
    import ubelt as ub
    dpath = ub.Path.appdir('ndsampler/tests/test_variable_backend').ensuredir()
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

    h, w = 1200, 1055

    for fname in ub.ProgIter(fnames, desc='create test data'):
        if 'rgb_' in fname:
            data = rng.rand(h, w, 3)
        elif 'rgba_' in fname:
            data = rng.rand(h, w, 4)
        elif 'gray_' in fname:
            data = rng.rand(h, w, 1)

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

    dset = kwcoco.CocoDataset.from_image_paths(fpaths)
    sampler = ndsampler.CocoSampler(dset, backend='cog', workdir=dpath)
    frames = sampler.frames
    # frames.prepare()

    if 1:
        for gid in frames.image_ids:
            print('======== < START IMAGE ID > ===============')
            gpath, cache_gpath = frames._gnames(gid)
            print('gpath = {!r}'.format(gpath))
            print('cache_gpath = {!r}'.format(cache_gpath))

            image = frames.load_image(gid)
            print('image = {!r}'.format(image))
            print('image.shape = {!r}'.format(image.shape))
            print('image.dtype = {!r}'.format(image.dtype))

            subdata = image[0:8, 0:8]
            print('subdata.dtype = {!r}'.format(subdata.dtype))
            print('subdata.shape = {!r}'.format(subdata.shape))
            # info = ub.cmd('gdalinfo ' + cache_gpath, verbose=0)
            # print('GDAL INFO:')
            # print(ub.indent(info['out']))
            # assert info['ret'] == 0

            # dataset = gdal.OpenEx(cache_gpath)
            dataset = gdal.Open(cache_gpath, gdal.GA_ReadOnly)
            md = dataset.GetMetadata('IMAGE_STRUCTURE')
            print('md = {!r}'.format(md))
            # Use dict.get method in case the metadata dict does not have a 'COMPRESSION' key
            compression = md.get('COMPRESSION', 'RAW')
            print('compression = {!r}'.format(compression))

            print('======== < END IMAGE ID > ===============')
