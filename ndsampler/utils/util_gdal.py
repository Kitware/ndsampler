from os.path import exists
import numpy as np
import ubelt as ub


try:
    import xdev
    profile = xdev.profile
except ImportError:
    profile = ub.identity


def have_gdal():
    try:
        import gdal
    except ImportError:
        return False
    else:
        return gdal is not None


def _fix_conda_gdal_hack():
    import os
    import gdal  # NOQA
    if os.pathsep != ';' and ';' in os.environ["PATH"]:
        # Workaround conda gdal issue
        # https://github.com/OSGeo/gdal/issues/1231
        # https://github.com/conda-forge/gdal-feedstock/issues/259
        os.environ["PATH"] = os.pathsep.join(os.environ["PATH"].split(';'))


def _doctest_check_cog(data, fpath):
    from ndsampler.utils.validate_cog import validate as _validate_cog
    import kwimage
    print('---- CHECK COG ---')
    print('fpath = {!r}'.format(fpath))
    disk_shape = kwimage.imread(fpath).shape
    errors = _validate_cog(fpath)
    warnings, errors, details = _validate_cog(fpath)
    print('disk_shape = {!r}'.format(disk_shape))
    print('details = ' + ub.repr2(details, nl=2))
    print('warnings = ' + ub.repr2(warnings, nl=1))
    print('errors = ' + ub.repr2(errors, nl=1))
    passed = not errors

    assert data.shape == disk_shape
    return passed


def _numpy_to_gdal_dtype(numpy_dtype):
    import gdal
    kindsize = (numpy_dtype.kind, numpy_dtype.itemsize)
    if kindsize == ('u', 1):
        eType = gdal.GDT_Byte
    elif kindsize == ('u', 2):
        eType = gdal.GDT_UInt16
    elif kindsize == ('i', 2):
        eType = gdal.GDT_Int16
    elif kindsize == ('f', 4):
        eType = gdal.GDT_Float32
    elif kindsize == ('f', 8):
        eType = gdal.GDT_Float64
    else:
        raise TypeError('Unsupported GDAL dtype for {}'.format(kindsize))
    return eType


def _benchmark_cog_conversions():
    """
    CommandLine:
        xdoctest -m ~/code/ndsampler/ndsampler/utils/util_gdal.py _benchmark_cog_conversions
    """
    # Benchmark
    # xdoc: +REQUIRES(--bench)
    from ndsampler.utils.validate_cog import validate
    import xdev
    import kwimage
    # Prepare test data
    shape = (8000, 8000, 1)
    print('Test data shape = {!r}'.format(shape))
    data = np.random.randint(0, 255, shape, dtype=np.uint16)
    print('Test data size = {}'.format(xdev.byte_str(data.size * data.dtype.itemsize)))
    src_fpath = ub.expandpath('~/raid/src.png')
    kwimage.imwrite(src_fpath, data)

    # Benchmark conversions
    dst_api_fpath = ub.expandpath('~/raid/dst_api.tiff')
    dst_cli_fpath = ub.expandpath('~/raid/dst_cli.tiff')
    dst_data_fpath = ub.expandpath('~/raid/dst_data.tiff')

    ti = ub.Timerit(3, bestof=3, verbose=3, unit='s')

    compress = 'RAW'
    compress = 'DEFLATE'
    blocksize = 256

    if 1:

        for timer in ti.reset('cov-convert-data'):
            ub.delete(dst_data_fpath)
            with timer:
                _imwrite_cloud_optimized_geotiff(dst_data_fpath, data, compress=compress, blocksize=blocksize)
        assert not len(validate(dst_data_fpath)[1])

        for timer in ti.reset('cog-convert-api2'):
            ub.delete(dst_api_fpath)
            with timer:
                _api_convert_cloud_optimized_geotiff2(src_fpath, dst_api_fpath, compress=compress, blocksize=blocksize)
        assert not len(validate(dst_api_fpath)[1])

        for timer in ti.reset('cog-convert-api'):
            ub.delete(dst_api_fpath)
            with timer:
                _api_convert_cloud_optimized_geotiff(src_fpath, dst_api_fpath, compress=compress, blocksize=blocksize)
        assert not len(validate(dst_api_fpath)[1])

    for timer in ti.reset('cog-convert-cli'):
        ub.delete(dst_cli_fpath)
        with timer:
            _cli_convert_cloud_optimized_geotiff(src_fpath, dst_cli_fpath, compress=compress, blocksize=blocksize)
    assert not len(validate(dst_data_fpath)[1])

    if ub.find_exe('cog'):
        # requires pip install cogeotiff
        for timer in ti.reset('cogeotiff cli --compress {}'.format(compress)):
            ub.delete(dst_cli_fpath)
            with timer:
                info = ub.cmd('cog create {} {} --compress {} --block-size {}'.format(src_fpath, dst_cli_fpath, compress, blocksize), verbose=0)
                assert info['ret'] == 0
        assert not len(validate(dst_data_fpath)[1])


@profile
def _imwrite_cloud_optimized_geotiff(fpath, data, compress='auto',
                                     blocksize=256):
    """
    Args:
        fpath (PathLike): file path to save the COG to.

        data (ndarray[ndim=3]): Raw HWC image data to save. Dimensions should
            be height, width, channels.

        compress (bool, default='auto'): Can be JPEG (lossy) or LZW (lossless),
            or DEFLATE (lossless). Can also be 'auto', which will try to
            hueristically choose a sensible choice.

        blocksize (int, default=256): size of tiled blocks

    References:
        https://geoexamples.com/other/2019/02/08/cog-tutorial.html#create-a-cog-using-gdal-python
        http://osgeo-org.1560.x6.nabble.com/gdal-dev-Creating-Cloud-Optimized-GeoTIFFs-td5320101.html
        https://gdal.org/drivers/raster/cog.html
        https://github.com/harshurampur/Geotiff-conversion
        https://github.com/sshuair/cogeotiff
        https://github.com/cogeotiff/rio-cogeo

    Notes:
        conda install gdal

        OR

        sudo apt install gdal-dev
        pip install gdal ---with special flags, forgot which though, sry

    CommandLine:
        xdoctest -m ndsampler.utils.util_gdal _imwrite_cloud_optimized_geotiff

    Example:
        >>> # xdoctest: +REQUIRES(module:gdal)
        >>> from ndsampler.utils.util_gdal import *  # NOQA
        >>> from ndsampler.utils.util_gdal import _imwrite_cloud_optimized_geotiff

        >>> data = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
        >>> fpath = '/tmp/foo.cog.tiff'
        >>> compress = 'JPEG'
        >>> _imwrite_cloud_optimized_geotiff(fpath, data, compress='JPEG')
        >>> assert _doctest_check_cog(data, fpath)
        >>> _imwrite_cloud_optimized_geotiff(fpath, data, compress='LZW')
        >>> assert _doctest_check_cog(data, fpath)

        >>> data = (np.random.rand(100, 100, 4) * 255).astype(np.uint8)
        >>> _imwrite_cloud_optimized_geotiff(fpath, data, compress='JPEG')
        >>> assert _doctest_check_cog(data, fpath)
        >>> _imwrite_cloud_optimized_geotiff(fpath, data, compress='LZW')
        >>> assert _doctest_check_cog(data, fpath)
        >>> _imwrite_cloud_optimized_geotiff(fpath, data, compress='DEFLATE')
        >>> assert _doctest_check_cog(data, fpath)

        >>> data = (np.random.rand(100, 100, 5) * 255).astype(np.uint8)
        >>> _imwrite_cloud_optimized_geotiff(fpath, data, compress='LZW')
        >>> assert _doctest_check_cog(data, fpath)

    Ignore:
        >>> import xdev
        >>> data = np.random.randint(0, 255, (8000, 8000, 3), dtype=np.uint8)
        >>> print(xdev.byte_str(data.size * data.dtype.itemsize))
        >>> fpath = fpath1 = ub.expandpath('~/ssd/foo.tiff')
        >>> fpath2 = ub.expandpath('~/raid/foo.tiff')
        >>> import ubelt as ub
        >>> ti = ub.Timerit(1, bestof=1, verbose=3)
        >>> #
        >>> for timer in ti.reset('SSD'):
        >>>     ub.delete(fpath1)
        >>>     with timer:
        >>>         _imwrite_cloud_optimized_geotiff(fpath1, data)
        >>> for timer in ti.reset('HDD'):
        >>>     ub.delete(fpath2)
        >>>     with timer:
        >>>         _imwrite_cloud_optimized_geotiff(fpath2, data)
    """
    import gdal
    if len(data.shape) == 2:
        data = data[:, :, None]

    y_size, x_size, num_bands = data.shape

    data_set = None
    if compress == 'auto':
        compress = _auto_compress(data=data)

    if compress not in ['JPEG', 'LZW', 'DEFLATE', 'RAW']:
        raise KeyError('unknown compress={}'.format(compress))

    if compress == 'JPEG' and num_bands >= 5:
        raise ValueError('Cannot use JPEG with more than 4 channels (got {})'.format(num_bands))

    eType = _numpy_to_gdal_dtype(data.dtype)
    if compress == 'JPEG':
        if eType not in [gdal.GDT_Byte, gdal.GDT_UInt16]:
            raise ValueError('JPEG compression must use 8 or 16 bit integer imagery')

    # TODO: optional RESAMPLING option
    # TODO: option to specify num levels
    # NEAREST/AVERAGE/BILINEAR/CUBIC/CUBICSPLINE/LANCZOS
    overview_resample = 'NEAREST'
    # overviewlist = [2, 4, 8, 16, 32, 64]
    overviewlist = []

    options = [
        'TILED=YES',
        'BIGTIFF=YES',
        'BLOCKXSIZE={}'.format(blocksize),
        'BLOCKYSIZE={}'.format(blocksize),
    ]
    if compress != 'RAW':
        options += ['COMPRESS={}'.format(compress)]
    if compress == 'JPEG' and num_bands == 3:
        # Using YCBCR speeds up jpeg compression by quite a bit
        options += ['PHOTOMETRIC=YCBCR']

    options = list(map(str, options))  # python2.7 support
    if overviewlist:
        options.append('COPY_SRC_OVERVIEWS=YES')

    # Create an in-memory dataset where we will prepare the COG data structure
    driver = gdal.GetDriverByName(str('MEM'))
    data_set = driver.Create(str(''), x_size, y_size, num_bands, eType=eType)
    for i in range(num_bands):
        band_data = np.ascontiguousarray(data[:, :, i])
        data_set.GetRasterBand(i + 1).WriteArray(band_data)

    if overviewlist:
        # Build the downsampled overviews (for fast zoom in / out)
        data_set.BuildOverviews(str(overview_resample), overviewlist)

    driver = None
    # Copy the in-memory dataset to an on-disk GeoTiff
    driver2 = gdal.GetDriverByName(str('GTiff'))
    data_set2 = driver2.CreateCopy(fpath, data_set, options=options)
    data_set = None

    # OK, so setting things to None turns out to be important. Gah!
    data_set2.FlushCache()

    # Dereference everything
    data_set2 = None
    driver2 = None
    return fpath


def _dtype_equality(dtype1, dtype2):
    """
    Check for numpy dtype equality

    References:
        https://stackoverflow.com/questions/26921836/correct-way-to-test-for-numpy-dtype

    Example:
        dtype1 = np.empty(0, dtype=np.uint8).dtype
        dtype2 = np.uint8
        _dtype_equality(dtype1, dtype2)
    """
    dtype1_ = getattr(dtype1, 'type', dtype1)
    dtype2_ = getattr(dtype2, 'type', dtype2)
    return dtype1_ == dtype2_


def _auto_compress(src_fpath=None, data=None, data_set=None):
    """
    Heuristic for automatically choosing compression type

    Args:
        src_fpath (str): path to source image if known
        data (ndarray): data pixels if known
        data_set (gdal.Dataset): gdal dataset if known

    Returns:
        str: gdal compression code

    Example:
        >>> assert _auto_compress(src_fpath='foo.jpg') == 'JPEG'
        >>> assert _auto_compress(src_fpath='foo.png') == 'LZW'
        >>> assert _auto_compress(data=np.random.rand(3, 2)) == 'RAW'
        >>> assert _auto_compress(data=np.random.rand(3, 2, 3).astype(np.uint8)) == 'JPEG'
        >>> assert _auto_compress(data=np.random.rand(3, 2, 4).astype(np.uint8)) == 'RAW'
        >>> assert _auto_compress(data=np.random.rand(3, 2, 1).astype(np.uint8)) == 'RAW'
    """
    compress = None
    num_channels = None
    dtype = None

    if src_fpath is not None:
        # the filepath might hint at which compress method is best.
        ext = src_fpath[-5:].lower()
        if ext.endswith(('.jpg', '.jpeg')):
            compress = 'JPEG'
        elif ext.endswith(('.png', '.png')):
            compress = 'LZW'

    if compress is None:

        if data_set is not None:
            if dtype is None:
                main_band = data_set.GetRasterBand(1)
                dtype = _GDAL_DTYPE_LUT[main_band.DataType]

            if num_channels is None:
                data_set.RasterCount == 3

        elif data is not None:
            if dtype is None:
                dtype = data.dtype

            if num_channels is None:
                if len(data.shape) == 3:
                    num_channels = data.shape[2]

    if compress is None:
        if _dtype_equality(dtype, np.uint8) and num_channels == 3:
            compress = 'JPEG'

    if compress is None:
        # which backend is best in this case?
        compress = 'RAW'
    return compress


def _cli_convert_cloud_optimized_geotiff(src_fpath, dst_fpath, compress='auto',
                                         blocksize=256):
    """
    For whatever reason using the CLI seems to simply be faster.

    Args:
        src_fpath (PathLike): file path to convert

        dst_fpath (PathLike): file path to save the COG to.

        blocksize (int, default=256): size of tiled blocks

        compress (bool, default='auto'): Can be JPEG (lossy) or LZW (lossless),
            or DEFLATE (lossless). Can also be 'auto', which will try to
            hueristically choose a sensible choice.

    Ignore:
        >>> import xdev
        >>> import kwimage
        >>> data = np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8)
        >>> print(xdev.byte_str(data.size * data.dtype.itemsize))
        >>> src_fpath = ub.expandpath('~/raid/src.tiff')
        >>> kwimage.imwrite(src_fpath, data)
        >>> dst_fpath = ub.expandpath('~/raid/dst.tiff')
        >>> ti = ub.Timerit(1, bestof=1, verbose=3)
        >>> for timer in ti.reset('SSD-api'):
        >>>     _cli_convert_cloud_optimized_geotiff(src_fpath, dst_fpath)
    """
    import gdal
    options = [
        'TILED=YES',
        'BIGTIFF=YES',
        'BLOCKXSIZE={}'.format(blocksize),
        'BLOCKYSIZE={}'.format(blocksize),
    ]

    data_set = None
    if compress == 'auto':
        data_set = gdal.Open(src_fpath, gdal.GA_ReadOnly)
        compress = _auto_compress(src_fpath=src_fpath, data_set=data_set)

    if compress != 'RAW':
        options += ['COMPRESS={}'.format(compress)]

    if compress == 'JPEG':
        data_set = data_set or gdal.Open(src_fpath, gdal.GA_ReadOnly)
        if data_set.RasterCount == 3:
            # Using YCBCR speeds up jpeg compression by quite a bit
            options += ['PHOTOMETRIC=YCBCR']

    overview_resample = 'NEAREST'

    # TODO: overview levels
    overviewlist = []
    temp_fpath = src_fpath
    if overviewlist:
        level_str = ' '.join(list(map(str, overviewlist)))
        info = ub.cmd('gdaladdo -r {} {} {}'.format(overview_resample, temp_fpath, level_str))
        if info['ret'] != 0:
            print(info['out'])
            print(info['err'])
            raise Exception('failed {}'.format(info))
        options.append('COPY_SRC_OVERVIEWS=YES')

    option_strs = ['-co {}'.format(o) for o in options]

    # Not sure if necessary
    option_strs = ['-of GTiff'] + option_strs
    # option_strs = ['-a_nodata None'] + option_strs

    tr_command = ['gdal_translate'] + option_strs + [temp_fpath, dst_fpath]

    command = ' '.join(tr_command)
    info = ub.cmd(command, verbose=0)
    if info['ret'] != 0:
        print(info['out'])
        print(info['err'])
        raise Exception('failed {}'.format(info))

    if not exists(dst_fpath):
        raise Exception('Failed to create specified COG file')

    return dst_fpath


def _api_convert_cloud_optimized_geotiff(src_fpath, dst_fpath,
                                         compress='JPEG', blocksize=256):
    """
    Optimization of imwrite specifically for converting files that already
    exist on disk. Skipping the initial load of data can be very helpful.

    CommandLine:
        xdoctest -m ~/code/ndsampler/ndsampler/utils/util_gdal.py _api_convert_cloud_optimized_geotiff --bench

    """
    import gdal
    # data_set = gdal.Open(src_fpath)
    data_set = gdal.Open(src_fpath, gdal.GA_ReadOnly)

    # TODO: optional RESAMPLING option
    # NEAREST/AVERAGE/BILINEAR/CUBIC/CUBICSPLINE/LANCZOS
    overview_resample = 'NEAREST'
    # overviewlist = [2, 4, 8, 16, 32, 64]
    overviewlist = []
    if len(overviewlist) > 0:
        # TODO: check if the overviews already exist
        # main_band = data_set.GetRasterBand(1)
        # ovr_count = main_band.GetOverviewCount()
        # if ovr_count != len(overviewlist):
        # TODO: expose overview levels as arg
        data_set.BuildOverviews(str(overview_resample), overviewlist)

    options = [
        'TILED=YES',
        'BIGTIFF=YES',
        'BLOCKXSIZE={}'.format(blocksize),
        'BLOCKYSIZE={}'.format(blocksize),
    ]
    if compress != 'RAW':
        options += ['COMPRESS={}'.format(compress)]
    if compress == 'JPEG' and data_set.RasterCount == 3:
        # Using YCBCR speeds up jpeg compression by quite a bit
        options += ['PHOTOMETRIC=YCBCR']
    if overviewlist:
        options.append('COPY_SRC_OVERVIEWS=YES')
    options = list(map(str, options))  # python2.7 support

    # Copy the in-memory dataset to an on-disk GeoTiff
    driver2 = gdal.GetDriverByName(str('GTiff'))
    data_set2 = driver2.CreateCopy(dst_fpath, data_set, options=options)
    data_set = None

    # OK, so setting things to None turns out to be important. Gah!
    data_set2.FlushCache()

    # Dereference everything
    data_set2 = None
    driver2 = None
    return dst_fpath


def _api_convert_cloud_optimized_geotiff2(src_fpath, dst_fpath,
                                          compress='JPEG', blocksize=256):
    """
    CommandLine:
        xdoctest -m ~/code/ndsampler/ndsampler/utils/util_gdal.py _api_convert_cloud_optimized_geotiff --bench
    """
    import gdal
    # data_set = gdal.Open(src_fpath)
    data_set = gdal.Open(src_fpath, gdal.GA_ReadOnly)

    # TODO: optional RESAMPLING option
    # NEAREST/AVERAGE/BILINEAR/CUBIC/CUBICSPLINE/LANCZOS
    overview_resample = 'NEAREST'
    # overviewlist = [2, 4, 8, 16, 32, 64]
    overviewlist = []
    if len(overviewlist) > 0:
        # TODO: check if the overviews already exist
        # main_band = data_set.GetRasterBand(1)
        # ovr_count = main_band.GetOverviewCount()
        # if ovr_count != len(overviewlist):
        # TODO: expose overview levels as arg
        data_set.BuildOverviews(str(overview_resample), overviewlist)

    options = [
        'TILED=YES',
        'BIGTIFF=YES',
        'BLOCKXSIZE={}'.format(blocksize),
        'BLOCKYSIZE={}'.format(blocksize),
    ]
    if compress != 'RAW':
        options += ['COMPRESS={}'.format(compress)]

    if compress == 'JPEG' and data_set.RasterCount == 3:
        # Using YCBCR speeds up jpeg compression by quite a bit
        options += ['PHOTOMETRIC=YCBCR']
    if overviewlist:
        options.append('COPY_SRC_OVERVIEWS=YES')
    options = list(map(str, options))  # python2.7 support

    translate_opts = gdal.TranslateOptions(format='GTiff', creationOptions=options)
    data_set2 = gdal.Translate(dst_fpath, data_set, options=translate_opts)
    data_set2 = None

    # # Copy the in-memory dataset to an on-disk GeoTiff
    # driver2 = gdal.GetDriverByName(str('GTiff'))
    # data_set2 = driver2.CreateCopy(dst_fpath, data_set, options=options)
    # data_set = None

    # # OK, so setting things to None turns out to be important. Gah!
    # data_set2.FlushCache()

    # # Dereference everything
    # data_set2 = None
    # driver2 = None
    return dst_fpath

_GDAL_DTYPE_LUT = {
    1: np.uint8,     2: np.uint16,
    3: np.int16,     4: np.uint32,      5: np.int32,
    6: np.float32,   7: np.float64,     8: np.complex_,
    9: np.complex_,  10: np.complex64,  11: np.complex128
}


class LazyGDalFrameFile(ub.NiceRepr):
    """
    TODO:
        - [ ] Move to its own backend module
        - [ ] When used with COCO, allow the image metadata to populate the
              height, width, and channels if possible.

    Example:
        >>> # xdoctest: +REQUIRES(module:gdal)
        >>> from ndsampler.utils.util_gdal import *  # NOQA
        >>> self = LazyGDalFrameFile.demo()
        >>> cog_fpath = self.cog_fpath
        >>> print('self = {!r}'.format(self))
        >>> self[0:3, 0:3]
        >>> self[:, :, 0]
        >>> self[0]
        >>> self[0, 3]

        >>> # import kwplot
        >>> # kwplot.imshow(self[:])
    """
    def __init__(self, cog_fpath):
        self.cog_fpath = cog_fpath

    @ub.memoize_property
    def _ds(self):
        import gdal
        ds = gdal.Open(self.cog_fpath, gdal.GA_ReadOnly)
        return ds

    @classmethod
    def demo(cls):
        from ndsampler.abstract_frames import SimpleFrames
        self = SimpleFrames.demo()
        self = self._load_image_cog(1)
        return self

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        ds = self._ds
        width = ds.RasterXSize
        height = ds.RasterYSize
        C = ds.RasterCount
        return (height, width, C)

    @property
    def dtype(self):
        main_band = self._ds.GetRasterBand(1)
        dtype = _GDAL_DTYPE_LUT[main_band.DataType]
        return dtype

    def __nice__(self):
        from os.path import basename
        return '.../' + basename(self.cog_fpath)

    def __getitem__(self, index):
        """
        References:
            https://gis.stackexchange.com/questions/162095/gdal-driver-create-typeerror
        """
        ds = self._ds
        width = ds.RasterXSize
        height = ds.RasterYSize
        C = ds.RasterCount

        if not ub.iterable(index):
            index = [index]

        index = list(index)
        if len(index) < 3:
            n = (3 - len(index))
            index = index + [None] * n

        ypart = _rectify_slice_dim(index[0], height)
        xpart = _rectify_slice_dim(index[1], width)
        channel_part = _rectify_slice_dim(index[2], C)
        trailing_part = [channel_part]

        if len(trailing_part) == 1:
            channel_part = trailing_part[0]
            rb_indices = range(*channel_part.indices(C))
        else:
            rb_indices = range(C)
            assert len(trailing_part) <= 1

        # TODO: preallocate like kwimage
        channels = []
        for i in rb_indices:
            rb = ds.GetRasterBand(1 + i)
            xsize = rb.XSize
            ysize = rb.YSize

            ystart, ystop = map(int, [ypart.start, ypart.stop])
            ysize = ystop - ystart

            xstart, xstop = map(int, [xpart.start, xpart.stop])
            xsize = xstop - xstart

            gdalkw = dict(xoff=xstart, yoff=ystart, win_xsize=xsize, win_ysize=ysize)
            channel = rb.ReadAsArray(**gdalkw)
            channels.append(channel)

        img_part = np.dstack(channels)
        return img_part


def _rectify_slice_dim(part, D):
    if part is None:
        return slice(0, D)
    elif isinstance(part, slice):
        start = 0 if part.start is None else max(0, part.start)
        stop = D if part.stop is None else min(D, part.stop)
        if stop < 0:
            stop = D + stop
        assert part.step is None
        part = slice(start, stop)
        return part
    elif isinstance(part, int):
        part = slice(part, part + 1)
    else:
        raise TypeError(part)
    return part


def validate_gdal_file(file):
    """
    Test to see if the image is all black.

    May fail on all-black images

    Example:
        >>> # xdoctest: +REQUIRES(module:gdal)
        >>> from ndsampler.utils.util_gdal import LazyGDalFrameFile
        >>> import kwimage
        >>> gpath = kwimage.grab_test_image_fpath()
        >>> file = LazyGDalFrameFile(gpath)
        >>> validate_gdal_file(file)
    """
    import numpy as np
    # Find center point of the image
    cx, cy = np.array(file.shape[0:2]) // 2
    center = [cx, cy]
    # Check if the center pixels have data, look at more data if needbe
    sizes = [8, 512, 2048, 5000]
    for d in sizes:
        index = tuple(slice(c - d, c + d) for c in center)
        partial_data = file[index]
        total = partial_data.sum()

        if 0:
            import kwimage
            tmp = (partial_data / partial_data.max()).astype(np.float32)
            tmp = (partial_data / 2 ** 11)

            from skimage.exposure import equalize_hist
            tmp = equalize_hist(partial_data)

            tmp8 = kwimage.ensure_uint255(tmp)
            kwimage.imwrite('debug.png', tmp8)
            from skimage.exposure import equalize_hist
            tmp = equalize_hist(partial_data)

            tmp8 = kwimage.ensure_uint255(tmp)
            kwimage.imwrite('debug.png', tmp8)

        if total > 0:
            break

    if total == 0:
        total = file[:].sum()

    is_valid = total > 0
    return is_valid
