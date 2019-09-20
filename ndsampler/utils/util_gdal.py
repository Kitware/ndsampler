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
    # we dont support floats to keep things somewhat simple
    # elif kindsize == ('f', 4):
    #     eType = gdal.GDT_Float32
    # elif kindsize == ('f', 8):
    #     eType = gdal.GDT_Float64
    else:
        raise TypeError('Unsupported GDAL dtype for {}'.format(kindsize))
    return eType


@profile
def _imwrite_cloud_optimized_geotiff(fpath, data, compress='JPEG'):
    """
    Args:
        fpath (PathLike): file path to save the COG to.

        data (ndarray[ndim=3]): Raw HWC image data to save. Dimensions should
            be height, width, channels.

        compress (bool, default='JPEG'): Can be JPEG (lossy) or LZW (lossless),
            or DEFLATE (lossless).

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

    Example
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
    y_size, x_size, num_bands = data.shape

    assert compress in ['JPEG', 'LZW', 'DEFLATE'], 'unknown compress'

    if compress == 'JPEG' and num_bands >= 5:
        raise ValueError('Cannot use JPEG with more than 4 channels')

    eType = _numpy_to_gdal_dtype(data.dtype)
    if compress == 'JPEG':
        if eType not in [gdal.GDT_Byte, gdal.GDT_UInt16]:
            raise ValueError('JPEG compression must use 8 or 16 bit integer imagery')

    options = [
        'TILED=YES',
        'COMPRESS={}'.format(compress),
        'BIGTIFF=YES',
        # 'PHOTOMETRIC=YCBCR',
        # TODO: optional BLOCKSIZE
    ]
    options = list(map(str, options))  # python2.7 support

    # TODO: optional RESAMPLING option
    # NEAREST/AVERAGE/BILINEAR/CUBIC/CUBICSPLINE/LANCZOS
    overview_resample = 'NEAREST'
    overview_levels = [2, 4, 8, 16, 32, 64]
    overview_levels = []

    if overview_levels:
        options.append('COPY_SRC_OVERVIEWS=YES')

    # Create an in-memory dataset where we will prepare the COG data structure
    driver = gdal.GetDriverByName(str('MEM'))
    data_set = driver.Create(str(''), x_size, y_size, num_bands, eType=eType)
    for i in range(num_bands):
        band_data = np.ascontiguousarray(data[:, :, i])
        data_set.GetRasterBand(i + 1).WriteArray(band_data)

    if overview_levels:
        # Build the downsampled overviews (for fast zoom in / out)
        data_set.BuildOverviews(str(overview_resample), overview_levels)

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


def _cli_convert_cloud_optimized_geotiff(src_fpath, dst_fpath, compress='JPEG'):
    """
    For whatever reason using the CLI seems to simply be faster.

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
    options = [
        'TILED=YES',
        'COMPRESS={}'.format(compress),
        'BIGTIFF=YES',
        # TODO: optional BLOCKSIZE
    ]
    overview_resample = 'NEAREST'

    # TODO: overview levels
    overview_levels = []
    temp_fpath = src_fpath
    if overview_levels:
        level_str = ' '.join(list(map(str, overview_levels)))
        info = ub.cmd('gdaladdo -r {} {} {}'.format(overview_resample, temp_fpath, level_str))
        assert info['ret'] == 0, 'failed {}'.format(info)

    option_strs = ['-co {}'.format(o) for o in options]
    tr_command = ['gdal_translate', temp_fpath, dst_fpath] + option_strs
    info = ub.cmd(' '.join(tr_command))
    assert info['ret'] == 0, 'failed {}'.format(info)
    return dst_fpath


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
        return np.dtype('uint8')

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
        if total > 0:
            break

    if total == 0:
        total = file[:].sum()

    is_valid = total > 0
    return is_valid
