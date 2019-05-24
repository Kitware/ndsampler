import numpy as np
import ubelt as ub


def _imwrite_cloud_optimized_geotiff(fpath, data, lossy=True):
    """
    Args:
        fpath (PathLike):
        data (ndarray):
        lossy (bool, default=True): Use lossy compression if True

    References:
        https://geoexamples.com/other/2019/02/08/cog-tutorial.html#create-a-cog-using-gdal-python
        http://osgeo-org.1560.x6.nabble.com/gdal-dev-Creating-Cloud-Optimized-GeoTIFFs-td5320101.html

    Notes:
        conda install gdal

        OR

        sudo apt install gdal-dev
        pip install gdal ---with special flags, forgot which though, sry

    Example
        >>> # DISABLE_DOCTEST
        >>> from ndsampler.utils.util_gdal import *  # NOQA
        >>> from ndsampler.utils.util_gdal import _imwrite_cloud_optimized_geotiff
        >>> data = (np.random.rand(1000, 1000, 3) * 255).astype(np.uint8)
        >>> fpath = '/tmp/foo.cog.tiff'

        >>> _imwrite_cloud_optimized_geotiff(fpath, data, lossy=False)
        >>> import netharn as nh
        >>> print(ub.repr2(nh.util.get_file_info(fpath)))
        >>> from ndsampler.utils.validate_cog import validate as _validate_cog
        >>> errors = _validate_cog(fpath)
        >>> warnings, errors, details = _validate_cog(fpath)
        >>> print('details = ' + ub.repr2(details))
        >>> print('warnings = ' + ub.repr2(warnings))
        >>> print('errors = ' + ub.repr2(errors))
        >>> assert not errors

        >>> _imwrite_cloud_optimized_geotiff(fpath, data, lossy=True)
        >>> import netharn as nh
        >>> print(ub.repr2(nh.util.get_file_info(fpath)))
        >>> from ndsampler.utils.validate_cog import validate as _validate_cog
        >>> errors = _validate_cog(fpath)
        >>> warnings, errors, details = _validate_cog(fpath)
        >>> print('details = ' + ub.repr2(details))
        >>> print('warnings = ' + ub.repr2(warnings))
        >>> print('errors = ' + ub.repr2(errors))
        >>> assert not errors

        fpath2 = '/tmp/foo.png'
        kwimage.imwrite(fpath2, data)
        print(nh.util.get_file_info(fpath2))

        fpath3 = '/tmp/foo.jpg'
        kwimage.imwrite(fpath3, data)
        print(nh.util.get_file_info(fpath3))
    """
    import gdal
    y_size, x_size, num_bands = data.shape

    if num_bands == 4:
        # Silently discarding the alpha channel
        num_bands = 3

    kindsize = (data.dtype.kind, data.dtype.itemsize)
    if kindsize == ('u', 1):
        eType = gdal.GDT_Byte
    else:
        raise TypeError(kindsize)

    driver = gdal.GetDriverByName(str('MEM'))
    data_set = driver.Create(str(''), x_size, y_size, num_bands,
                             eType=eType)

    for i in range(num_bands):
        band_data = np.ascontiguousarray(data[:, :, i])
        data_set.GetRasterBand(i + 1).WriteArray(band_data)

    data_set.BuildOverviews(str('NEAREST'), [2, 4, 8, 16, 32, 64])

    if lossy:
        options = ['COPY_SRC_OVERVIEWS=YES', 'TILED=YES', 'COMPRESS=JPEG', 'PHOTOMETRIC=YCBCR']
        # options = ['COPY_SRC_OVERVIEWS=YES', 'TILED=YES', 'COMPRESS=JPEG']
    else:
        options = ['COPY_SRC_OVERVIEWS=YES', 'TILED=YES', 'COMPRESS=LZW']

    options = list(map(str, options))  # python2.7 support

    driver2 = gdal.GetDriverByName(str('GTiff'))
    data_set2 = driver2.CreateCopy(fpath, data_set, options=options)
    # OK, so setting things to None turns out to be important. Gah!
    data_set2.FlushCache()

    data_set = None
    data_set2 = None
    driver = driver2 = None

    # if False:
    #     z = gdal.Open(fpath, gdal.GA_ReadOnly)
    #     z.GetRasterBand(1).GetMetadataItem('BLOCK_OFFSET_0_0', 'TIFF')
    return fpath


class LazyGDalFrameFile(ub.NiceRepr):
    """
    TODO:
        - [ ] Move to its own backend module
        - [ ] When used with COCO, allow the image metadata to populate the
              height, width, and channels if possible.

    Example:
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
