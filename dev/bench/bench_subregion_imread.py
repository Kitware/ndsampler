import ubelt as ub
import numpy as np
import tempfile
import cv2
import timerit
from os.path import join
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def gdal_subregion_imread(img_fpath, index):
    """
    References:
        https://gis.stackexchange.com/questions/162095/gdal-driver-create-typeerror
    """
    import gdal
    ypart, xpart = index[0:2]
    trailing_part = index[2:]

    ds = gdal.Open(img_fpath, gdal.GA_ReadOnly)

    if len(trailing_part) == 1:
        channel_part = trailing_part[0]
        rb_indices = range(*channel_part.indices(ds.RasterCount))
    else:
        rb_indices = range(ds.RasterCount)
        assert len(trailing_part) <= 1

    channels = []
    for i in rb_indices:
        ds.RasterCount

        rb = ds.GetRasterBand(1)
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


def imwrite_cloud_optimized_geotiff(fpath, data):
    """
    References:
        https://geoexamples.com/other/2019/02/08/cog-tutorial.html#create-a-cog-using-gdal-python

    Example:
        data = (np.random.rand(1000, 1000, 3) * 255).astype(np.uint8)
        fpath = '/tmp/foo.cog.tiff'
        import netharn as nh
        print(nh.util.get_file_info(fpath))

        fpath2 = '/tmp/foo.png'
        kwimage.imwrite(fpath2, data)
        print(nh.util.get_file_info(fpath2))

        fpath3 = '/tmp/foo.jpg'
        kwimage.imwrite(fpath3, data)
        print(nh.util.get_file_info(fpath3))
    """
    import gdal
    y_size, x_size, num_bands = data.shape
    # driver = gdal.GetDriverByName('GTiff')
    # data_set = driver.Create(fpath, x_size, y_size, num_bands,
    #                          gdal.GDT_Float32,
    #                          options=["TILED=YES", "COMPRESS=LZW",
    #                                   "INTERLEAVE=BAND"])

    # for i in range(num_bands):
    #     data_set.GetRasterBand(i + 1).WriteArray(data[..., i])

    # data_set.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64])

    kindsize = (data.dtype.kind, data.dtype.itemsize)
    if kindsize == ('u', 1):
        eType = gdal.GDT_Byte
    else:
        raise TypeError(kindsize)

    driver = gdal.GetDriverByName('MEM')
    data_set = driver.Create('', x_size, y_size, num_bands,
                             eType=eType)

    for i in range(num_bands):
        data_set.GetRasterBand(i + 1).WriteArray(data[i])

    data = None
    data_set.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64])

    driver = gdal.GetDriverByName('GTiff')
    data_set2 = driver.CreateCopy(fpath, data_set,
                                  options=["COPY_SRC_OVERVIEWS=YES",
                                           "TILED=YES",
                                           "COMPRESS=LZW"])
    data_set2.FlushCache()
    return fpath


def time_ondisk_crop(size=512, dim=3, region='small_random', num=24):
    """
    Ignore:
        >>> from bench_subregion_imread import *  # NOQA
        >>> import xdev
        >>> globals().update(xdev.get_func_kwargs(time_ondisk_crop))
    """
    enabled = {
        'in_memory': False,
        'memmap': True,

        'PIL': False,
        'OpenCV': True,

        'VIPS': True,

        'GDAL': True,
        'HDF5': True,
    }

    print('\n---')

    # DATA = 'custom'
    DATA = 'random'

    if DATA == 'random':
        # data = (np.random.rand(500, 500, 3) * 255).astype(np.uint8)
        print('Generate random data, size={}, dim={}, mode={}'.format(
            size, dim, region))
        x, y = np.meshgrid(np.arange(0, size), np.arange(0, size))
        data = np.ascontiguousarray((np.dstack([x] * dim) % 256).astype(np.uint8))
        data = data.squeeze()
    elif DATA == 'custom':
        print('USE CUSTOM data, size={}, dim={}, mode={}'.format(
            size, dim, region))
        assert False
        custom_fpath = '.ptif'
        import kwimage
        data = kwimage.imread(custom_fpath)
    else:
        raise KeyError(DATA)

    print('Make temp directory to prepare data')
    dpath = tempfile.mkdtemp()

    lossy_ext = ('.jpg', '.ptif', '.cog')

    img_fpaths = {
        # 'png': join(dpath, 'foo.png'),
        # 'jpg': join(dpath, 'foo.jpg'),
        # 'tif': join(dpath, 'foo.tif'),
    }
    pil_img = Image.fromarray(data)
    for k, v in img_fpaths.items():
        print('DUMP v = {!r}'.format(v))
        pil_img.save(v)

    img_fpaths['cog'] = join(dpath, 'foo.cog')
    imwrite_cloud_optimized_geotiff(img_fpaths['cog'], data)

    if DATA == 'custom':
        from os.path import splitext
        import shutil
        ext = splitext(custom_fpath)[1][1:]
        tmp_fpath = join(dpath, 'foo' + ext)
        shutil.copy2(custom_fpath, tmp_fpath)
        img_fpaths.update({
            'custom_' + ext: tmp_fpath,
        })

    mem_fpaths = {}
    if enabled['memmap']:
        mem_fpaths = {
            'npy': join(dpath, 'foo.npy'),
        }
        for key, fpath in mem_fpaths.items():
            print('DUMP fpath = {!r}'.format(fpath))
            np.save(fpath, data)

    h5_fpaths = {}
    if enabled['HDF5']:
        import h5py
        h5_params = {
            # 'basic': {},
            # 'chunks': {'chunks': (32, 32, 1)},
            # 'lzf': {'compression': 'lzf'},
            # 'lzf_chunk32': {'compression': 'lzf', 'chunks': (32, 32, 1)},
            'lzf_chunk128': {'compression': 'lzf', 'chunks': (128, 128, 1)},
        }
        for key, kw in h5_params.items():
            print('Dump h5 ' + key)
            fpath = h5_fpaths[key] = join(dpath, key + '.h5')
            with h5py.File(fpath, 'w') as h5_file:
                dset = h5_file.create_dataset('DATA', data.shape, data.dtype, **kw)
                dset[...] = data

    import netharn as nh
    bytes_on_disk = {}
    for k, v in mem_fpaths.items():
        bytes_on_disk['mem_' + k] = nh.util.get_file_info(v)['filesize']
    for k, v in img_fpaths.items():
        bytes_on_disk['img_' + k] = nh.util.get_file_info(v)['filesize']
    for k, v in h5_fpaths.items():
        bytes_on_disk['hdf5_' + k] = nh.util.get_file_info(v)['filesize']

    mb_on_disk = ub.map_vals(lambda x: str(round(x * 1e-6, 2)) + ' MB', bytes_on_disk)
    print('on-disk memory usage: ' + ub.repr2(mb_on_disk, nl=1))

    result = {}

    def record_result(timer):
        ti = timer.parent
        val = ti.min(), ti.mean(), ti.std()
        result[ti.label] = val

    rng = np.random.RandomState()

    def get_index():
        """ Get a subregion to load """
        if region == 'small_random':
            # Small window size, but random location
            size = (172, 172)
            h, w = size
            a = rng.randint(0, data.shape[0] - h)
            b = rng.randint(0, data.shape[1] - w)
            index = tuple([slice(a, a + h), slice(b, b + w)])
        elif region == 'random':
            a, b = sorted(rng.randint(0, data.shape[0], size=2))
            c, d = sorted(rng.randint(0, data.shape[1], size=2))
            index = tuple([slice(a, b + 1), slice(c, d + 1)])
        elif region == 'corner':
            index = tuple([slice(0, 8), slice(0, 8)])
        else:
            raise KeyError(index)
            # index = region
        if len(data.shape) > 2:
            index = index + tuple([slice(0, 3)])
        area = (index[1].start, index[0].start, index[1].stop, index[0].stop)
        shape = tuple([s.stop - s.start for s in index])
        return index, area, shape

    def TIMERIT(label):
        # Ensure each timer run uses the same random numbers
        rng.seed(0)
        return timerit.Timerit(
            num=num,
            bestof=1,
            label=label,
            # unit='us',
            unit='ms',
        )

    print('Begin benchmarks\n')

    if enabled['in_memory']:
        for timer in TIMERIT('in-memory slice'):
            index, area, shape = get_index()
            want = data[index]
            with timer:
                got = data[index]
        record_result(timer)

    if enabled['memmap']:
        for key, fpath in mem_fpaths.items():
            for timer in TIMERIT('np.memmap load+slice ' + key):
                index, area, shape = get_index()
                want = data[index]
                with timer:
                    file1 = np.memmap(fpath, dtype=data.dtype.name,
                                      shape=data.shape, offset=128, mode='r')
                    got = file1[index]
                assert np.all(got == want)
            record_result(timer)

    if enabled['memmap']:
        for key, fpath in mem_fpaths.items():
            for timer in TIMERIT('np.load load+slice ' + key):
                index, area, shape = get_index()
                want = data[index]
                with timer:
                    file2 = np.load(fpath, mmap_mode='r')
                    got = file2[index]
                assert np.all(got == want)
            record_result(timer)

    if enabled['PIL']:
        for key, fpath in img_fpaths.items():
            for timer in TIMERIT('PIL open+crop (minimal) ' + key):
                index, area, shape = get_index()
                want = data[index]
                with timer:
                    core = Image.open(fpath).crop(area)
            record_result(timer)

    if enabled['PIL']:
        for key, fpath in img_fpaths.items():
            for timer in TIMERIT('PIL open+crop+getdata+asarray ' + key):
                index, area, shape = get_index()
                want = data[index]
                with timer:
                    core = Image.open(fpath).crop(area)
                    got = np.asarray(core.getdata(), dtype=np.uint8)
                    got.shape = shape
            assert fpath.endswith(lossy_ext) or np.all(got == want)
            record_result(timer)

    if enabled['PIL']:
        for key, fpath in img_fpaths.items():
            for timer in TIMERIT('PIL open+asarray+slice ' + key):
                index, area, shape = get_index()
                want = data[index]
                with timer:
                    got = np.asarray(Image.open(fpath))[index]
            assert fpath.endswith(lossy_ext) or np.all(got == want)
            record_result(timer)

    if enabled['OpenCV']:
        for key, fpath in img_fpaths.items():
            for timer in TIMERIT('OpenCV imread+slice ' + key):
                index, area, shape = get_index()
                want = data[index]
                with timer:
                    got = cv2.imread(fpath, flags=cv2.IMREAD_UNCHANGED)[index]
                if len(index) > 2:
                    got = got[:, :, ::-1]
            assert fpath.endswith(lossy_ext) or np.all(got == want)
            record_result(timer)

    if enabled['GDAL']:
        for key, fpath in img_fpaths.items():
            for timer in TIMERIT('GDAL subregion ' + key):
                index, area, shape = get_index()
                want = data[index]
                with timer:
                    got = gdal_subregion_imread(fpath, index)
            assert fpath.endswith(lossy_ext) or np.all(got == want)
            record_result(timer)

    # pip install pyvips
    if enabled['VIPS']:
        import pyvips
        for key, fpath in img_fpaths.items():
            for timer in TIMERIT('VIPS ' + key):
                index, area, shape = get_index()
                want = data[index]
                left, top = area[0:2]
                width, height = shape[0:2][::-1]
                vips_img = pyvips.Image.new_from_file(
                    fpath,
                    # access='sequential',
                    access='random',
                    # memory=False,
                    # fail=True
                )
                with timer:
                    vips_sub = vips_img.crop(left, top, width, height)
                    got = np.ndarray(buffer=vips_sub.write_to_memory(),
                                     dtype=np.uint8,
                                     shape=[vips_sub.height, vips_sub.width,
                                            vips_sub.bands])
            assert fpath.endswith(lossy_ext) or np.all(got == want)
            record_result(timer)

    if enabled['HDF5']:
        for key, fpath in h5_fpaths.items():
            for timer in TIMERIT('HDF5 ' + key):
                with h5py.File(fpath, 'r') as file:
                    dset = file['DATA']
                    index, area, shape = get_index()
                    want = data[index]
                    with timer:
                        got = dset[index]
                assert fpath.endswith(lossy_ext) or np.all(got == want)
            record_result(timer)

    return result


def benchmark_ondisk_crop():
    import kwplot
    plt = kwplot.autoplt()

    region = 'small_random'

    dim = 3
    # xdata = [64, 128, 256, 512]
    # xdata = [64, 128, 256, 320, 512, 640, 768, 896, 1024]
    # xdata = np.linspace(64, 4096, num=8).astype(np.int)
    # xdata = np.linspace(64, 2048, num=8).astype(np.int)
    # xdata = np.linspace(64, 1024, num=8).astype(np.int)

    xdata = [256, 1024, 4096, 8192, 16384]
    # xdata = [256, 1024, 4096, 8192]
    xdata = [256, 1024, 2048]
    # xdata = [256]

    ydata = ub.ddict(list)
    # for size in [64, 128, 256, 512, 1024, 2048, 4096]:
    for size in xdata:
        result = time_ondisk_crop(size, dim=dim, region=region, num=5)
        for key, val in result.items():
            min, mean, std = val
            ydata[key].append(mean * 1e6)

    # Sort legend by descending time taken on the largest image
    ydata = ub.odict(sorted(ydata.items(), key=lambda i: -i[1][-1]))

    kwplot.multi_plot(
        xdata, ydata, ylabel='micro-seconds (us)', xlabel='image size',
        title='Chip region={} benchmark for {}D image data'.format(region, dim),
        # yscale='log',
        ymin=1,
    )
    plt.show()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/ndsampler/dev/bench/bench_subregion_imread.py
    """
    benchmark_ondisk_crop()
