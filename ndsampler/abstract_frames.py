# -*- coding: utf-8 -*-
"""
Fast access to subregions of images


TODO:
    - [X] Implement npy memmap backend
    - [ ] Implement gdal COG.TIFF backend
        - [ ] Use as COG if input file is a COG
        - [ ] Convert to COG if needed
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import itertools as it
import numpy as np
import os
# import shutil
# import tempfile
import ubelt as ub
# import lockfile
# https://stackoverflow.com/questions/489861/locking-a-file-in-python
import fasteners  # supercedes lockfile / oslo_concurrency?
import atomicwrites
import six

import warnings
from PIL import Image
from os.path import exists
from os.path import join

if six.PY2:
    from backports.functools_lru_cache import lru_cache
else:
    from functools import lru_cache


class Frames(object):
    """
    Abstract implementation of Frames.

    Need to overload constructor and implement _lookup_gpath.

    Example:
        >>> from ndsampler.abstract_frames import *
        >>> self = SimpleFrames.demo()
        >>> file = self.load_image(1)
        >>> print('file = {!r}'.format(file))
        >>> self._backend = 'npy'
        >>> assert self.load_image(1).shape == (512, 512, 3)
        >>> assert self.load_region(1, (slice(-20), slice(-10))).shape == (492, 502, 3)
        >>> self._backend = 'cog'
        >>> assert self.load_image(1).shape == (512, 512, 3)
        >>> assert self.load_region(1, (slice(-20), slice(-10))).shape == (492, 502, 3)

    Benchmark:
        >>> import ubelt as ub
        >>> #
        >>> ti = ub.Timerit(100, bestof=3, verbose=2)
        >>> self = SimpleFrames.demo()
        >>> #
        >>> self._backend = 'cog'
        >>> for timer in ti.reset('cog-small-subregion'):
        >>>     self.load_image(1)[10:42, 10:42]
        >>> #
        >>> self._backend = 'npy'
        >>> for timer in ti.reset('npy-small-subregion'):
        >>>     self.load_image(1)[10:42, 10:42]
        >>> print('----')
        >>> #
        >>> self._backend = 'cog'
        >>> for timer in ti.reset('cog-large-subregion'):
        >>>     self.load_image(1)[3:-3, 3:-3]
        >>> #
        >>> self._backend = 'npy'
        >>> for timer in ti.reset('npy-large-subregion'):
        >>>     self.load_image(1)[3:-3, 3:-3]
        >>> print('----')
        >>> #
        >>> self._backend = 'cog'
        >>> for timer in ti.reset('cog-loadimage'):
        >>>     self.load_image(1)
        >>> #
        >>> self._backend = 'npy'
        >>> for timer in ti.reset('npy-loadimage'):
        >>>     self.load_image(1)
    """

    def __init__(self, id_to_hashid=None, hashid_mode='PATH', workdir=None,
                 backend='cog'):

        # self._backend = 'npy'
        self._backend = backend
        assert self._backend in ['cog', 'npy']

        if id_to_hashid is None:
            id_to_hashid = {}
        else:
            hashid_mode = 'GIVEN'

        self.id_to_hashid = id_to_hashid

        if workdir is None:
            workdir = ub.get_app_cache_dir('ndsampler')
            warnings.warn('Frames workdir not specified. '
                          'Defaulting to {!r}'.format(workdir))

        self.workdir = workdir
        self._cache_dpath = None

        self.hashid_mode = hashid_mode

        if self.hashid_mode not in ['PATH', 'PIXELS', 'GIVEN']:
            raise KeyError(self.hashid_mode)

    @property
    def cache_dpath(self):
        """
        Returns the path where cached frame representations will be stored
        """
        if self._cache_dpath is None:
            if self._backend == 'npy':
                self._cache_dpath = ub.ensuredir(join(self.workdir, '_cache/frames'))
            else:
                self._cache_dpath = ub.ensuredir(join(self.workdir, '_cache/frames/' + self._backend))
        return self._cache_dpath

    def _lookup_gpath(self, image_id):
        raise NotImplementedError

    def _lookup_hashid(self, image_id):
        """
        Get the hashid of a particular image
        """
        if image_id not in self.id_to_hashid:
            # Compute the hash if we it does not exist yet
            gpath = self._lookup_gpath(image_id)
            if self.hashid_mode == 'PATH':
                # Hash the full path to the image data
                hashid = ub.hash_data(gpath, hasher='sha1', base='hex')
            elif self.hashid_mode == 'PIXELS':
                # Hash the pixels in selfthe image
                hashid = ub.hash_file(gpath, hasher='sha1', base='hex')
            elif self.hashid_mode == 'GIVEN':
                raise IndexError(
                    'Requested image_id={} was not given'.format(image_id))
            else:
                raise KeyError(self.hashid_mode)

            # Assume the image info will not change and cache the hashid.
            # If the info does change this cache must be notified and cleared.
            self.id_to_hashid[image_id] = hashid

        return self.id_to_hashid[image_id]

    def load_region(self, image_id, region=None, bands=None, scale=0):
        """
        Ammortized O(1) image subregion loading

        Args:
            image_id (int): image identifier
            region (Tuple[slice, ...]): space-time region within an image
            bands (str): NotImplemented
            scale (float): NotImplemented
        """
        region = self._rectify_region(region)
        if all(r.start is None and r.stop is None for r in region):
            im = self.load_image(image_id, bands=bands, scale=scale,
                                 cache=False)
        else:
            file = self.load_image(image_id, bands=bands, scale=scale,
                                   cache=True)
            im = file[region]
        return im

    def _rectify_region(self, region):
        if region is None:
            region = tuple([])
        if len(region) < 2:
            # Add empty dimensions
            tail = tuple([slice(None)] * (2 - len(region)))
            region = tuple(region) + tail
        return region

    def load_image(self, image_id, bands=None, scale=0, cache=True):
        if bands is not None:
            raise NotImplementedError('cannot handle different bands')

        if scale != 0:
            raise NotImplementedError('can only handle scale=0')

        if not cache:
            gpath = self._lookup_gpath(image_id)
            raw_data = np.asarray(Image.open(gpath))
            return raw_data

        if self._backend == 'cog':
            return self._load_image_cog(image_id, cache=cache)
        elif self._backend == 'npy':
            return self._load_image_npy(image_id, cache=cache)
        else:
            raise KeyError(self._backend)

    @lru_cache(4)  # Keeps frequently accessed images open
    def _load_image_npy(self, image_id, cache=True):
        """
        Returns a memmapped reference to the entire image
        """
        gpath = self._lookup_gpath(image_id)
        hashid = self._lookup_hashid(image_id)

        mem_gname = '{}_{}.npy'.format(image_id, hashid)
        mem_gpath = join(self.cache_dpath, mem_gname)
        if not exists(mem_gpath):
            self.ensure_npy_representation(gpath, mem_gpath)

        # I can't figure out why a race condition exists here.  I can't
        # reproduce it with a small example. In the meantime, this hack should
        # mitigate the problem by simply trying again until it works.
        HACKY_RACE_CONDITION_FIX = True
        if HACKY_RACE_CONDITION_FIX:
            for try_num in it.count():
                try:
                    file = np.load(mem_gpath, mmap_mode='r')
                except Exception:
                    print('\n\n')
                    print('ERROR: FAILED TO LOAD CACHED FILE')
                    print('mem_gpath = {!r}'.format(mem_gpath))
                    print('exists(mem_gpath) = {!r}'.format(exists(mem_gpath)))
                    print('Recompute cache: Try number {}'.format(try_num))
                    print('\n\n')
                    if try_num > 9000:
                        # Something really bad must be happening, stop trying
                        raise
                    try:
                        os.remove(mem_gpath)
                    except Exception:
                        print('WARNING: CANNOT REMOVE CACHED FRAME.')
                    self.ensure_npy_representation(gpath, mem_gpath)
                else:
                    break
        else:
            # FIXME;
            # There seems to be a race condition that triggers the error:
            # ValueError: mmap length is greater than file size
            try:
                file = np.load(mem_gpath, mmap_mode='r')
            except ValueError:
                print('\n\n')
                print('ERROR')
                print('mem_gpath = {!r}'.format(mem_gpath))
                print('exists(mem_gpath) = {!r}'.format(exists(mem_gpath)))
                print('\n\n')
                raise
        return file

    @lru_cache(4)  # Keeps frequently accessed images open
    def _load_image_cog(self, image_id, cache=True):
        """
        Returns a special array-like object with a COG GeoTIFF backend
        """
        gpath = self._lookup_gpath(image_id)
        hashid = self._lookup_hashid(image_id)

        cog_gname = '{}_{}.cog.tiff'.format(image_id, hashid)
        cog_gpath = join(self.cache_dpath, cog_gname)
        if not exists(cog_gpath):
            self._ensure_cog_representation(gpath, cog_gpath)
        file = LazyGDalFrameFile(cog_gpath)
        return file

    @staticmethod
    def _ensure_cog_representation(gpath, cog_gpath):
        _locked_cache_write(_cog_cache_write, gpath, cache_gpath=cog_gpath)

    @staticmethod
    def ensure_npy_representation(gpath, mem_gpath):
        """
        Ensures that mem_gpath exists in a multiprocessing-safe way
        """
        _locked_cache_write(_npy_cache_write, gpath, cache_gpath=mem_gpath)


def _cog_cache_write(gpath, cache_gpath):
    """
    Example:
        >>> import ndsampler
        >>> from ndsampler.abstract_frames import *
        >>> workdir = ub.ensure_app_cache_dir('ndsampler')
        >>> dset = ndsampler.CocoDataset.demo()
        >>> imgs = dset.images()
        >>> id_to_name = imgs.lookup('file_name', keepid=True)
        >>> id_to_path = {gid: join(dset.img_root, name)
        >>>               for gid, name in id_to_name.items()}
        >>> self = SimpleFrames(id_to_path, workdir=workdir)
        >>> image_id = ub.peek(id_to_name)
        >>> gpath = self._lookup_gpath(image_id)
        >>> hashid = self._lookup_hashid(image_id)
        >>> cog_gname = '{}_{}.cog.tiff'.format(image_id, hashid)
        >>> cache_gpath = cog_gpath = join(self.cache_dpath, cog_gname)
    """
    # Load all the image data and dump it to npy format
    import kwimage
    raw_data = kwimage.imread(gpath)
    _imwrite_cloud_optimized_geotiff(cache_gpath, raw_data)


def _npy_cache_write(gpath, cache_gpath):
    # Load all the image data and dump it to npy format
    import kwimage
    raw_data = kwimage.imread(gpath)
    # raw_data = np.asarray(Image.open(gpath))

    # Even with the file-lock and atomic save there is still a race
    # condition somewhere. Not sure what it is.
    # _semi_atomic_numpy_save(cache_gpath, raw_data)

    with atomicwrites.atomic_write(cache_gpath, mode='wb', overwrite=True) as file:
        np.save(file, raw_data)


def _locked_cache_write(_write_func, gpath, cache_gpath):
    """
    Ensures that mem_gpath exists in a multiprocessing-safe way
    """
    import multiprocessing

    lock_fpath = cache_gpath + '.lock'

    DEBUG = 0

    if DEBUG:
        procid = multiprocessing.current_process()
        def _debug(msg):
            with open(lock_fpath + '.debug', 'a') as f:
                f.write(msg + '\n')
        _debug(lock_fpath)
        _debug('{} attempts aquire'.format(procid))

    # Ensure that another process doesn't write to the same image
    # FIXME: lockfiles may not be cleaned up gracefully
    # See: https://github.com/harlowja/fasteners/issues/26
    lock_fpath = cache_gpath + '.lock'
    with fasteners.InterProcessLock(lock_fpath):

        if DEBUG:
            _debug('{} aquires'.format(procid))

        if not exists(cache_gpath):
            _write_func(gpath, cache_gpath)

            if DEBUG:
                _debug('{} wrote'.format(procid))
        else:
            if DEBUG:
                _debug('{} does not need to write'.format(procid))

        if DEBUG:
            _debug('{} releasing'.format(procid))

    if DEBUG:
        _debug('{} released'.format(procid))


# def _semi_atomic_numpy_save(fpath, data):
#     """
#     Save to a temporary file. Then move to `fpath` using an atomic operation.

#     If the file that is being saved to is on the same file system the move
#     operation is atomic, otherwise it may not be [1].

#     References:
#         ..[1] https://stackoverflow.com/questions/3716325/is-shutil-move-atomic
#     """

#     # TODO: use atomicwrites instead
#     tmp = tempfile.NamedTemporaryFile(delete=False)
#     try:
#         # Use temporary files to avoid partially written data
#         np.save(tmp, data)
#         tmp.close()
#         # Use an atomic operation (if possible) to move the temporary saved
#         # file to the target location.
#         shutil.move(tmp.name, fpath)
#     finally:
#         tmp.close()
#         if exists(tmp.name):
#             os.remove(tmp.name)


class SimpleFrames(Frames):
    """
    Basic concrete implementation of frames objects

    Args:
        id_to_path (Dict): mapping from image-id to image path

    Example:
        >>> from ndsampler.abstract_frames import *
        >>> self = SimpleFrames.demo()
        >>> self._backend == 'npy'
        >>> assert self.load_image(1).shape == (512, 512, 3)
        >>> assert self.load_region(1, (slice(-20), slice(-10))).shape == (492, 502, 3)
    """
    def __init__(self, id_to_path, id_to_hashid=None, hashid_mode='PATH',
                 workdir=None):
        super(SimpleFrames, self).__init__(id_to_hashid=id_to_hashid,
                                           hashid_mode=hashid_mode,
                                           workdir=workdir)
        self.id_to_path = id_to_path

    def _lookup_gpath(self, image_id):
        return self.id_to_path[image_id]

    @classmethod
    def demo(self):
        import ndsampler
        workdir = ub.ensure_app_cache_dir('ndsampler')
        dset = ndsampler.CocoDataset.demo()
        imgs = dset.images()
        id_to_name = imgs.lookup('file_name', keepid=True)
        id_to_path = {gid: join(dset.img_root, name)
                      for gid, name in id_to_name.items()}
        self = SimpleFrames(id_to_path, workdir=workdir)
        return self


def __notes__():
    """
    Testing alternate implementations
    """

    def _subregion_with_pil(img, region):
        """
        References:
            https://stackoverflow.com/questions/23253205/read-a-subset-of-pixels-with-pil-pillow/50823293#50823293
        """
        # It is much faster to crop out a subregion from a npy memmap file
        # than it is to work with a raw image (apparently). Thus the first
        # time we see an image, we load the entire thing, and dump it to a
        # cache directory in npy format. Then any subsequent time the image
        # is needed, we efficiently load it from numpy.
        import kwimage
        gpath = kwimage.grab_test_image_fpath()
        # Directly load a crop region with PIL (this is slow, why?)
        pil_img = Image.open(gpath)

        width = img['width']
        height = img['height']

        yregion, xregion = region
        left, right, _ = xregion.indices(width)
        upper, lower, _ = yregion.indices(height)
        area = left, upper, right, lower

        # Ok, so loading images with PIL is still slow even with crop.  Not
        # sure why. I though I measured that it was significantly better
        # before. Oh well. Let hack this and the first time we read an image we
        # will dump it to a numpy file and then memmap into it to crop out the
        # appropriate slice.
        sub_img = pil_img.crop(area)
        im = np.asarray(sub_img)
        return im

    def _alternate_memap_creation(mem_gpath, img, region):
        # not sure where the 128 magic number comes from
        # https://stackoverflow.com/questions/51314748/is-the-bytes-offset-in-files-produced-by-np-save-always-128
        width = img['width']
        height = img['height']
        img_shape = (height, width, 3)
        img_dtype = np.dtype('uint8')
        file = np.memmap(mem_gpath, dtype=img_dtype, shape=img_shape,
                         offset=128, mode='r')
        return file


def _imwrite_cloud_optimized_geotiff(fpath, data):
    """
    References:
        https://geoexamples.com/other/2019/02/08/cog-tutorial.html#create-a-cog-using-gdal-python

    Notes:
        conda install gdal

        OR

        sudo apt install gdal-dev
        pip install gdal ---with special flags, forgot which though, sry

    Example:
        >>> data = (np.random.rand(1000, 1000, 3) * 255).astype(np.uint8)
        >>> fpath = '/tmp/foo.cog.tiff'
        >>> _imwrite_cloud_optimized_geotiff(fpath, data)
        >>> import netharn as nh
        >>> print(nh.util.get_file_info(fpath))

        fpath2 = '/tmp/foo.png'
        kwimage.imwrite(fpath2, data)
        print(nh.util.get_file_info(fpath2))

        fpath3 = '/tmp/foo.jpg'
        kwimage.imwrite(fpath3, data)
        print(nh.util.get_file_info(fpath3))
    """
    import gdal
    y_size, x_size, num_bands = data.shape

    kindsize = (data.dtype.kind, data.dtype.itemsize)
    if kindsize == ('u', 1):
        eType = gdal.GDT_Byte
    else:
        raise TypeError(kindsize)

    driver = gdal.GetDriverByName('MEM')
    data_set = driver.Create('', x_size, y_size, num_bands,
                             eType=eType)

    for i in range(num_bands):
        data_set.GetRasterBand(i + 1).WriteArray(data[:, :, i])

    data = None
    data_set.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64])

    driver = gdal.GetDriverByName('GTiff')
    data_set2 = driver.CreateCopy(fpath, data_set,
                                  options=["COPY_SRC_OVERVIEWS=YES",
                                           "TILED=YES",
                                           "COMPRESS=LZW"])
    data_set2.FlushCache()
    return fpath


class LazyGDalFrameFile(ub.NiceRepr):
    """
    Example:
        >>> from ndsampler.abstract_frames import *
        >>> self = SimpleFrames.demo()
        >>> file = self._load_image_cog(1)
        >>> cog_fpath = file.cog_fpath
        >>> print('file = {!r}'.format(file))
        >>> file[0:3, 0:3]
        >>> file[:, :, 0]
        >>> file[0]
        >>> file[0, 3]
    """
    def __init__(self, cog_fpath):
        self.cog_fpath = cog_fpath

    @property
    def shape(self):
        import gdal
        ds = gdal.Open(self.cog_fpath, gdal.GA_ReadOnly)
        width = ds.RasterXSize
        height = ds.RasterYSize
        C = ds.RasterCount
        return (height, width, C)

    def __nice__(self):
        from os.path import basename
        return '.../' + basename(self.cog_fpath)

    def __getitem__(self, index):
        """
        References:
            https://gis.stackexchange.com/questions/162095/gdal-driver-create-typeerror
        """
        import gdal
        ds = gdal.Open(self.cog_fpath, gdal.GA_ReadOnly)

        width = ds.RasterXSize
        height = ds.RasterYSize
        C = ds.RasterCount

        def rectify_slice(part, D):
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

        if not ub.iterable(index):
            index = [index]

        index = list(index)
        if len(index) < 3:
            n = (3 - len(index))
            index = index + [None] * n

        ypart = rectify_slice(index[0], height)
        xpart = rectify_slice(index[1], width)
        channel_part = rectify_slice(index[2], C)
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
