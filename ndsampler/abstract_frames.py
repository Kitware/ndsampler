# -*- coding: utf-8 -*-
"""
Fast access to subregions of images


TODO:
    - [X] Implement npy memmap backend
    - [X] Implement gdal COG.TIFF backend
        - [X] Use as COG if input file is a COG
        - [X] Convert to COG if needed
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
            import kwimage
            gpath = self._lookup_gpath(image_id)
            raw_data = kwimage.imread(gpath)
            # raw_data = np.asarray(Image.open(gpath))
            return raw_data
        else:
            if self._backend == 'cog':
                return self._load_image_cog(image_id)
            elif self._backend == 'npy':
                return self._load_image_npy(image_id)
            else:
                raise KeyError(self._backend)

    @lru_cache(1)  # Keeps frequently accessed images open
    def _load_image_npy(self, image_id):
        """
        Returns a memmapped reference to the entire image
        """
        gpath = self._lookup_gpath(image_id)
        gpath, cache_gpath = self._gnames(image_id, mode='npy')
        mem_gpath = cache_gpath

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

    def _gnames(self, image_id, mode):
        gpath = self._lookup_gpath(image_id)
        hashid = self._lookup_hashid(image_id)
        fname_base = os.path.basename(gpath).split('.')[0]
        # We actually don't want to write the image-id in the filename because
        # the same file may be in different datasets with different ids.
        if mode == 'cog':
            # cache_gname = '{}_{}_{}.cog.tiff'.format(image_id, fname_base, hashid)
            cache_gname = '{}_{}.cog.tiff'.format(fname_base, hashid)
        elif mode == 'npy':
            # cache_gname = '{}_{}_{}.npy'.format(image_id, fname_base, hashid)
            cache_gname = '{}_{}.npy'.format(fname_base, hashid)
        else:
            raise KeyError(mode)
        cache_gpath = join(self.cache_dpath, cache_gname)
        return gpath, cache_gpath

    @lru_cache(1)  # Keeps frequently accessed images open
    def _load_image_cog(self, image_id):
        """
        Returns a special array-like object with a COG GeoTIFF backend
        """
        gpath, cache_gpath = self._gnames(image_id, mode='cog')
        cog_gpath = cache_gpath

        if not exists(cog_gpath):
            if not exists(gpath):
                raise OSError('Source image gpath={!r} for image_id={!r} does not exist!'.format(gpath, image_id))

            # If the file already is a cog, just use it
            from ndsampler.validate_cog import validate as _validate_cog
            warnings, errors, details = _validate_cog(gpath)

            if 1:
                from multiprocessing import current_process
                DEBUG = 0
                if current_process().name == 'MainProcess':
                    DEBUG = 2
            else:
                DEBUG = 0

            if DEBUG:
                print('<DEBUG INFO>')
                if DEBUG > 2:
                    print('details = ' + ub.repr2(details))
                    print('warnings = ' + ub.repr2(warnings))
                    print('errors (why we need to ensure) = ' + ub.repr2(errors))
                print('BUILDING COG REPRESENTATION')
                print('gpath = {!r}'.format(gpath))
                if DEBUG > 1:
                    import netharn as nh
                    print(' * info(gpath) = ' + ub.repr2(nh.util.get_file_info(gpath)))

            gpath_is_cog = not bool(errors)
            if gpath_is_cog:
                # If we already are a cog, then just use it
                ub.symlink(gpath, cog_gpath)
            else:
                self._ensure_cog_representation(gpath, cog_gpath)

            if DEBUG:
                print('cog_gpath = {!r}'.format(cog_gpath))
                if DEBUG > 1:
                    import netharn as nh
                    print(' * info(cog_gpath) = ' + ub.repr2(nh.util.get_file_info(cog_gpath)))
                print('</DEBUG INFO>')

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
        >>> _cog_cache_write(gpath, cache_gpath)
    """
    # Load all the image data and dump it to npy format
    import kwimage
    raw_data = kwimage.imread(gpath)
    # TODO: THERE HAS TO BE A CORRECT WAY TO DO THIS.
    # However, I'm not sure what it is. I extend my appologies to whoever is
    # maintaining this code. Note: mode MUST be 'w'

    DEBUG = 1
    if DEBUG:
        import multiprocessing
        proc = multiprocessing.current_process()
        def _debug(msg):
            with open(cache_gpath + '.proxy.debug', 'a') as f:
                f.write('[{}, {}, {}] '.format(ub.timestamp(), proc, proc.pid) + msg + '\n')
        _debug('attempts aquire'.format())

    with atomicwrites.atomic_write(cache_gpath + '.proxy', mode='w', overwrite=True) as file:
        if DEBUG:
            _debug('begin')
            _debug('gpath = {}'.format(gpath))
            _debug('cache_gpath = {}'.format(cache_gpath))
        try:
            file.write('begin: {}\n'.format(ub.timestamp()))
            file.write('gpath = {}\n'.format(gpath))
            file.write('cache_gpath = {}\n'.format(cache_gpath))
            if not exists(cache_gpath):
                _imwrite_cloud_optimized_geotiff(cache_gpath, raw_data, lossy=True)
                if DEBUG:
                    _debug('finished write: {}\n'.format(ub.timestamp()))
            else:
                if DEBUG:
                    _debug('ALREADY EXISTS did not write: {}\n'.format(ub.timestamp()))
            file.write('end: {}\n'.format(ub.timestamp()))
        except Exception as ex:
            file.write('FAILED DUE TO EXCEPTION: {}: {}\n'.format(ex, ub.timestamp()))
            if DEBUG:
                _debug('FAILED DUE TO EXCEPTION: {}'.format(ex))
        finally:
            if DEBUG:
                _debug('finally')

    RUN_CORRUPTION_CHECKS = True
    if RUN_CORRUPTION_CHECKS:
        # CHECK THAT THE DATA WAS WRITTEN CORRECTLY
        file = LazyGDalFrameFile(cache_gpath)
        orig_sum = raw_data.sum()
        cache_sum = file[:].sum()
        if DEBUG:
            _debug('orig_sum = {}'.format(orig_sum))
            _debug('cache_sum = {}'.format(cache_sum))
        if orig_sum > 0 and cache_sum == 0:
            print('FAILED TO WRITE COG FILE')
            if DEBUG:
                _debug('FAILED TO WRITE COG FILE')
            ub.delete(cache_gpath)
            raise Exception('FAILED TO WRITE COG FILE CORRECTLY')


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
    lock_fpath = cache_gpath + '.lock'

    DEBUG = 1

    if DEBUG:
        import multiprocessing
        proc = multiprocessing.current_process()
        def _debug(msg):
            with open(lock_fpath + '.debug', 'a') as f:
                f.write('[{}, {}, {}] '.format(ub.timestamp(), proc, proc.pid) + msg + '\n')
        _debug('lock_fpath = {}'.format(lock_fpath))
        _debug('attempt aquire')

    try:
        # Ensure that another process doesn't write to the same image
        # FIXME: lockfiles may not be cleaned up gracefully
        # See: https://github.com/harlowja/fasteners/issues/26
        with fasteners.InterProcessLock(lock_fpath):

            if DEBUG:
                _debug('aquires')
                _debug('will read {}'.format(gpath))
                _debug('will write {}'.format(cache_gpath))

            if not exists(cache_gpath):
                _write_func(gpath, cache_gpath)

                if DEBUG:
                    _debug('wrote'.format())
            else:
                if DEBUG:
                    _debug('does not need to write'.format())

            if DEBUG:
                _debug('releasing'.format())

        if DEBUG:
            _debug('released'.format())
    except Exception as ex:
        if DEBUG:
            _debug('GOT EXCEPTION: {}'.format(ex))
    finally:
        if DEBUG:
            _debug('finally')


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
        from PIL import Image
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

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ndsampler.abstract_frames import *
        >>> from ndsampler.abstract_frames import _imwrite_cloud_optimized_geotiff
        >>> data = (np.random.rand(1000, 1000, 3) * 255).astype(np.uint8)
        >>> fpath = '/tmp/foo.cog.tiff'

        >>> _imwrite_cloud_optimized_geotiff(fpath, data, lossy=False)
        >>> import netharn as nh
        >>> print(ub.repr2(nh.util.get_file_info(fpath)))
        >>> from ndsampler.validate_cog import validate as _validate_cog
        >>> errors = _validate_cog(fpath)
        >>> warnings, errors, details = _validate_cog(fpath)
        >>> print('details = ' + ub.repr2(details))
        >>> print('warnings = ' + ub.repr2(warnings))
        >>> print('errors = ' + ub.repr2(errors))
        >>> assert not errors

        >>> _imwrite_cloud_optimized_geotiff(fpath, data, lossy=True)
        >>> import netharn as nh
        >>> print(ub.repr2(nh.util.get_file_info(fpath)))
        >>> from ndsampler.validate_cog import validate as _validate_cog
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
        >>> from ndsampler.abstract_frames import *
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
