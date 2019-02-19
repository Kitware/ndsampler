# -*- coding: utf-8 -*-
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
    """

    def __init__(self, id_to_hashid=None, hashid_mode='PATH', workdir=None):

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
            self._cache_dpath = ub.ensuredir(join(self.workdir, '_cache/frames'))
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

    @lru_cache(4)  # Keeps frequently accessed memmaps open
    def load_image(self, image_id, bands=None, scale=0, cache=True):
        """
        Returns a memmapped reference to the entire image
        """
        if bands is not None:
            raise NotImplementedError('cannot handle different bands')

        if scale != 0:
            raise NotImplementedError('can only handle scale=0')

        gpath = self._lookup_gpath(image_id)

        if not cache:
            raw_data = np.asarray(Image.open(gpath))
            return raw_data

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

    @staticmethod
    def ensure_npy_representation(gpath, mem_gpath):
        """
        Ensures that mem_gpath exists in a multiprocessing-safe way
        """
        import multiprocessing

        lock_fpath = mem_gpath + '.lock'

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
        lock_fpath = mem_gpath + '.lock'
        with fasteners.InterProcessLock(lock_fpath):

            if DEBUG:
                _debug('{} aquires'.format(procid))

            if not exists(mem_gpath):
                # Load all the image data and dump it to npy format
                raw_data = np.asarray(Image.open(gpath))

                # Even with the file-lock and atomic save there is still a race
                # condition somewhere. Not sure what it is.
                # _semi_atomic_numpy_save(mem_gpath, raw_data)

                with atomicwrites.atomic_write(mem_gpath, mode='wb', overwrite=True) as file:
                    np.save(file, raw_data)

                if DEBUG:
                    _debug('{} wrote'.format(procid))
            else:
                if DEBUG:
                    _debug('{} does not need to write'.format(procid))

            if DEBUG:
                _debug('{} releasing'.format(procid))

        if DEBUG:
            _debug('{} released'.format(procid))

    _load_subregion = load_region


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
        >>> import ndsampler
        >>> from ndsampler.abstract_frames import *
        >>> workdir = ub.ensure_app_cache_dir('ndsampler')
        >>> dset = ndsampler.CocoDataset.demo()
        >>> id_to_path = {img['id']: join(dset.img_root, img['file_name'])
        >>>               for img in dset.imgs.values()}
        >>> self = SimpleFrames(id_to_path, workdir=workdir)
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
        import kwil
        gpath = kwil.grab_test_image_fpath()
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
