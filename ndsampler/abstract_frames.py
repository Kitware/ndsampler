# -*- coding: utf-8 -*-
"""
Fast access to subregions of images.

This implements the core convert-and-cache-as-cog logic, which enables us to
read from subregions of images quickly.


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
import ubelt as ub
# import lockfile
# https://stackoverflow.com/questions/489861/locking-a-file-in-python
import fasteners  # supercedes lockfile / oslo_concurrency?
import atomicwrites
import six

import warnings
from os.path import exists
from os.path import join
from ndsampler.utils import util_gdal

if six.PY2:
    # TODO: find or make an LRU implementation where the maxsize can be
    # dynamically changed during runtime.
    from backports.functools_lru_cache import lru_cache
else:
    from functools import lru_cache

# try:
#     import xdev
#     profile = xdev.profile
# except ImportError:
#     profile = ub.identity


DEBUG_COG_ATOMIC_WRITE = 0
DEBUG_FILE_LOCK_CACHE_WRITE = 0
DEBUG_LOAD_COG = int(ub.argflag('--debug-load-cog'))
RUN_COG_CORRUPTION_CHECKS = True


class CorruptCOG(Exception):
    pass


class Frames(object):
    """
    Abstract implementation of Frames.

    While this is an abstract class, it contains most of the `Frames`
    functionality. The inheriting class needs to overload the constructor and
    `_lookup_gpath`, which maps an image-id to its path on disk.

    Args:
        id_to_hashid (Dict, optional):
            A custom mapping from image-id to a globally unique hashid for that
            image. Typically you should not specify this unless
            hashid_mode='GIVEN'.

        hashid_mode (str, default='PATH'): The method used to compute a unique
            identifier for every image. to can be PATH, PIXELS, or GIVEN.

        workdir (PathLike): This is the directory where `Frames` can store
            cached results. This SHOULD be specified.

        backend (str | Dict): Determine the backend to use for fast subimage
            region lookups. This can either be a string 'cog' or 'npy'. This
            can also be a config dictionary for fine-grained backend control.
            For this case, 'type': specified cog or npy, and only COG has
            additional options which are:
                {
                    'type': 'cog',
                    'config': {
                    'compress': <'LZW' | 'JPEG | 'DEFLATE' | 'auto'>,
                    }
                }

    Example:
        >>> from ndsampler.abstract_frames import *
        >>> self = SimpleFrames.demo(backend='npy')
        >>> file = self.load_image(1)
        >>> print('file = {!r}'.format(file))
        >>> assert self.load_image(1).shape == (512, 512, 3)
        >>> assert self.load_region(1, (slice(-20), slice(-10))).shape == (492, 502, 3)
        >>> # xdoctest: +REQUIRES(module:gdal)
        >>> self = SimpleFrames.demo(backend='cog')
        >>> assert self.load_image(1).shape == (512, 512, 3)
        >>> assert self.load_region(1, (slice(-20), slice(-10))).shape == (492, 502, 3)

    Benchmark:
        >>> import ubelt as ub
        >>> #
        >>> ti = ub.Timerit(100, bestof=3, verbose=2)
        >>> #
        >>> self = SimpleFrames.demo(backend='cog')
        >>> for timer in ti.reset('cog-small-subregion'):
        >>>     self.load_image(1)[10:42, 10:42]
        >>> #
        >>> self = SimpleFrames.demo(backend='npy')
        >>> for timer in ti.reset('npy-small-subregion'):
        >>>     self.load_image(1)[10:42, 10:42]
        >>> print('----')
        >>> #
        >>> self = SimpleFrames.demo(backend='cog')
        >>> for timer in ti.reset('cog-large-subregion'):
        >>>     self.load_image(1)[3:-3, 3:-3]
        >>> #
        >>> self = SimpleFrames.demo(backend='npy')
        >>> for timer in ti.reset('npy-large-subregion'):
        >>>     self.load_image(1)[3:-3, 3:-3]
        >>> print('----')
        >>> #
        >>> self = SimpleFrames.demo(backend='cog')
        >>> for timer in ti.reset('cog-loadimage'):
        >>>     self.load_image(1)
        >>> #
        >>> self = SimpleFrames.demo(backend='npy')
        >>> for timer in ti.reset('npy-loadimage'):
        >>>     self.load_image(1)
    """

    DEFAULT_NPY_CONFIG = {
        'type': 'npy',
        'config': {},
    }

    DEFAULT_COG_CONFIG = {
        'type': 'cog',
        'config': {
            'compress': 'auto',
        },
        '_hack_old_names': False,  # This will be removed in the future
        '_hack_use_cli': True,  # Uses the gdal-CLI to create cogs, which frustratingly seems to be faster
    }

    def __init__(self, id_to_hashid=None, hashid_mode='PATH', workdir=None,
                 backend='auto'):

        self._backend = None
        self._backend_hashid = None  # hash of backend config parameters
        self._cache_dpath = None

        self._update_backend(backend)

        if id_to_hashid is None:
            id_to_hashid = {}
        else:
            hashid_mode = 'GIVEN'

        self.id_to_hashid = id_to_hashid

        if workdir is None:
            workdir = ub.get_app_cache_dir('ndsampler')
            if self._backend['type'] is not None:
                warnings.warn('Frames workdir not specified. '
                              'Defaulting to {!r}'.format(workdir))
        self.workdir = workdir
        self.hashid_mode = hashid_mode

        if self.hashid_mode not in ['PATH', 'PIXELS', 'GIVEN']:
            raise KeyError(self.hashid_mode)

    def _update_backend(self, backend):
        """ change the backend and update internals accordingly """
        self._backend = self._coerce_backend_config(backend)
        self._cache_dpath = None
        self._backend_hashid = ub.hash_data(
            sorted(self._backend['config'].items()))[0:8]

    @classmethod
    def _coerce_backend_config(cls, backend='auto'):
        """
        Coerce a backend argument into a valid configuration dictionary.

        Returns:
            Dict: a dictionary with two items: 'type', which is a string and
                and 'config', which is a dictionary of parameters for the
                specific type.
        """
        import copy

        if backend is None:
            return {'type': None, 'config': {}}

        # TODO: allow for heterogeneous backends
        if backend == 'auto':
            # Use defaults that work on the system
            if util_gdal.have_gdal():
                backend = 'cog'
            else:
                backend = 'npy'
        if isinstance(backend, six.string_types):
            backend = {'type': backend, 'config': {}}

        backend_type = backend.get('type', None)
        if backend_type == 'cog':
            util_gdal._fix_conda_gdal_hack()
            final = copy.deepcopy(cls.DEFAULT_COG_CONFIG)
        elif backend_type == 'npy':
            final = copy.deepcopy(cls.DEFAULT_NPY_CONFIG)
        else:
            raise ValueError(
                'Backend dictionary must specify type as either npy or cog,'
                ' but we got {}'.format(backend_type)
            )

        # Check the top-level dictionary has no unknown keys
        unknown_keys = set(backend) - set(final)
        if unknown_keys:
            raise ValueError('Backend got unknown keys: {}'.format(unknown_keys))

        # Check the config-level dictionary has no unknown keys
        inner_kw = backend.get('config', {})
        default_kw = final['config']
        unknown_config_keys = set(inner_kw) - set(default_kw)
        if unknown_config_keys:
            raise ValueError('Backend config got unknown keys: {}'.format(unknown_config_keys))

        # Only update expected values in the subconfig
        # Update the outer config
        outer_kw = backend.copy()
        outer_kw.pop('config', None)
        final.update(outer_kw)

        # Update the inner config
        final['config'].update(inner_kw)
        return final

    @property
    def cache_dpath(self):
        """
        Returns the path where cached frame representations will be stored.

        This will be None if there is no backend.
        """
        if self._cache_dpath is None:
            backend_type = self._backend['type']
            if backend_type is None:
                return None
            elif backend_type == 'cog':
                if self._backend['_hack_old_names']:
                    # Old style didn't care about the particular config. Used whatever the first config was.
                    dpath = join(self.workdir, '_cache', 'frames',
                                 backend_type)
                else:
                    # New style respects user config choice
                    dpath = join(self.workdir, '_cache', 'frames',
                                 backend_type, self._backend_hashid)
            elif backend_type == 'npy':
                dpath = join(self.workdir, '_cache', 'frames', backend_type)
            else:
                raise KeyError(backend_type)
            self._cache_dpath = ub.ensuredir(dpath)
        return self._cache_dpath

    def _gnames(self, image_id, mode=None):
        """
        Lookup the original image name and its cached efficient representation
        name
        """
        gpath = self._lookup_gpath(image_id)
        if mode is None:
            if self._backend is None:
                return gpath, None
            else:
                mode = self._backend['type']
        hashid = self._lookup_hashid(image_id)
        fname_base = os.path.basename(gpath).split('.')[0]
        # We actually don't want to write the image-id in the filename because
        # the same file may be in different datasets with different ids.
        if mode == 'cog':
            cache_gname = '{}_{}.cog.tiff'.format(fname_base, hashid)
        elif mode == 'npy':
            cache_gname = '{}_{}.npy'.format(fname_base, hashid)
        else:
            raise KeyError(mode)
        cache_gpath = join(self.cache_dpath, cache_gname)
        return gpath, cache_gpath

    def _lookup_gpath(self, image_id):
        """
        A user specified function that maps an image id to its path on disk.

        Args:
            image_id: the image id (usually an integer)

        Returns:
            PathLike: path to the image
        """
        raise NotImplementedError

    @property
    def image_ids(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        return self.load_image(image_id)

    def _lookup_hashid(self, image_id):
        """
        Get the hashid of a particular image.

        NOTE: THIS IS OVERWRITTEN BY THE COCO FRAMES TO ONLY USE THE RELATIVE
        FILE PATH
        """
        if image_id not in self.id_to_hashid:
            # Compute the hash if we it does not exist yet
            gpath = self._lookup_gpath(image_id)
            if self.hashid_mode == 'PATH':
                # Hash the full path to the image data
                # NOTE: this logic is not machine independent
                hashid = ub.hash_data(gpath, hasher='sha1', base='hex')
            elif self.hashid_mode == 'PIXELS':
                # Hash the pixels in the image
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
        Ammortized O(1) image subregion loading (assuming constant region size)

        Args:
            image_id (int): image identifier
            region (Tuple[slice, ...]): space-time region within an image
            bands (str): NotImplemented
            scale (float): NotImplemented
        """
        region = self._rectify_region(region)
        if all(r.start is None and r.stop is None for r in region):
            # Avoid forcing a cache computation when loading the full image
            im = self.load_image(image_id, bands=bands, scale=scale,
                                 cache=False)
        else:
            file = self.load_image(image_id, bands=bands, scale=scale,
                                   cache=True)
            im = file[region]
        return im

    def load_frame(self, image_id):
        """
        TODO: FINISHME

        Returns a frame object that lazy loads on slice
        """
        class LazyFrame(object):
            def __init__(self, frames, image_id):
                self._data = None
                self._frames = frames
                self._image_id = image_id

            def __getitem__(self, region):
                if self._data is None:
                    if all(r.start is None and r.stop is None for r in region):
                        # Avoid forcing a cache computation when loading the full image
                        self._data = self._frames.load_image(self._image_id,
                                                             cache=False)
                    else:
                        self._data = self._frames.load_image(self._image_id,
                                                             cache=True)
                return self._frame[region]
        im = LazyFrame(self, image_id)
        return im

    def _rectify_region(self, region):
        if region is None:
            region = tuple([])
        if len(region) < 2:
            # Add empty dimensions
            tail = tuple([slice(None)] * (2 - len(region)))
            region = tuple(region) + tail
        return region

    def load_image(self, image_id, bands=None, scale=0, cache=True,
                   noreturn=False):
        """
        Load the image data for a particular image id

        Args:
            image_id (int): the id of the image to load
            cache (bool, default=True): ensure and return the efficient backend
                cached representation.
            bands : NotImplemented
            scale : NotImplemented

            noreturn (bool, default=False): if True, nothing is returned.
                This is useful if you simply want to ensure the cached
                representation.

        Returns:
            ArrayLike: an indexable array like representation, possibly
                memmapped.
        """
        if bands is not None:
            raise NotImplementedError('cannot handle different bands')

        if scale != 0:
            raise NotImplementedError('can only handle scale=0')

        if not cache :
            import kwimage
            gpath = self._lookup_gpath(image_id)
            data = kwimage.imread(gpath)
        else:
            if self._backend['type'] is None:
                data = self._load_image_full(image_id)
            elif self._backend['type'] == 'cog':
                data = self._load_image_cog(image_id)
            elif self._backend['type'] == 'npy':
                data = self._load_image_npy(image_id)
            else:
                raise KeyError(self._backend['type'])
        if noreturn:
            data = None
        return data

    @lru_cache(1)  # Keeps frequently accessed images open
    def _load_image_full(self, image_id):
        import kwimage
        gpath = self._lookup_gpath(image_id)
        raw_data = kwimage.imread(gpath)
        return raw_data

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
                except Exception as ex:
                    print('\n\n')
                    print('ERROR: FAILED TO LOAD CACHED FILE')
                    print('ex = {!r}'.format(ex))
                    print('mem_gpath = {!r}'.format(mem_gpath))
                    print('exists(mem_gpath) = {!r}'.format(exists(mem_gpath)))
                    print('Recompute cache: Try number {}'.format(try_num))
                    print('\n\n')

                    if try_num > 20:
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
            from ndsampler.utils.validate_cog import validate as _validate_cog
            warnings, errors, details = _validate_cog(gpath)

            if DEBUG_LOAD_COG:
                from multiprocessing import current_process
                from threading import current_thread
                is_main = (
                    current_thread().name == 'MainThread' and
                    current_process().name == 'MainProcess'
                )
                DEBUG = 0
                if is_main:
                    DEBUG = 2
            else:
                DEBUG = 0

            if DEBUG:
                print('\n<DEBUG INFO>')
                print('Missing cog_gpath={}'.format(cog_gpath))
                print('gpath = {!r}'.format(gpath))

            gpath_is_cog = not bool(errors)
            if ub.argflag('--debug-validate-cog') or DEBUG > 2:
                print('details = {}'.format(ub.repr2(details, nl=1)))
                print('errors (why we need to ensure) = {}'.format(ub.repr2(errors, nl=1)))
                print('warnings = {}'.format(ub.repr2(warnings, nl=1)))
                print('gpath_is_cog = {!r}'.format(gpath_is_cog))

            if gpath_is_cog:
                # If we already are a cog, then just use it
                if DEBUG:
                    print('Image is already a cog, symlinking to cache')
                ub.symlink(gpath, cog_gpath)
            else:
                config = self._backend['config'].copy()
                config['hack_use_cli'] = self._backend['_hack_use_cli']
                if DEBUG:
                    print('BUILDING COG REPRESENTATION')
                    print('gpath = {!r}'.format(gpath))
                    print('cog config = {}'.format(ub.repr2(config, nl=2)))
                    if DEBUG > 1:
                        try:
                            import netharn as nh
                            print(' * info(gpath) = ' + ub.repr2(nh.util.get_file_info(gpath)))
                        except ImportError:
                            pass
                self._ensure_cog_representation(gpath, cog_gpath, config)

            if DEBUG:
                print('cog_gpath = {!r}'.format(cog_gpath))
                if DEBUG > 1:
                    try:
                        import netharn as nh
                        print(' * info(cog_gpath) = ' + ub.repr2(nh.util.get_file_info(cog_gpath)))
                    except ImportError:
                        pass
                print('</DEBUG INFO>')

        file = util_gdal.LazyGDalFrameFile(cog_gpath)
        return file

    @staticmethod
    def _ensure_cog_representation(gpath, cog_gpath, config):
        try:
            _locked_cache_write(_cog_cache_write, gpath, cache_gpath=cog_gpath,
                                config=config)
        except CorruptCOG:
            import warnings
            msg = ('!!! COG was corrupted, trying once more')
            warnings.warn(msg)
            print(msg)
            _locked_cache_write(_cog_cache_write, gpath, cache_gpath=cog_gpath,
                                config=config)

    @staticmethod
    def ensure_npy_representation(gpath, mem_gpath, config=None):
        """
        Ensures that mem_gpath exists in a multiprocessing-safe way
        """
        _locked_cache_write(_npy_cache_write, gpath, cache_gpath=mem_gpath,
                            config=config)

    def prepare(self, workers=0, use_stamp=True):
        """
        Precompute the cached frame conversions

        Args:
            workers (int, default=0): number of parallel threads for this
                io-bound task

        Example:
            >>> from ndsampler.abstract_frames import *
            >>> workdir = ub.ensure_app_cache_dir('ndsampler/tests/test_cog_precomp')
            >>> print('workdir = {!r}'.format(workdir))
            >>> ub.delete(workdir)
            >>> ub.ensuredir(workdir)
            >>> self = SimpleFrames.demo(backend='npy', workdir=workdir)
            >>> print('self = {!r}'.format(self))
            >>> print('self.cache_dpath = {!r}'.format(self.cache_dpath))
            >>> #_ = ub.cmd('tree ' + workdir, verbose=3)
            >>> self.prepare()
            >>> self.prepare()
            >>> #_ = ub.cmd('tree ' + workdir, verbose=3)
            >>> _ = ub.cmd('ls ' + self.cache_dpath, verbose=3)

        Example:
            >>> from ndsampler.abstract_frames import *
            >>> import ndsampler
            >>> workdir = ub.get_app_cache_dir('ndsampler/tests/test_cog_precomp2')
            >>> ub.delete(workdir)
            >>> # TEST NPY
            >>> #
            >>> sampler = ndsampler.CocoSampler.demo(workdir=workdir, backend='npy')
            >>> self = sampler.frames
            >>> ub.delete(self.cache_dpath)  # reset
            >>> self.prepare()  # serial, miss
            >>> self.prepare()  # serial, hit
            >>> ub.delete(self.cache_dpath)  # reset
            >>> self.prepare(workers=3)  # parallel, miss
            >>> self.prepare(workers=3)  # parallel, hit
            >>> #
            >>> ## TEST COG
            >>> # xdoctest: +REQUIRES(module:gdal)
            >>> sampler = ndsampler.CocoSampler.demo(workdir=workdir, backend='cog')
            >>> self = sampler.frames
            >>> ub.delete(self.cache_dpath)  # reset
            >>> self.prepare()  # serial, miss
            >>> self.prepare()  # serial, hit
            >>> ub.delete(self.cache_dpath)  # reset
            >>> self.prepare(workers=3)  # parallel, miss
            >>> self.prepare(workers=3)  # parallel, hit
        """
        if self.cache_dpath is None:
            print('Frames backend is None, skip prepare')
            return

        ub.ensuredir(self.cache_dpath)
        # Note: this usually acceses the hashid attribute of util.HashIdentifiable
        hashid = getattr(self, 'hashid', None)

        # TODO:
        #     Add some image preprocessing ability here
        stamp = ub.CacheStamp('prepare_frames_stamp', dpath=self.cache_dpath,
                              cfgstr=hashid, verbose=3)
        stamp.cacher.enabled = bool(hashid) and bool(use_stamp)

        # print('frames stamp hashid = {!r}'.format(hashid))
        # print('frames cache_dpath = {!r}'.format(self.cache_dpath))
        # print('stamp.cacher.enabled = {!r}'.format(stamp.cacher.enabled))

        if stamp.expired() or hashid is None:
            from ndsampler.utils import util_futures
            from concurrent import futures
            # Use thread mode, because we are mostly in doing io.
            executor = util_futures.Executor(mode='thread', max_workers=workers)
            with executor as executor:
                job_list = []
                gids = self.image_ids
                for image_id in ub.ProgIter(gids, desc='Frames: submit prepare jobs'):
                    gpath, cache_gpath = self._gnames(image_id)
                    if not exists(cache_gpath):
                        job = executor.submit(
                            self.load_image, image_id, cache=True,
                            noreturn=True)
                        job_list.append(job)

                for job in ub.ProgIter(futures.as_completed(job_list),
                                       total=len(job_list), adjust=False, freq=1,
                                       desc='Frames: collect prepare jobs'):
                    job.result()
            stamp.renew()


# @profile
def _cog_cache_write(gpath, cache_gpath, config=None):
    """
    CommandLine:
        xdoctest -m ndsampler.abstract_frames _cog_cache_write

    Example:
        >>> # xdoctest: +REQUIRES(module:gdal)
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
        >>> _cog_cache_write(gpath, cache_gpath, {})
    """
    assert config is not None
    hack_use_cli = config.pop('hack_use_cli', False)

    if DEBUG_COG_ATOMIC_WRITE:
        import multiprocessing
        from multiprocessing import current_process
        from threading import current_thread
        is_main = (current_thread().name == 'MainThread' and
                   current_process().name == 'MainProcess')
        proc = multiprocessing.current_process()
        def _debug(msg):
            msg_ = '[{}, {}, {}] '.format(ub.timestamp(), proc, proc.pid) + msg + '\n'
            if is_main:
                print(msg_)
            with open(cache_gpath + '.atomic.debug', 'a') as f:
                f.write(msg_)
        _debug('attempts aquire'.format())

    if not hack_use_cli:
        # Load all the image data and dump it to cog format
        import kwimage
        if DEBUG_COG_ATOMIC_WRITE:
            _debug('reading data')
        raw_data = kwimage.imread(gpath)
        # raw_data = kwimage.atleast_3channels(raw_data, copy=False)
    # TODO: THERE HAS TO BE A CORRECT WAY TO DO THIS.
    # However, I'm not sure what it is. I extend my appologies to whoever is
    # maintaining this code. Note: mode MUST be 'w'

    with atomicwrites.atomic_write(cache_gpath + '.atomic', mode='w', overwrite=True) as file:
        if DEBUG_COG_ATOMIC_WRITE:
            _debug('begin')
            _debug('gpath = {}'.format(gpath))
            _debug('cache_gpath = {}'.format(cache_gpath))
        try:
            file.write('begin: {}\n'.format(ub.timestamp()))
            file.write('gpath = {}\n'.format(gpath))
            file.write('cache_gpath = {}\n'.format(cache_gpath))
            if not exists(cache_gpath):
                if not hack_use_cli:
                    util_gdal._imwrite_cloud_optimized_geotiff(
                        cache_gpath, raw_data, **config)
                else:
                    # The CLI is experimental and might make this pipeline
                    # faster by avoiding the initial read.
                    util_gdal._cli_convert_cloud_optimized_geotiff(
                        gpath, cache_gpath, **config)
                if DEBUG_COG_ATOMIC_WRITE:
                    _debug('finished write: {}\n'.format(ub.timestamp()))
            else:
                if DEBUG_COG_ATOMIC_WRITE:
                    _debug('ALREADY EXISTS did not write: {}\n'.format(ub.timestamp()))
            file.write('end: {}\n'.format(ub.timestamp()))
        except Exception as ex:
            file.write('FAILED DUE TO EXCEPTION: {}: {}\n'.format(ex, ub.timestamp()))
            if DEBUG_COG_ATOMIC_WRITE:
                _debug('FAILED DUE TO EXCEPTION: {}'.format(ex))
            raise
        finally:
            if DEBUG_COG_ATOMIC_WRITE:
                _debug('finally')

    if RUN_COG_CORRUPTION_CHECKS:
        # CHECK THAT THE DATA WAS WRITTEN CORRECTLY
        file = util_gdal.LazyGDalFrameFile(cache_gpath)
        is_valid = util_gdal.validate_gdal_file(file)
        if not is_valid:
            if hack_use_cli:
                import kwimage
                raw_data = kwimage.imread(gpath)
            # The check may fail on zero images, so check that
            orig_sum = raw_data.sum()
            # cache_sum = file[:].sum()
            # if DEBUG:
            #     _debug('is_valid = {}'.format(is_valid))
            #     _debug('cache_sum = {}'.format(cache_sum))
            if orig_sum > 0:
                print('FAILED TO WRITE COG FILE')
                print('orig_sum = {!r}'.format(orig_sum))
                # print(kwimage.imread(cache_gpath).sum())
                if DEBUG_COG_ATOMIC_WRITE:
                    _debug('FAILED TO WRITE COG FILE')
                ub.delete(cache_gpath)
                raise CorruptCOG('FAILED TO WRITE COG FILE CORRECTLY')

    # raise RuntimeError('FOOBAR')


def _npy_cache_write(gpath, cache_gpath, config=None):
    # Load all the image data and dump it to npy format
    import kwimage
    raw_data = kwimage.imread(gpath)
    # raw_data = np.asarray(Image.open(gpath))

    # Even with the file-lock and atomic save there is still a race
    # condition somewhere. Not sure what it is.
    # _semi_atomic_numpy_save(cache_gpath, raw_data)

    # with atomicwrites.atomic_write(cache_gpath, mode='wb', overwrite=True) as file:
    with open(cache_gpath, mode='wb') as file:
        np.save(file, raw_data)


def _locked_cache_write(_write_func, gpath, cache_gpath, config=None):
    """
    Ensures that mem_gpath exists in a multiprocessing-safe way
    """
    lock_fpath = cache_gpath + '.lock'

    if DEBUG_FILE_LOCK_CACHE_WRITE:
        import multiprocessing
        from multiprocessing import current_process
        from threading import current_thread
        is_main = (current_thread().name == 'MainThread' and
                   current_process().name == 'MainProcess')
        proc = multiprocessing.current_process()
        def _debug(msg):
            msg_ = '[{}, {}, {}] '.format(ub.timestamp(), proc, proc.pid) + msg + '\n'
            if is_main:
                print(msg_)
            with open(lock_fpath + '.debug', 'a') as f:
                f.write(msg_)
        _debug('lock_fpath = {}'.format(lock_fpath))
        _debug('attempt aquire')

    try:
        # Ensure that another process doesn't write to the same image
        # FIXME: lockfiles may not be cleaned up gracefully
        # See: https://github.com/harlowja/fasteners/issues/26
        with fasteners.InterProcessLock(lock_fpath):

            if DEBUG_FILE_LOCK_CACHE_WRITE:
                _debug('aquires')
                _debug('will read {}'.format(gpath))
                _debug('will write {}'.format(cache_gpath))

            if not exists(cache_gpath):
                _write_func(gpath, cache_gpath, config)

                if DEBUG_FILE_LOCK_CACHE_WRITE:
                    _debug('wrote'.format())
            else:
                if DEBUG_FILE_LOCK_CACHE_WRITE:
                    _debug('does not need to write'.format())

            if DEBUG_FILE_LOCK_CACHE_WRITE:
                _debug('releasing'.format())

        if DEBUG_FILE_LOCK_CACHE_WRITE:
            _debug('released'.format())
    except Exception as ex:
        if DEBUG_FILE_LOCK_CACHE_WRITE:
            _debug('GOT EXCEPTION: {}'.format(ex))
    finally:
        if DEBUG_FILE_LOCK_CACHE_WRITE:
            _debug('finally')


class SimpleFrames(Frames):
    """
    Basic concrete implementation of frames objects

    Args:
        id_to_path (Dict): mapping from image-id to image path

    Example:
        >>> from ndsampler.abstract_frames import *
        >>> self = SimpleFrames.demo(backend='npy')
        >>> assert self.load_image(1).shape == (512, 512, 3)
        >>> assert self.load_region(1, (slice(-20), slice(-10))).shape == (492, 502, 3)
    """
    def __init__(self, id_to_path, id_to_hashid=None, hashid_mode='PATH',
                 workdir=None, **kw):
        super(SimpleFrames, self).__init__(id_to_hashid=id_to_hashid,
                                           hashid_mode=hashid_mode,
                                           workdir=workdir, **kw)
        self.id_to_path = id_to_path

    def _lookup_gpath(self, image_id):
        return self.id_to_path[image_id]

    @ub.memoize_property
    def image_ids(self):
        return list(self.id_to_path.keys())

    @classmethod
    def demo(self, **kw):
        """
        Get a smple frames object
        """
        import ndsampler
        dset = ndsampler.CocoDataset.demo()
        imgs = dset.images()
        id_to_name = imgs.lookup('file_name', keepid=True)
        id_to_path = {gid: join(dset.img_root, name)
                      for gid, name in id_to_name.items()}
        if kw.get('workdir', None) is None:
            kw['workdir'] = ub.ensure_app_cache_dir('ndsampler')
        self = SimpleFrames(id_to_path, **kw)
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
