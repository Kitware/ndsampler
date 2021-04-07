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
import ubelt as ub
import os
import six
import copy

import warnings
from os.path import exists, join, dirname
from ndsampler.utils import util_gdal
from ndsampler.utils import util_lru
from ndsampler.frame_cache import (
    _cog_cache_write, _npy_cache_write, _locked_cache_write, CorruptCOG)


DEBUG_LOAD_COG = int(ub.argflag('--debug-load-cog'))


try:
    from xdev import profile
except Exception:
    profile = ub.identity


class Frames(object):
    """
    Abstract implementation of Frames.

    While this is an abstract class, it contains most of the ``Frames``
    functionality. The inheriting class needs to overload the constructor and
    ``_lookup_gpath``, which maps an image-id to its path on disk.

    Args:
        hashid_mode (str, default='PATH'): The method used to compute a unique
            identifier for every image. to can be PATH, PIXELS, or GIVEN.
            TODO: Add DVC as a method (where it uses the name of the symlink)?

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
                    'compress': <'LZW' | 'JPEG | 'DEFLATE' | 'ZSTD' | 'auto'>,
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
        '_hack_use_cli': True,  # Uses the gdal-CLI to create cogs, which frustratingly seems to be faster
    }

    def __init__(self, hashid_mode='PATH', workdir=None, backend='auto'):

        self._backend = None
        self._backend_hashid = None  # hash of backend config parameters
        self._cache_dpath = None

        # keep an lru cache for repeated access to the same data
        self._lru = util_lru.LRUDict.new(max_size=1, impl='auto')

        self._update_backend(backend)

        # This is a cache that will be populated on the fly
        self._id_to_pathinfo = {}

        if workdir is None:
            workdir = ub.get_app_cache_dir('ndsampler')
            if self._backend['type'] is not None:
                warnings.warn('Frames workdir not specified. '
                              'Defaulting to {!r}'.format(workdir))
        self.workdir = workdir
        self.hashid_mode = hashid_mode

        if self.hashid_mode not in ['PATH', 'PIXELS', 'GIVEN']:
            # TODO: DVC
            raise KeyError(self.hashid_mode)

    def __getstate__(self):
        # don't carry the LRU cache throuch pickle operations
        state = self.__dict__.copy()
        if state['_lru'] is not None:
            state['_lru'] = True
        return state

    def __setstate__(self, state):
        if state['_lru'] is True:
            state['_lru'] = util_lru.LRUDict.new(max_size=1, impl='auto')
        self.__dict__.update(state)

    def _update_backend(self, backend):
        """ change the backend and update internals accordingly """
        self._backend = self._coerce_backend_config(backend)
        self._cache_dpath = None
        self._backend_hashid = ub.hash_data(
            sorted(self._backend['config'].items()))[0:8]
        self._id_to_pathinfo = {}

    @classmethod
    def _coerce_backend_config(cls, backend='auto'):
        """
        Coerce a backend argument into a valid configuration dictionary.

        Returns:
            Dict: a dictionary with two items: 'type', which is a string and
                and 'config', which is a dictionary of parameters for the
                specific type.
        """
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
            raise ValueError('Backend config got unknown keys: {}'.format(
                unknown_config_keys))

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
                dpath = join(self.workdir, '_cache', 'frames', backend_type,
                             self._backend_hashid)
            elif backend_type == 'npy':
                dpath = join(self.workdir, '_cache', 'frames', backend_type)
            else:
                raise KeyError('backend_type = {}'.format(backend_type))
            self._cache_dpath = ub.ensuredir(dpath)
        return self._cache_dpath

    def _build_pathinfo(self, image_id):
        """
        A user specified function that maps an image id to paths to relevant
        resources on disk. These resources are also indexed by channel.

        SeeAlso:
            ``_populate_chan_info`` for helping populate cache info in each
            channel.

        Args:
            image_id: the image id (usually an integer)

        Returns:
            Dict: with the following structure:
                {
                    <NotFinalized>
                    'channels': {
                        <channel_spec>: {'path': <abspath>, ...},
                        ...
                    }
                }
        """
        raise NotImplementedError

    def _lookup_pathinfo(self, image_id):
        if image_id in self._id_to_pathinfo:
            pathinfo = self._id_to_pathinfo[image_id]
        else:
            pathinfo = self._build_pathinfo(image_id)
            self._id_to_pathinfo[image_id] = pathinfo
        return pathinfo

    def _populate_chan_info(self, chan, root=''):
        """
        Helper to construct a path dictionary in the ``_build_pathinfo`` method
        based on the current hashing and caching settings.
        """
        backend_type = self._backend['type']

        if 'file_name' in chan:
            fname = chan['file_name']
            gpath = join(root, fname)
            chan['path'] = gpath
        elif 'path' in chan:
            gpath = chan['path']
            fname = gpath
        else:
            raise Exception('no file_name or path info')

        if backend_type is not None:
            if backend_type == 'cog':
                ext = '.cog.tiff'
            elif backend_type == 'npy':
                ext = '.npy'
            else:
                raise KeyError('backend_type = {}'.format(backend_type))

            hashid_mode = self.hashid_mode

            # hash directory structure
            cache_dpath = self.cache_dpath
            hashid = self._build_file_hashid(root, fname, hashid_mode)
            cache_gpath = join(cache_dpath, hashid[0:2], hashid[2:] + ext)
            chan['hashid'] = hashid
            chan['hashid_mode'] = hashid_mode
            chan['cache'] = cache_gpath

        chan['cache_type'] = backend_type

    @staticmethod
    def _build_file_hashid(root, suffix, hashid_mode):
        """
        Build a hashid for a specific file given as a path root and suffix.
        """
        gpath = join(root, suffix)
        if hashid_mode == 'PATH':
            # Hash the full path to the image data
            # NOTE: this logic is not machine independent
            hashid = ub.hash_data(suffix, hasher='sha1', base='hex')
        elif hashid_mode == 'PIXELS':
            # Hash the pixels in the image
            hashid = ub.hash_file(gpath, hasher='sha1', base='hex')
        elif hashid_mode == 'DVC':
            raise NotImplementedError('todo')
        elif hashid_mode == 'GIVEN':
            raise Exception('given mode no longer supported')
        else:
            raise KeyError(hashid_mode)
        return hashid

    @property
    def image_ids(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        return self.load_image(image_id)

    @profile
    def load_region(self, image_id, region=None, channels=None, scale=0,
                    width=None, height=None):
        """
        Ammortized O(1) image subregion loading (assuming constant region size)

        Args:
            image_id (int): image identifier
            region (Tuple[slice, ...]): space-time region within an image
            channels (str): NotImplemented
            scale (float): NotImplemented
            width (int): if the width of the entire image is know specify it
            height (int): if the height of the entire image is know specify it
        """
        if region is not None:
            if len(region) < 2:
                # Add empty dimensions
                tail = tuple([slice(None)] * (2 - len(region)))
                region = tuple(region) + tail
            if all(r.start is None and r.stop is None for r in region):
                # Avoid forcing a cache computation when loading the full image
                flag = (
                    region[0].stop in [None, height] and
                    region[1].stop in [None, width]
                )
                if flag:
                    region = None

        # setting region to None disables memmap/geotiff caching
        cache = region is not None
        file = self.load_image(image_id, channels=channels, scale=scale,
                               cache=cache)
        if region is not None:
            im = file[region]
        return im

    def load_frame(self, image_id):
        """
        TODO: FINISHME or rename to lazy frame?

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
        return region

    @profile
    def load_image(self, image_id, channels=None, scale=0, cache=True,
                   noreturn=False):
        """
        Load the image data for a particular image id

        Args:
            image_id (int): the id of the image to load
            cache (bool, default=True): ensure and return the efficient backend
                cached representation.
            channels : NotImplemented
            scale : NotImplemented

            noreturn (bool, default=False): if True, nothing is returned.
                This is useful if you simply want to ensure the cached
                representation.

        Returns:
            ArrayLike: an indexable array like representation, possibly
                memmapped.
        """
        if scale != 0:
            raise NotImplementedError('can only handle scale=0')

        pathinfo = self._lookup_pathinfo(image_id)
        if channels is None:
            channels = pathinfo['default']

        chan = pathinfo['channels'][channels]
        gpath = chan['path']

        if not cache :
            import kwimage
            data = kwimage.imread(gpath)
        else:
            cache_key = (image_id, channels)

            if self._lru is not None:
                if cache_key in self._lru:
                    file = self._lru[cache_key]
                    return file

            pathinfo['channels'][channels]
            if self._backend['type'] != chan['cache_type']:
                raise Exception('inconsistent backend')

            if chan['cache_type'] is None:
                import kwimage
                file = kwimage.imread(gpath)
            elif chan['cache_type'] == 'cog':
                cache_gpath = chan['cache']
                file = self._ensure_image_cog(gpath, cache_gpath)
            elif chan['cache_type'] == 'npy':
                cache_gpath = chan['cache']
                file = self._ensure_image_npy(gpath, cache_gpath)
            else:
                raise KeyError(self._backend['type'])

            data = file

            if self._lru is not None:
                self._lru[cache_key] = data

        if noreturn:
            data = None
        return data

    def _ensure_image_npy(self, gpath, cache_gpath):
        """
        Ensures that cache_gpath exists in a multiprocessing-safe way

        Returns a memmapped reference to the entire image
        """
        cache_gpath = cache_gpath

        if not exists(cache_gpath):
            ub.ensuredir(dirname(cache_gpath))
            _locked_cache_write(_npy_cache_write, gpath,
                                cache_gpath=cache_gpath, config=None)

        # I can't figure out why a race condition exists here.  I can't
        # reproduce it with a small example. In the meantime, this hack should
        # mitigate the problem by simply trying again until it works.
        HACKY_RACE_CONDITION_FIX = True
        if HACKY_RACE_CONDITION_FIX:
            for try_num in it.count():
                try:
                    file = np.load(cache_gpath, mmap_mode='r')
                except Exception as ex:
                    print('\n\n')
                    print('ERROR: FAILED TO LOAD CACHED FILE')
                    print('ex = {!r}'.format(ex))
                    print('cache_gpath = {!r}'.format(cache_gpath))
                    print('exists(cache_gpath) = {!r}'.format(exists(cache_gpath)))
                    print('Recompute cache: Try number {}'.format(try_num))
                    print('\n\n')

                    if try_num > 20:
                        # Something really bad must be happening, stop trying
                        raise
                    try:
                        os.remove(cache_gpath)
                    except Exception:
                        print('WARNING: CANNOT REMOVE CACHED FRAME.')
                    _locked_cache_write(_npy_cache_write, gpath,
                                        cache_gpath=cache_gpath, config=None)
                else:
                    break
        else:
            # FIXME;
            # There seems to be a race condition that triggers the error:
            # ValueError: mmap length is greater than file size
            try:
                file = np.load(cache_gpath, mmap_mode='r')
            except ValueError:
                print('\n\n')
                print('ERROR')
                print('cache_gpath = {!r}'.format(cache_gpath))
                print('exists(cache_gpath) = {!r}'.format(exists(cache_gpath)))
                print('\n\n')
                raise
        return file

    def _ensure_image_cog(self, gpath, cache_gpath):
        """
        Returns a special array-like object with a COG GeoTIFF backend
        """
        cog_gpath = cache_gpath

        if not exists(cog_gpath):
            if not exists(gpath):
                raise OSError((
                    'Source image gpath={!r} '
                    'does not exist!').format(gpath))

            ub.ensuredir(dirname(cog_gpath))

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
                try:
                    _locked_cache_write(_cog_cache_write, gpath,
                                        cache_gpath=cog_gpath, config=config)
                except CorruptCOG:
                    import warnings
                    msg = ('!!! COG was corrupted, trying once more')
                    warnings.warn(msg)
                    print(msg)
                    _locked_cache_write(_cog_cache_write, gpath,
                                        cache_gpath=cog_gpath, config=config)

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

    def prepare(self, gids=None, workers=0, use_stamp=True):
        """
        Precompute the cached frame conversions

        Args:
            gids (List[int] | None): specific image ids to prepare.
                If None prepare all images.
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
        #     Add some image preprocessing ability here?
        stamp = ub.CacheStamp('prepare_frames_stamp_v2', dpath=self.cache_dpath,
                              depends=hashid, verbose=3)
        stamp.cacher.enabled = bool(hashid) and bool(use_stamp) and gids is None

        if stamp.expired() or hashid is None:
            from ndsampler.utils import util_futures
            from concurrent import futures
            # Use thread mode, because we are mostly in doing io.
            executor = util_futures.Executor(mode='thread', max_workers=workers)
            with executor as executor:
                if gids is None:
                    gids = self.image_ids

                missing_cache_infos = []
                for gid in ub.ProgIter(gids, desc='lookup missing cache paths'):
                    pathinfo = self._lookup_pathinfo(gid)
                    for chan in pathinfo['channels'].values():
                        if not exists(chan['cache']):
                            missing_cache_infos.append((gid, chan['channels']))

                prog = ub.ProgIter(missing_cache_infos,
                                   desc='Frames: submit prepare jobs')
                job_list = [
                    executor.submit(
                        self.load_image, image_id, channels=channels,
                        cache=True, noreturn=True)
                    for image_id, channels in prog]

                for job in ub.ProgIter(futures.as_completed(job_list),
                                       total=len(job_list), adjust=False, freq=1,
                                       desc='Frames: collect prepare jobs'):
                    job.result()
            stamp.renew()


class SimpleFrames(Frames):
    """
    Basic concrete implementation of frames objects for images where there is a
    strict one-file-to-one-image mapping (i.e. no auxiliary images).

    Args:
        id_to_path (Dict): mapping from image-id to image path

    Example:
        >>> from ndsampler.abstract_frames import *
        >>> self = SimpleFrames.demo(backend='npy')
        >>> pathinfo = self._build_pathinfo(1)
        >>> print('pathinfo = {}'.format(ub.repr2(pathinfo, nl=3)))

        >>> assert self.load_image(1).shape == (512, 512, 3)
        >>> assert self.load_region(1, (slice(-20), slice(-10))).shape == (492, 502, 3)
    """
    def __init__(self, id_to_path, workdir=None, backend='auto'):
        super(SimpleFrames, self).__init__(
            hashid_mode='PATH', workdir=workdir, backend=backend)
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
        import kwcoco
        dset = kwcoco.CocoDataset.demo()
        id_to_path = {
            gid: dset.get_image_fpath(gid) for gid in dset.imgs.keys()
        }
        if kw.get('workdir', None) is None:
            kw['workdir'] = ub.ensure_app_cache_dir('ndsampler')
        self = SimpleFrames(id_to_path, **kw)
        return self

    def _build_pathinfo(self, image_id):
        pathinfo = {
            'id': image_id,
            'path': self.id_to_path[image_id],
            'channels': None,
            'transform': None,
        }
        pathinfo = {
            'default': None,
            'channels': {
                None: pathinfo,
            }
        }
        for chan in pathinfo['channels'].values():
            self._populate_chan_info(chan, root='')
        return pathinfo
