"""
Tools for caching intermediate frame representations.
"""
# import lockfile
# https://stackoverflow.com/questions/489861/locking-a-file-in-python
from os.path import dirname
import fasteners  # supercedes lockfile / oslo_concurrency?
import atomicwrites
import ubelt as ub
import numpy as np
import os
import itertools as it
from os.path import exists
from ndsampler.utils import util_gdal


DEBUG_COG_ATOMIC_WRITE = 0
DEBUG_FILE_LOCK_CACHE_WRITE = 0
RUN_COG_CORRUPTION_CHECKS = True
DEBUG_LOAD_COG = int(ub.argflag('--debug-load-cog'))


class CorruptCOG(Exception):
    pass


# @profile
def _cog_cache_write(gpath, cache_gpath, config=None):
    """
    CommandLine:
        xdoctest -m ndsampler.abstract_frames _cog_cache_write

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> import ndsampler
        >>> from ndsampler.abstract_frames import *
        >>> import kwcoco
        >>> workdir = ub.ensure_app_cache_dir('ndsampler')
        >>> dset = kwcoco.CocoDataset.demo()
        >>> imgs = dset.images()
        >>> id_to_name = imgs.lookup('file_name', keepid=True)
        >>> id_to_path = {gid: join(dset.img_root, name)
        >>>               for gid, name in id_to_name.items()}
        >>> self = SimpleFrames(id_to_path, workdir=workdir)
        >>> image_id = ub.peek(id_to_name)
        >>> #gpath = self._lookup_gpath(image_id)

        #### EXIT
        # >>> hashid = self._lookup_hashid(image_id)
        # >>> cog_gname = '{}_{}.cog.tiff'.format(image_id, hashid)
        # >>> cache_gpath = cog_gpath = join(self.cache_dpath, cog_gname)
        # >>> _cog_cache_write(gpath, cache_gpath, {})
    """
    assert config is not None

    # FIXME: if gdal_translate is not installed (because libgdal exists, but
    # gdal-bin doesn't) then this seems to fail without throwing an error when
    # hack_use_cli=1.
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
        is_valid = util_gdal.validate_nonzero_data(file)
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


def _npy_cache_write(gpath, cache_gpath, config=None):
    # Load all the image data and dump it to npy format
    import kwimage
    raw_data = kwimage.imread(gpath)

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


def _lookup_dvc_hash(path):
    """
    proof of concept

    Ignore:
        path = '/home/joncrall/data/dvc-repos/viame_dvc/public/Benthic/US_NE_2017_CFF_HABCAM/annotations_flatfish.kwcoco.json'
        _lookup_dvc_hash(path)
        path = '/home/joncrall/data/dvc-repos/viame_dvc/public/Benthic/US_NE_2017_CFF_HABCAM/Left/201509.20170718.222415535.269652.png'
        _lookup_dvc_hash(path)
    """
    from os.path import split, join, basename
    import yaml
    import json
    from dvc import config
    def _descend(path):
        prev = None
        curr = path
        while curr != prev:
            yield curr
            prev = curr
            curr = split(curr)[0]

    SYMLINK_HACK = True
    if SYMLINK_HACK:
        # This should be faster but less reliable
        from os.path import islink
        import os
        if islink(path):
            linked_path = os.readlink(path)
            rest, p1 = split(linked_path)
            junk, p2 = split(rest)

            if len(p2) == 2 and len(p1) == 30:
                hashid = p2 + p1
                return hashid

    dvc_path = None
    dvc_root = None

    managed_path = None
    for cand in _descend(path):
        dvc_cand = cand + '.dvc'
        dvc_root_cand = join(cand, '.dvc')
        if dvc_path is None and exists(dvc_cand):
            # We found the DVC file
            dvc_path = dvc_cand
            managed_path = cand

        if exists(dvc_root_cand):
            dvc_root = dvc_root_cand
            break

    if dvc_root is None or dvc_path is None:
        raise Exception('Not enough DVC info')

    with open(dvc_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)

    dvc_config = config.Config(dvc_dir=dvc_root)
    dvc_cache_dpath = dvc_config['cache']['dir']

    with open(dvc_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)

    if len(data['outs']) != 1:
        raise NotImplementedError('not sure what to do otherwise')

    for out in data['outs']:
        hashid = out['md5']
        if hashid.endswith('.dir'):
            fpath = join(dvc_cache_dpath, hashid[0:2], hashid[2:])
            with open(fpath, 'r') as file:
                data = json.load(file)
            for item in data:
                content_path = join(managed_path, item['relpath'])
                if content_path == path:
                    found = item['md5']
                    break
        else:
            assert out['path'] == basename(path)
            found = hashid
        return found

    # path.split(os.path.sep)


def _ensure_image_npy(gpath, cache_gpath):
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


def _ensure_image_cog(gpath, cache_gpath, config, hack_use_cli=True):
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
            config = config.copy()
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
