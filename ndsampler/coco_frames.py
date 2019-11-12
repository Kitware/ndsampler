# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from . import abstract_frames
from . import util
from os.path import join
import ubelt as ub


class CocoFrames(abstract_frames.Frames, util.HashIdentifiable):
    """
    wrapper around coco-style dataset to allow for getitem syntax

    CommandLine:
        xdoctest -m ndsampler.coco_frames CocoFrames

    Example:
        >>> from ndsampler.coco_frames import *
        >>> import ndsampler
        >>> import ubelt as ub
        >>> workdir = ub.ensure_app_cache_dir('ndsampler')
        >>> dset = ndsampler.CocoDataset.demo(workdir=workdir)
        >>> dset._ensure_imgsize()
        >>> self = CocoFrames(dset, workdir=workdir)
        >>> assert self.load_image(1).shape == (512, 512, 3)
        >>> assert self.load_image(1)[:-20, :-10].shape == (492, 502, 3)
        >>> assert self.load_region(1, (slice(-20), slice(-10))).shape == (492, 502, 3)

    Example:
        >>> from ndsampler import coco_sampler
        >>> self = coco_sampler.CocoSampler.demo().frames
        >>> assert self.load_image(1).shape == (600, 600, 3)
        >>> assert self.load_image(1)[:-20, :-10].shape == (580, 590, 3)
    """
    def __init__(self, dset, hashid_mode='PATH', workdir=None, verbose=0, backend='auto'):
        super(CocoFrames, self).__init__(hashid_mode=hashid_mode,
                                         workdir=workdir, backend=backend)
        self.dset = dset
        self.verbose = verbose

    def _lookup_gpath(self, image_id):
        img = self.dset.imgs[image_id]
        if self.dset.img_root:
            gpath = join(self.dset.img_root, img['file_name'])
        else:
            gpath = img['file_name']
        return gpath

    @ub.memoize_property
    def image_ids(self):
        return list(self.dset.imgs.keys())

    def _make_hashid(self):
        _hashid = getattr(self.dset, 'hashid', None)
        if _hashid is None:
            if self.verbose > 1:
                print('Constructing frames hashid')
            self.dset._build_hashid()
            _hashid = getattr(self.dset, 'hashid', None)
        return _hashid, None

    def load_region(self, image_id, region=None):
        region = self._rectify_region(region)
        # Check to see if the region is the entire image
        if all(r.start is None and r.stop is None for r in region):
            region = None
        elif all(r.start in [0, None] for r in region):
            img = self.dset.imgs[image_id]
            flag = region[0].stop in [None, img['height']]
            flag &= region[1].stop in [None, img['width']]
            if flag:
                region = None  # setting region to None disables memmap/geotiff caching

        return super(CocoFrames, self).load_region(image_id, region)

    def _lookup_hashid(self, image_id):
        """
        Get the hashid of a particular image
        """
        if image_id not in self.id_to_hashid:
            USE_RELATIVE_PATH = not self._backend.get('_hack_old_names', False)
            if USE_RELATIVE_PATH and self.hashid_mode == 'PATH':
                # Hash the relative path to the image data
                img = self.dset.imgs[image_id]
                rel_gpath = img['file_name']
                hashid = ub.hash_data(rel_gpath, hasher='sha1', base='hex')
                self.id_to_hashid[image_id] = hashid
            else:
                return super(CocoFrames, self)._lookup_hashid(image_id)
        return self.id_to_hashid[image_id]
