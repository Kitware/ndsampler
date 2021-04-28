# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from ndsampler import abstract_frames
from ndsampler.utils import util_misc
import numpy as np
import warnings
import ubelt as ub


try:
    from xdev import profile
except Exception:
    profile = ub.identity


class CocoFrames(abstract_frames.Frames, util_misc.HashIdentifiable):
    """
    wrapper around coco-style dataset to allow for getitem syntax

    CommandLine:
        xdoctest -m ndsampler.coco_frames CocoFrames

    Example:
        >>> from ndsampler.coco_frames import *
        >>> import ndsampler
        >>> import kwcoco
        >>> import ubelt as ub
        >>> workdir = ub.ensure_app_cache_dir('ndsampler')
        >>> dset = kwcoco.CocoDataset.demo(workdir=workdir)
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
    def __init__(self, dset, hashid_mode='PATH', workdir=None, verbose=0,
                 backend='auto'):
        super(CocoFrames, self).__init__(hashid_mode=hashid_mode,
                                         workdir=workdir, backend=backend)
        self.dset = dset
        self.verbose = verbose
        self._image_ids = None

    @property
    def image_ids(self):
        if self._image_ids is None:
            import numpy as np
            # Use ndarrays to prevent copy-on-write as best as possible
            self._image_ids = np.array(list(self.dset.imgs.keys()))
        return self._image_ids

    def _make_hashid(self):
        _hashid = getattr(self.dset, 'hashid', None)
        if _hashid is None:
            if self.verbose > 1:
                print('Constructing frames hashid')
            self.dset._build_hashid()
            _hashid = getattr(self.dset, 'hashid', None)
        return _hashid, None

    def load_region(self, image_id, region=None, channels=ub.NoParam):
        img = self.dset.imgs[image_id]
        width = img.get('width', None)
        height = img.get('height', None)
        return super(CocoFrames, self).load_region(image_id, region,
                                                   width=width, height=height,
                                                   channels=channels)

    @profile
    def _build_pathinfo(self, image_id):
        """
        Returns:
            See Parent Method Docs

        Example:
            >>> import ndsampler
            >>> sampler1 = ndsampler.CocoSampler.demo('vidshapes5-aux')
            >>> sampler2 = ndsampler.CocoSampler.demo('vidshapes5-multispectral')
            >>> self = sampler1.frames
            >>> pathinfo = self._build_pathinfo(1)
            >>> print('pathinfo = {}'.format(ub.repr2(pathinfo, nl=3)))

            >>> self = sampler2.frames
            >>> pathinfo = self._build_pathinfo(1)
            >>> print('pathinfo = {}'.format(ub.repr2(pathinfo, nl=3)))
        """
        img = self.dset.imgs[image_id]
        default_fname = img.get('file_name', None)
        default_channels = img.get('channels', None)

        channels = {}
        pathinfo = {
            'id': image_id,
            'channels': channels,
            'default': default_channels,
            'width': img.get('width', None),
            'height': img.get('height', None),
        }
        root = self.dset.bundle_dpath
        if default_fname is not None:
            chan = {
                'file_name': default_fname,
                'channels': default_channels,
            }
            channels[default_channels] = chan

        for aux in img.get('auxiliary', []):
            fname = aux['file_name']
            warp_aux_to_img = aux.get('warp_aux_to_img', None)
            if warp_aux_to_img is None:
                # backwards compat for deprecated base_to_aux
                warp_img_to_aux = aux.get('base_to_aux', None)
                if warp_img_to_aux is not None:
                    warnings.warn(
                        'base_to_aux is deprecated, invert and use '
                        'warp_aux_to_img', DeprecationWarning)
                else:
                    warp_img_to_aux = aux.get('warp_img_to_aux', None)
                if warp_img_to_aux is not None:
                    mat_aux_to_img = np.linalg.inv(warp_img_to_aux['matrix'])
                    warp_aux_to_img = {
                        'type': 'affine',
                        'matrix': mat_aux_to_img.tolist(),
                    }
            chan = {
                'file_name': fname,
                'channels': aux['channels'],
                'warp_aux_to_img': warp_aux_to_img,
            }
            channels[aux['channels']] = chan

        for chan in pathinfo['channels'].values():
            self._populate_chan_info(chan, root=root)
        return pathinfo
