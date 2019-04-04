# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import numpy as np
import kwimage
import six
from ndsampler import coco_dataset
from ndsampler import coco_regions
from ndsampler import coco_frames
from ndsampler import abstract_sampler
from ndsampler import util


class CocoSampler(abstract_sampler.AbstractSampler, util.HashIdentifiable,
                  ub.NiceRepr):
    """
    Samples patches of positives and negative detection windows from a COCO
    dataset. Can be used for training FCN or RPN based classifiers / detectors.

    Does data loading, padding, etc...

    Args:
        dset (ndsampler.CocoDataset): a coco-formatted dataset

    Example:
        >>> from ndsampler.coco_sampler import *
        >>> self = CocoSampler.demo('photos')
        ...
        >>> print(sorted(self.class_ids))
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> print(self.n_positives)
        4

    Example:
        >>> import ndsampler
        >>> self = ndsampler.CocoSampler.demo('photos')
        >>> p_sample = self.load_positive()
        >>> n_sample = self.load_negative()
        >>> self = ndsampler.CocoSampler.demo('shapes')
        >>> p_sample2 = self.load_positive()
        >>> n_sample2 = self.load_negative()
        >>> for sample in [p_sample, n_sample, p_sample2, n_sample2]:
        >>>     assert 'annots' in sample
        >>>     assert 'im' in sample
        >>>     assert 'rel_boxes' in sample['annots']
        >>>     assert 'rel_ssegs' in sample['annots']
        >>>     assert 'rel_kpts' in sample['annots']
        >>>     assert 'cids' in sample['annots']
        >>>     assert 'aids' in sample['annots']
    """

    @classmethod
    def demo(cls, key='shapes', workdir=None, **kw):
        from ndsampler import toydata
        if key == 'shapes':
            dset = coco_dataset.CocoDataset(toydata.demodata_toy_dset(**kw))
        elif key == 'photos':
            dset = coco_dataset.CocoDataset.demo(**kw)
            toremove = [ann for ann in dset.anns.values() if 'bbox' not in ann]
            dset.remove_annotations(toremove)
            dset.add_category('background', id=0)
        else:
            raise KeyError(key)
        if workdir is None:
            workdir = ub.ensure_app_cache_dir('ndsampler')
        self = CocoSampler(dset, workdir=workdir)
        return self

    def __init__(self, dset, workdir=None, autoinit=True, verbose=0):
        super(CocoSampler, self).__init__()
        self.workdir = workdir
        self.dset = dset
        self.regions = None
        self.frames = None
        self.verbose = verbose
        self.BACKGROUND_CLASS_ID = None

        if autoinit:
            self._init()

    def _init(self):
        self.dset._ensure_imgsize()
        if self.dset.anns is None:
            self.dset._build_index()
        self.regions = coco_regions.CocoRegions(self.dset,
                                                workdir=self.workdir,
                                                verbose=self.verbose)
        self.frames = coco_frames.CocoFrames(self.dset, workdir=self.workdir)

        self.catgraph = self.regions.catgraph

        # === Hacked in attributes ===
        self.kp_classes = self.dset.keypoint_categories()
        self.BACKGROUND_CLASS_ID = self.regions.BACKGROUND_CLASS_ID  # currently hacked in

    @property
    def classes(self):
        return self.catgraph

    def _depends(self):
        hashid_parts = ub.odict()
        hashid_parts['regions_hashid'] = self.regions.hashid
        hashid_parts['frames_hashid'] = self.frames.hashid
        return hashid_parts

    def lookup_class_name(self, class_id):
        return self.regions.lookup_class_name(class_id)

    def lookup_class_id(self, class_name):
        return self.regions.lookup_class_id(class_name)

    @property
    def n_positives(self):
        return self.regions.n_positives

    @property
    def n_annots(self):
        return self.regions.n_annots

    @property
    def n_samples(self):
        return self.regions.n_samples

    def __len__(self):
        return self.n_samples

    @property
    def n_images(self):
        return self.regions.n_images

    @property
    def n_categories(self):
        return self.regions.n_categories

    @property
    def class_ids(self):
        return self.regions.class_ids

    @property
    def image_ids(self):
        return self.regions.image_ids

    def preselect(self, **kwargs):
        return self.regions.preselect(**kwargs)

    def load_image_with_annots(self, image_id):
        """
        Args:
            image_id (int): the coco image id

        Returns:
            Tuple[Dict, List[Dict]]:
                img: the coco image dict augmented with imdata
                anns: the coco annotations in this image

        Example:
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> rng = None
            >>> img, anns = self.load_image_with_annots(1)
            >>> dets = kwimage.Detections.from_coco_annots(anns, dset=self.dset)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(img['imdata'][:])
            >>> dets.draw()
            >>> kwplot.show_if_requested()
        """
        full_image = self.frames.load_image(image_id)
        gid = image_id
        coco_dset = self.dset
        img = coco_dset.imgs[gid].copy()
        aids = coco_dset.index.gid_to_aids[gid]
        anns = [coco_dset.anns[aid] for aid in aids]
        img['imdata'] = full_image
        return img, anns

    def load_image(self, image_id):
        full_image = self.frames.load_image(image_id)
        return full_image

    def load_item(self, index, pad=None, window_dims=None):
        """
        Loads from positives and then negatives.
        """
        if index < self.n_positives:
            sample = self.load_positive(index, pad=pad, window_dims=window_dims)
        else:
            index = index - self.n_positives
            sample = self.load_negative(index, pad=pad, window_dims=window_dims)
        return sample

    def load_positive(self, index=None, pad=None, window_dims=None, rng=None):
        """
        Args:
            index (int): index of positive target
            pad (tuple): (height, width) extra context to add to each size.
                This helps prevent augmentation from producing boundary effects
            window_dims (tuple): (height, width).  overwrite the height/width
                in tr and extract a window around the center of the object
            window_dims (tuple): (height, width) area around the center
                of the target object to sample.

        Returns:
            Dict: sample: dict containing keys
                im (ndarray): image data
                tr (dict): contains the same input items as tr but additionally
                    specifies rel_cx and rel_cy, which gives the center
                    of the target w.r.t the returned **padded** sample.
                annots (dict): Dict of aids, cids, and rel/abs boxes

        Example:
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> rng = None
            >>> sample = self.load_positive(pad=(10, 10), window_dims=(3, 3))
            >>> assert sample['im'].shape[0] == 23
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(sample)
            >>> kwplot.show_if_requested()
        """
        tr = self.regions.get_positive(index, rng=rng)
        sample = self.load_sample(tr, pad=pad, window_dims=window_dims)
        return sample

    def load_negative(self, index=None, pad=None, window_dims=None, rng=None):
        """
        Args:
            index (int): if specified loads a specific negative from the
                presampled pool, otherwise the next negative in the pool is
                returned.
            pad (tuple): (height, width) extra context to add to each size.
                This helps prevent augmentation from producing boundary effects
            window_dims (tuple): (height, width) area around the center
                of the target negative region to sample.

        Returns:
            Dict: sample: dict containing keys
                im (ndarray): image data
                tr (dict): contains the same input items as tr but additionally
                    specifies rel_cx and rel_cy, which gives the center
                    of the target w.r.t the returned **padded** sample.
                annots (dict): Dict of aids, cids, and rel/abs boxes

        Example:
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> rng = None
            >>> sample = self.load_negative(rng=rng, pad=(0, 0))
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> box = kwimage.Boxes(tr.reindex(['rel_cx', 'rel_cy', 'width', 'height']).values, 'cxywh')
            >>> kwplot.imshow(sample)
            >>> kwplot.draw_boxes(box)
            >>> kwplot.show_if_requested()

        Example:
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> rng = None
            >>> sample = self.load_negative(rng=rng, pad=(0, 0), window_dims=(64, 64))
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> box = kwimage.Boxes(tr.reindex(['rel_cx', 'rel_cy', 'width', 'height']).values, 'cxywh')
            >>> kwplot.imshow(sample, fnum=1, doclf=True)
            >>> kwplot.draw_boxes(box)
            >>> kwplot.show_if_requested()
        """
        tr = self.regions.get_negative(index, rng=rng)
        sample = self.load_sample(tr, pad=pad, window_dims=window_dims)
        return sample

    def load_sample(self, tr, pad=None, window_dims=None, visible_thresh=0.1,
                    padkw={'mode': 'constant'}):
        """
        Loads the volume data associated with the bbox and frame of a target

        Args:
            tr (dict): image and bbox info for a positive / negative target.
                must contain the keys ['cx', 'cy', 'gid'], if `window_dims` is
                None it must also contain the keys ['width' and 'height'].
            pad (tuple): (height, width) extra context to add to window dims.
                This helps prevent augmentation from producing boundary effects
            window_dims (tuple): (height, width) overrides the height/width
                in tr to determine the extracted window size
            visible_thresh (float): does not return annotations with visibility
                less than this threshold.
            padkw (dict): kwargs for `numpy.pad`

        Returns:
            Dict: sample: dict containing keys
                im (ndarray): image data
                tr (dict): contains the same input items as tr but additionally
                    specifies rel_cx and rel_cy, which gives the center
                    of the target w.r.t the returned **padded** sample.
                annots (dict): containing items:
                    aids (list): annotation ids
                    cids (list): category ids
                    rel_cxywh (ndarray): boxes relative to the sample
                    rel_ssegs (ndarray): segmentations relative to the sample
                    rel_kpts (ndarray): keypoints relative to the sample

        CommandLine:
            xdoctest -m ndsampler.coco_sampler CocoSampler.load_sample:2 --show

            xdoctest -m ndsampler.coco_sampler CocoSampler.load_sample:1 --show
            xdoctest -m ndsampler.coco_sampler CocoSampler.load_sample:3 --show

        Example:
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> # The target (tr) lets you specify an arbitrary window
            >>> tr = {'gid': 1, 'cx': 5, 'cy': 2, 'width': 6, 'height': 6}
            >>> sample = self.load_sample(tr)
            >>> print('sample.shape = {!r}'.format(sample['im'].shape))
            sample.shape = (6, 6, 3)

        Example:
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> tr = self.regions.get_positive(0)
            >>> pad = (25, 25)
            >>> sample = self.load_sample(tr, pad, window_dims='square')
            >>> print('im.shape = {!r}'.format(sample['im'].shape))
            im.shape = (135, 135, 3)
            >>> pad = (0, 0)
            >>> sample = self.load_sample(tr, pad)
            >>> print('im.shape = {!r}'.format(sample['im'].shape))
            im.shape = (52, 85, 3)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(sample['im'])
            >>> kwplot.show_if_requested()

        Example:
            >>> # sample an out of bounds target
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> tr = self.regions.get_positive(0)
            >>> window_dims = None
            >>> window_dims = (364, 364)
            >>> sample = self.load_sample(tr, window_dims=window_dims)
            >>> annots = sample['annots']
            >>> assert len(annots['aids']) > 0
            >>> assert len(annots['rel_cxywh']) == len(annots['aids'])
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> abs_frame = self.frames.load_image(sample['tr']['gid'])
            >>> tf_rel_to_abs = sample['params']['tf_rel_to_abs']
            >>> abs_boxes = annots['rel_boxes'].warp(tf_rel_to_abs)
            >>> abs_ssegs = annots['rel_ssegs'].warp(tf_rel_to_abs)
            >>> abs_kpts = annots['rel_kpts'].warp(tf_rel_to_abs)
            >>> # Draw box in original image context
            >>> kwplot.imshow(abs_frame, pnum=(1, 2, 1), fnum=1)
            >>> abs_boxes.translate([-.5, -.5]).draw()
            >>> abs_kpts.draw(color='green', radius=10)
            >>> abs_ssegs.draw(color='red', alpha=.5)
            >>> # Draw box in relative sample context
            >>> kwplot.imshow(sample['im'], pnum=(1, 2, 2), fnum=1)
            >>> annots['rel_boxes'].translate([-.5, -.5]).draw()
            >>> annots['rel_ssegs'].draw(color='red', alpha=.6)
            >>> annots['rel_kpts'].draw(color='green', alpha=.4, radius=10)
            >>> kwplot.show_if_requested()

        Example:
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo('photos')
            >>> tr = self.regions.get_positive(1)
            >>> window_dims = (300, 150)
            >>> pad = None
            >>> sample = self.load_sample(tr, pad, window_dims=window_dims)
            >>> assert sample['im'].shape[0:2] == window_dims
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(sample['im'], colorspace='rgb')
            >>> kwplot.show_if_requested()
        """
        if pad is None:
            pad = (0, 0)
        gid = tr['gid']

        center = (tr['cy'], tr['cx'])

        # Determine the image extent
        img = self.dset.imgs[gid]
        data_dims = (img['height'], img['width'])

        # Determine the requested window size
        if window_dims is None:
            window_dims = (tr['height'], tr['width'])
            window_dims = np.ceil(np.array(window_dims)).astype(np.int)
            window_dims = tuple(window_dims.tolist())
        elif isinstance(window_dims, six.string_types) and window_dims == 'square':
            window_dims = (tr['height'], tr['width'])
            window_dims = np.ceil(np.array(window_dims)).astype(np.int)
            window_dims = tuple(window_dims.tolist())
            maxdim = max(window_dims)
            window_dims = (maxdim, maxdim)

        data_slice, extra_padding = _get_slice(data_dims, center, window_dims, pad=pad)

        # Load the image data
        im = self.frames.load_region(gid, data_slice)
        if extra_padding:
            if im.ndim != len(extra_padding):
                extra_padding = extra_padding + [(0, 0)]  # Handle channels
            im = np.pad(im, extra_padding, **padkw)

        ndim = 2  # number of space-time dimensions (ignore channel)
        st_dims = [(sl.start, sl.stop) for sl in data_slice[0:ndim]]

        # Translations for real sub-pixel center positions
        if extra_padding:
            pad_dims = extra_padding[0:ndim]
            st_dims = [(s - pad[0], t + pad[1])
                       for (s, t), pad in zip(st_dims, pad_dims)]

        (y_start, y_stop), (x_start, x_stop) = st_dims[-2:]

        # tlbr box in original image space around this sampled patch
        sample_tlbr = kwimage.Boxes([x_start, y_start, x_stop, y_stop], 'tlbr')
        # Find which bounding boxes are visible in this region
        overlap_aids = self.regions.overlapping_aids(
            gid, sample_tlbr, visible_thresh=visible_thresh)

        # Get info about all annotations inside this window
        overlap_annots = self.dset.annots(overlap_aids)
        overlap_cids = overlap_annots.cids

        abs_boxes = overlap_annots.boxes

        # overlap_keypoints = [
        #     for aid in overlap_aids
        # ]
        # Transform spatial information to be relative to the sample

        offset = np.array([-x_start, -y_start])
        rel_boxes = abs_boxes.translate(offset)

        # Handle segmentations and keypoints if they exist
        sseg_list = []
        kpts_list = []

        coco_dset = self.dset
        kp_classes = self.kp_classes
        for aid in overlap_aids:
            ann = coco_dset.anns[aid]
            coco_sseg = ann.get('segmentation', None)
            coco_kpts = ann.get('keypoints', None)
            if coco_kpts is not None:
                kpnames = coco_dset._lookup_kpnames(ann['category_id'])
                coco_xyf = np.array(coco_kpts).reshape(-1, 3)
                # TODO: use Points._from_coco
                flags = (coco_xyf.T[2] > 0)
                xy_pts = coco_xyf[flags, 0:2]
                kpnames = list(ub.compress(kpnames, flags))
                kp_class_idxs = np.array([kp_classes.index(n) for n in kpnames])
                abs_points = kwimage.Points(xy=xy_pts,
                                            class_idxs=kp_class_idxs,
                                            classes=kp_classes)
                rel_points = abs_points.translate(offset)
            else:
                rel_points = None

            if coco_sseg is not None:
                # x = _coerce_coco_segmentation(coco_sseg, data_dims)
                # abs_sseg = kwimage.Mask.coerce(coco_sseg, dims=data_dims)
                abs_sseg = kwimage.MultiPolygon.coerce(coco_sseg, dims=data_dims)
                # abs_sseg = abs_sseg.to_multi_polygon()
                rel_sseg = abs_sseg.translate(offset)
            else:
                rel_sseg = None

            kpts_list.append(rel_points)
            sseg_list.append(rel_sseg)

        import skimage
        tf_rel_to_abs = skimage.transform.AffineTransform(
            translation=-offset
        ).params

        rel_ssegs = kwimage.PolygonList(sseg_list)
        rel_kpts = kwimage.PolygonList(kpts_list)
        rel_kpts.meta['classes'] = self.kp_classes

        annots = {
            'aids': np.array(overlap_aids),
            'cids': np.array(overlap_cids),

            'rel_cxywh': rel_boxes.to_cxywh().data,
            'abs_cxywh': abs_boxes.to_cxywh().data,

            'rel_boxes': rel_boxes,
            'rel_ssegs': rel_ssegs,
            'rel_kpts': rel_kpts,
        }

        # Note the center coordinates in the padded sample reference frame
        tr_ = tr.copy()
        sample = {
            'im': im,
            'tr': tr_,
            'params': {
                'offset': offset,
                'tf_rel_to_abs': tf_rel_to_abs,
                'pad': pad,
            },
            'annots': annots

        }
        return sample


def _get_slice(data_dims, center, window_dims, pad=None):
    """
    Finds the slice to sample data around a center in a particular image

    Args:
        data_dims (Tuple[int]): image size (height, width)
        center (Tuple[float]): center location (cy, cx)
        window_dims (Tuple[int]): window size (height, width)
        pad (Tuple[int]): extra context to add to each size (height, width)
            This helps prevent augmentation from producing boundary effects

    Returns:
        Tuple:
            data_slice - a fancy slice corresponding to the image. This
                slice may not correspond to the full window size if the
                requested bounding box goes out of bounds.
            extra_padding - extra padding needed after slicing to achieve
                the requested window size.

    Example:
        >>> center = (2, 5)
        >>> window_dims = (6, 6)
        >>> data_dims = (600, 600)
        >>> data_slice, extra_padding = _get_slice(data_dims, center, window_dims)
        >>> assert extra_padding == [(1, 0), (0, 0)]
        >>> assert data_slice == (slice(0, 5), slice(2, 8))

     Example:
        >>> center = (2, 5)
        >>> window_dims = (64, 64)
        >>> data_dims = (600, 600)
        >>> data_slice, extra_padding = _get_slice(data_dims, center, window_dims)
        >>> assert extra_padding == [(30, 0), (27, 0)]
        >>> assert data_slice == (slice(0, 34, None), slice(0, 37, None))
    """
    if pad is None:
        pad = (0, 0)

    # Compute lower and upper coordinates of the window bounding box
    low_dims = [int(np.floor(c - d_win / 2.0))
                for c, d_win in zip(center, window_dims)]
    high_dims = [int(np.floor(c + d_win / 2.0))
                 for c, d_win in zip(center, window_dims)]

    if __debug__:
        for d_win, d_low, d_high in zip(window_dims, low_dims, high_dims):
            assert d_high - d_low == d_win

    # Find the lower and upper coordinates that corresponds to the
    # requested window in the real image. If the window goes out of bounds,
    # then use extra padding to achieve the requested window shape.
    data_slice, extra_padding = _rectify_slice(data_dims, low_dims,
                                               high_dims, pad)

    data_slice = tuple([slice(low, high) for low, high in data_slice])
    if sum(map(sum, extra_padding)) == 0:
        extra_padding = None
    return data_slice, extra_padding


def _rectify_slice(data_dims, low_dims, high_dims, pad):
    """
    Given image dimensions, bounding box dimensions, and a padding get the
    corresponding slice from the image and any extra padding needed to achieve
    the requested window size.

    Args:
        data_dims (tuple): n-dimension data sizes (e.g. 2d height, width)
        low_dims (tuple): bounding box low values (e.g. 2d ymin, xmin)
        high_dims (tuple): bounding box high values (e.g. 2d ymax, xmax)
        pad (tuple): padding applied to both sides of each dim

    Returns:
        Tuple:
            data_slice - low and high values of a fancy slice corresponding to
                the image with shape `data_dims`. This slice may not correspond
                to the full window size if the requested bounding box goes out
                of bounds.
            extra_padding - extra padding needed after slicing to achieve
                the requested window size.

    Example:
        >>> # Case where 2D-bbox is inside the data dims on left edge
        >>> # Comprehensive 1D-cases are in the unit-test file
        >>> from ndsampler.coco_sampler import *
        >>> data_dims  = [300, 300]
        >>> low_dims   = [0, 0]
        >>> high_dims  = [10, 10]
        >>> pad        = [10, 5]
        >>> a, b = _rectify_slice(data_dims, low_dims, high_dims, pad)
        >>> print('data_slice = {!r}'.format(a))
        >>> print('extra_padding = {!r}'.format(b))
        data_slice = [(0, 20), (0, 15)]
        extra_padding = [(10, 0), (5, 0)]
    """
    # Determine the real part of the image that can be sliced out
    data_slice = []
    extra_padding = []
    for D_img, d_low, d_high, d_pad in zip(data_dims, low_dims, high_dims, pad):
        if d_low > d_high:
            raise ValueError('d_low > d_high: {} > {}'.format(d_low, d_high))
        # Determine where the bounds would be if the image size was inf
        raw_low = d_low - d_pad
        raw_high = d_high + d_pad
        # Clip the slice positions to the real part of the image
        sl_low = min(D_img, max(0, raw_low))
        sl_high = min(D_img, max(0, raw_high))
        data_slice.append((sl_low, sl_high))

        # Add extra padding when the window extends past the real part
        low_diff = sl_low - raw_low
        high_diff = raw_high - sl_high

        # Hand the case where both raw coordinates are out of bounds
        extra_low = max(0, low_diff + min(0, high_diff))
        extra_high = max(0, high_diff + min(0, low_diff))
        extra = (extra_low, extra_high)
        extra_padding.append(extra)
    return data_slice, extra_padding


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m ndsampler.coco_sampler
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
