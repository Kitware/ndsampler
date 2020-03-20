# -*- coding: utf-8 -*-
"""
Example:
    >>> # Imagine you have some images
    >>> import kwimage
    >>> image_paths = [
    >>>     kwimage.grab_test_image_fpath('astro'),
    >>>     kwimage.grab_test_image_fpath('carl'),
    >>>     kwimage.grab_test_image_fpath('airport'),
    >>> ]  # xdoc: +IGNORE_WANT
    ['~/.cache/kwimage/demodata/KXhKM72.png',
     '~/.cache/kwimage/demodata/flTHWFD.png',
     '~/.cache/kwimage/demodata/Airport.jpg']
    >>> # And you want to randomly load subregions of them in O(1) time
    >>> import ndsampler
    >>> # First make a COCO dataset that refers to your images (and possibly annotations)
    >>> dataset = {
    >>>     'images': [{'id': i, 'file_name': fpath} for i, fpath in enumerate(image_paths)],
    >>>     'annotations': [],
    >>>     'categories': [],
    >>> }
    >>> coco_dset = ndsampler.CocoDataset(dataset)
    >>> print(coco_dset)
    <CocoDataset(tag=None, n_anns=0, n_imgs=3, n_cats=0)>
    >>> # Now pass the dataset to a sampler and tell it where it can store temporary files
    >>> workdir = ub.ensure_app_cache_dir('ndsampler/demo')
    >>> sampler = ndsampler.CocoSampler(coco_dset, workdir=workdir)
    >>> # Now you can load arbirary samples by specifing a target dictionary
    >>> # with an image_id (gid) center location (cx, cy) and width, height.
    >>> target = {'gid': 0, 'cx': 200, 'cy': 200, 'width': 100, 'height': 100}
    >>> sample = sampler.load_sample(target)
    >>> # The sample contains the image data, any visible annotations, a reference
    >>> # to the original target, and params of the transform used to sample this
    >>> # patch
    >>> print(sorted(sample.keys()))
    ['annots', 'im', 'params', 'tr']
    >>> im = sample['im']
    >>> print(im.shape)
    (100, 100, 3)
    >>> # The load sample function is at the core of what ndsampler does
    >>> # There are other helper functions like load_positive / load_negative
    >>> # which deal with annotations. See those for more details.
    >>> # For random negative sampling see coco_regions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import numpy as np
import kwimage
import six
from ndsampler import coco_dataset
from ndsampler import coco_regions
from ndsampler import coco_frames
from ndsampler import abstract_sampler
from ndsampler.utils import util_misc


class CocoSampler(abstract_sampler.AbstractSampler, util_misc.HashIdentifiable,
                  ub.NiceRepr):
    """
    Samples patches of positives and negative detection windows from a COCO
    dataset. Can be used for training FCN or RPN based classifiers / detectors.

    Does data loading, padding, etc...

    Args:
        dset (ndsampler.CocoDataset): a coco-formatted dataset

        backend (str | Dict): either 'cog' or 'npy', or a dict with
            `{'type': str, 'config': Dict}`. See AbstractFrames for more
            details.

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
    def demo(cls, key='shapes', workdir=None, backend='auto', **kw):
        """
        Create a toy coco sampler for testing and demo puposes

        SeeAlso:
            * ndsampler.CocoDataset.demo
        """
        dset = coco_dataset.CocoDataset.demo(key=key, **kw)
        if key == 'photos':
            toremove = [ann for ann in dset.anns.values() if 'bbox' not in ann]
            dset.remove_annotations(toremove)
            dset.add_category('background', id=0)
        if workdir is None:
            workdir = ub.ensure_app_cache_dir('ndsampler')
        self = CocoSampler(dset, workdir=workdir, backend=backend)
        return self

    def __init__(self, dset, workdir=None, autoinit=True, backend='auto',
                 verbose=0):
        super(CocoSampler, self).__init__()
        self.workdir = workdir
        self.dset = dset
        self.regions = None
        self.frames = None

        # save at least until we init the frames / regions
        self._backend = backend

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
        self.frames = coco_frames.CocoFrames(
            self.dset,
            workdir=self.workdir,
            backend=self._backend,
        )

        # === Hacked in attributes ===
        self.kp_classes = self.dset.keypoint_categories()
        self.BACKGROUND_CLASS_ID = self.regions.BACKGROUND_CLASS_ID  # currently hacked in

    @property
    def classes(self):
        if self.regions is None:
            return None
        return self.regions.classes

    @property
    def catgraph(self):
        """
        DEPRICATED, use self.classes instead
        """
        if self.regions is None:
            return None
        return self.regions.classes

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

    def load_image_with_annots(self, image_id, cache=True):
        """
        Args:
            image_id (int): the coco image id

            cache (bool, default=True): if True returns the fast
                subregion-indexable file reference. Otherwise, eagerly loads
                the entire image.

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
        full_image = self.load_image(image_id, cache=cache)
        coco_dset = self.dset
        img = coco_dset.imgs[image_id].copy()
        anns = self.load_annotations(image_id)
        img['imdata'] = full_image
        return img, anns

    def load_annotations(self, image_id):
        """
        Loads the annotations within an image

        Args:
            image_id (int): the coco image id

        Returns:
            List[Dict]: list of coco annotation dictionaries
        """
        coco_dset = self.dset
        aids = coco_dset.index.gid_to_aids[image_id]
        anns = [coco_dset.anns[aid] for aid in aids]
        return anns

    def load_image(self, image_id, cache=True):
        """
        Loads the annotations within an image

        Args:
            image_id (int): the coco image id

            cache (bool, default=True): if True returns the fast
                subregion-indexable file reference. Otherwise, eagerly loads
                the entire image.

        Returns:
            ArrayLike: either ndarray data or a indexable reference
        """
        full_image = self.frames.load_image(image_id, cache=cache)
        return full_image

    def load_item(self, index, pad=None, window_dims=None, with_annots=True):
        """
        Loads item from either positive or negative regions pool.

        Lower indexes will return positive regions and higher indexes will
        return negative regions.

        The main paradigm of the sampler is that sampler.regions maintains a
        pool of target regions, you can influence what that pool is at any
        point by calling sampler.regions.preselect (usually either at the start
        of learning, or maybe after every epoch, etc..), and you use load_item
        to load the index-th item from that preselected pool. Depending on how
        you preselected the pool, the returned item might correspond to a
        positive or negative region.

        Args:
            index (int): index of target region
            pad (tuple): (height, width) extra context to add to each size.
                This helps prevent augmentation from producing boundary effects
            window_dims (tuple): (height, width) area around the center
                of the target region to sample.
            with_annots (bool | str, default=True):
                if True, also extracts information about any annotation that
                overlaps the region of interest (subject to visibility_thresh).
                Can also be a List[str] that specifies which specific subinfo
                should be extracted. Valid strings in this list are: boxes,
                keypoints, and segmenation.

        Returns:
            Dict: sample: dict containing keys
                im (ndarray): image data
                tr (dict): contains the same input items as tr but additionally
                    specifies rel_cx and rel_cy, which gives the center
                    of the target w.r.t the returned **padded** sample.
                annots (dict): Dict of aids, cids, and rel/abs boxes
        """
        if index < self.n_positives:
            sample = self.load_positive(index, pad=pad,
                                        window_dims=window_dims,
                                        with_annots=with_annots)
        else:
            index = index - self.n_positives
            sample = self.load_negative(index, pad=pad,
                                        window_dims=window_dims,
                                        with_annots=with_annots)
        return sample

    def load_positive(self, index=None, pad=None, window_dims=None,
                      with_annots=True, rng=None):
        """
        Load an item from the the positive pool of regions.

        Args:
            index (int): index of positive target
            pad (tuple): (height, width) extra context to add to each size.
                This helps prevent augmentation from producing boundary effects
            window_dims (tuple): (height, width) area around the center
                of the target object to sample.
            with_annots (bool | str, default=True):
                if True, also extracts information about any annotation that
                overlaps the region of interest (subject to visibility_thresh).
                Can also be a List[str] that specifies which specific subinfo
                should be extracted. Valid strings in this list are: boxes,
                keypoints, and segmenation.

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
            >>> kwplot.imshow(sample['im'])
            >>> kwplot.show_if_requested()
        """
        tr = self.regions.get_positive(index, rng=rng)
        sample = self.load_sample(tr, pad=pad, window_dims=window_dims,
                                  with_annots=with_annots)
        return sample

    def load_negative(self, index=None, pad=None, window_dims=None,
                      with_annots=True, rng=None):
        """
        Load an item from the the negative pool of regions.

        Args:
            index (int): if specified loads a specific negative from the
                presampled pool, otherwise the next negative in the pool is
                returned.
            pad (tuple): (height, width) extra context to add to each size.
                This helps prevent augmentation from producing boundary effects
            window_dims (tuple): (height, width) area around the center
                of the target negative region to sample.
            with_annots (bool | str, default=True):
                if True, also extracts information about any annotation that
                overlaps the region of interest (subject to visibility_thresh).
                Can also be a List[str] that specifies which specific subinfo
                should be extracted. Valid strings in this list are: boxes,
                keypoints, and segmenation.

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
        sample = self.load_sample(tr, pad=pad, window_dims=window_dims,
                                  with_annots=with_annots)
        return sample

    def load_sample(self, tr, pad=None, window_dims=None, visible_thresh=0.0,
                    with_annots=True, padkw={'mode': 'constant'}):
        """
        Loads the volume data associated with the bbox and frame of a target

        Args:
            tr (dict): image and bbox info for a positive / negative target.
                must contain the keys ['gid', 'cx', 'cy'], if `window_dims` is
                None it must also contain the keys ['width' and 'height'].

                NEW: tr can now contain the key `slices`, which maps a tuple of
                slices, one slice for each of the n dimensions.  If specified
                this will overwrite the 'cx', 'cy' keys. The 'gid' key is still
                required, and `pad` does still have an effect.

            pad (tuple): (height, width) extra context to add to window dims.
                This helps prevent augmentation from producing boundary effects

            window_dims (tuple | str): (height, width) overrides the height/width
                in tr to determine the extracted window size. Can also be
                'extent' or 'square', which determines the final size using
                target information.

            visible_thresh (float): does not return annotations with visibility
                less than this threshold.

            padkw (dict): kwargs for `numpy.pad`

            with_annots (bool | str, default=True):
                if True, also extracts information about any annotation that
                overlaps the region of interest (subject to visibility_thresh).
                Can also be a List[str] that specifies which specific subinfo
                should be extracted. Valid strings in this list are: boxes,
                keypoints, and segmenation.

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
        sample = self._load_slice(tr, window_dims, pad, padkw)

        if with_annots or ub.iterable(with_annots):
            self._populate_overlap(sample, visible_thresh, with_annots)
        return sample

    def _rectify_tr(self, tr, data_dims, window_dims=None, pad=None):
        """
        Use the target to compute the actual slice within the image as well as
        what additional padding is needed, if any.
        """
        ndim = 2  # number of space-time dimensions (ignore channel)
        if pad is None:
            pad = 0
        pad = tuple(_ensure_iterablen(pad, ndim))

        if 'slices' in tr:
            # Slice was explicitly specified
            import warnings
            if bool(set(tr) & {'cx', 'cy', 'height', 'width'}) or window_dims:
                warnings.warn('data_slice was specified, but ignored keys are present')
            requested_slice = tr['slices']
            data_slice, extra_padding = _rectify_slice2(requested_slice, data_dims, pad)
        else:
            # A center / width / height was specified
            center = (tr['cy'], tr['cx'])
            # Determine the requested window size
            if window_dims is None:
                window_dims = 'extent'

            if isinstance(window_dims, six.string_types):
                if window_dims == 'extent':
                    window_dims = (tr['height'], tr['width'])
                    window_dims = np.ceil(np.array(window_dims)).astype(np.int)
                    window_dims = tuple(window_dims.tolist())
                elif window_dims == 'square':
                    window_dims = (tr['height'], tr['width'])
                    window_dims = np.ceil(np.array(window_dims)).astype(np.int)
                    window_dims = tuple(window_dims.tolist())
                    maxdim = max(window_dims)
                    window_dims = (maxdim, maxdim)
                else:
                    raise KeyError(window_dims)

            data_slice, extra_padding = _get_slice(data_dims, center, window_dims, pad=pad)

        st_dims = [(sl.start, sl.stop) for sl in data_slice[0:ndim]]

        return data_slice, extra_padding, st_dims

    def _load_slice(self, tr, window_dims=None, pad=None,
                    padkw={'mode': 'constant'}):
        """
        Example:
            >>> # sample an out of bounds target
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> tr = self.regions.get_positive(0)
            >>> sample = self._load_slice(tr)
            >>> print('sample = {!r}'.format(ub.map_vals(type, sample)))
        """
        import skimage
        ndim = 2  # number of space-time dimensions (ignore channel)
        if pad is None:
            pad = 0
        pad = tuple(_ensure_iterablen(pad, ndim))

        gid = tr['gid']
        # Determine the image extent
        img = self.dset.imgs[gid]
        data_dims = (img['height'], img['width'])

        data_slice, extra_padding, st_dims = self._rectify_tr(
            tr, data_dims, window_dims=window_dims, pad=pad)

        # Load the image data
        # frame = self.frames.load_image(gid)  # TODO: lazy load on slice
        # im = frame[data_slice]

        im = self.frames.load_region(gid, data_slice)
        if extra_padding:
            if im.ndim != len(extra_padding):
                extra_padding = extra_padding + [(0, 0)]  # Handle channels
            im = np.pad(im, extra_padding, **padkw)

        # Translations for real sub-pixel center positions
        if extra_padding:
            pad_dims = extra_padding[0:ndim]
            st_dims = [(s - pad[0], t + pad[1])
                       for (s, t), pad in zip(st_dims, pad_dims)]

        (y_start, y_stop), (x_start, x_stop) = st_dims[-2:]
        sample_tlbr = kwimage.Boxes([x_start, y_start, x_stop, y_stop], 'tlbr')

        offset = np.array([-x_start, -y_start])

        tf_rel_to_abs = skimage.transform.AffineTransform(
            translation=-offset
        ).params

        sample = {
            'im': im,
            'tr': tr.copy(),
            'params': {
                'offset': offset,
                'tf_rel_to_abs': tf_rel_to_abs,
                'sample_tlbr': sample_tlbr,
                'st_dims': st_dims,
                'data_dims': data_dims,
                'pad': pad,
            },
        }
        return sample

    def _populate_overlap(self, sample, visible_thresh=0.1, with_annots=True):
        """
        Add information about annotations overlapping the sample.

        with_annots can be a + separated string or list of the the special keys:
            'segmentation' and 'keypoints'.

        Example:
            >>> # sample an out of bounds target
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> tr = self.regions.get_item(0)
            >>> sample = self._load_slice(tr)
            >>> sample = self._populate_overlap(sample)
            >>> print('sample = {}'.format(ub.repr2(ub.util_dict.dict_diff(sample, ['im']), nl=-1)))
        """

        if with_annots is True:
            with_annots = ['segmentation', 'keypoints', 'boxes']
        elif isinstance(with_annots, six.string_types):
            with_annots = with_annots.split('+')

        if __debug__:
            for k in with_annots:
                assert k in ['segmentation', 'keypoints', 'boxes'], 'k={!r}'.format(k)

        tr = sample['tr']
        gid = tr['gid']

        params = sample['params']
        sample_tlbr = params['sample_tlbr']
        offset = params['offset']
        data_dims = params['data_dims']
        # tlbr box in original image space around this sampled patch

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

        rel_boxes = abs_boxes.translate(offset)

        # Handle segmentations and keypoints if they exist
        sseg_list = []
        kpts_list = []

        coco_dset = self.dset
        kp_classes = self.kp_classes
        for aid in overlap_aids:
            ann = coco_dset.anns[aid]

            # TODO: it should probably be the regions's responsibilty to load
            # and return these kwimage data structures.
            rel_points = None
            if 'keypoints' in with_annots:
                coco_kpts = ann.get('keypoints', None)
                if coco_kpts is not None and len(coco_kpts) > 0:
                    if isinstance(ub.peek(coco_kpts), dict):
                        # new style coco keypoint encoding
                        abs_points = kwimage.Points.from_coco(
                            coco_kpts, classes=kp_classes)
                    else:
                        # using old style coco keypoint encoding, we need look up
                        # keypoint class from object classes and then pass in the
                        # relevant info
                        kpnames = coco_dset._lookup_kpnames(ann['category_id'])
                        kp_class_idxs = np.array([kp_classes.index(n) for n in kpnames])
                        abs_points = kwimage.Points.from_coco(
                            coco_kpts, kp_class_idxs, kp_classes)
                    rel_points = abs_points.translate(offset)

            rel_sseg = None
            if 'segmentation' in with_annots:
                coco_sseg = ann.get('segmentation', None)
                if coco_sseg is not None:
                    # x = _coerce_coco_segmentation(coco_sseg, data_dims)
                    # abs_sseg = kwimage.Mask.coerce(coco_sseg, dims=data_dims)
                    abs_sseg = kwimage.MultiPolygon.coerce(coco_sseg, dims=data_dims)
                    # abs_sseg = abs_sseg.to_multi_polygon()
                    rel_sseg = abs_sseg.translate(offset)

            kpts_list.append(rel_points)
            sseg_list.append(rel_sseg)

        rel_ssegs = kwimage.PolygonList(sseg_list)
        rel_kpts = kwimage.PointsList(kpts_list)
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
        tr_ = sample['tr']

        main_aid = tr_.get('aid', None)
        if main_aid is not None:
            # Determine which (if any) index in "annots" corresponds to the
            # main aid (if we even have a main aid)
            cand_idxs = np.where(annots['aids'] == main_aid)[0]
            if len(cand_idxs) == 0:
                tr_['annot_idx'] = -1
            elif len(cand_idxs) == 1:
                tr_['annot_idx'] = cand_idxs[0]
            else:
                raise AssertionError('impossible state: len(cand_idxs)={}'.format(len(cand_idxs)))
        else:
            tr_['annot_idx'] = -1

        sample['annots'] = annots
        return sample


def padded_slice(data, in_slice, ndim=None, pad_slice=None,
                 pad_mode='constant', **padkw):
    """
    Allows slices with out-of-bound coordinates.  Any out of bounds coordinate
    will be sampled via padding.

    Note:
        Negative slices have a different meaning here then they usually do.
        Normally, they indicate a wrap-around or a reversed stride, but here
        they index into out-of-bounds space (which depends on the pad mode).
        For example a slice of -2:1 literally samples two pixels to the left of
        the data and one pixel from the data, so you get two padded values and
        one data value.

    Args:
        data (Sliceable[T]): data to slice into. Any channels must be the last dimension.
        in_slice (Tuple[slice, ...]): slice for each dimensions
        ndim (int): number of spatial dimensions
        pad_slice (List[int|Tuple]): additional padding of the slice

    Returns:
        Tuple[Sliceable, Dict] :

            data_sliced: subregion of the input data (possibly with padding,
                depending on if the original slice went out of bounds)

            transform : information on how to return to the original coordinates

                Currently a dict containing:
                    st_dims: a list indicating the low and high space-time
                        coordinate values of the returned data slice.

    Example:
        >>> data = np.arange(5)
        >>> in_slice = [slice(-2, 7)]

        >>> data_sliced, transform = padded_slice(data, in_slice)
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        np.array([0, 0, 0, 1, 2, 3, 4, 0, 0])

        >>> data_sliced, transform = padded_slice(data, in_slice, pad_slice=(3, 3))
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0])

        >>> data_sliced, transform = padded_slice(data, slice(3, 4), pad_slice=[(1, 0)])
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        np.array([2, 3])

    """
    if isinstance(in_slice, slice):
        in_slice = [in_slice]

    ndim = len(in_slice)

    data_dims = data.shape[:ndim]

    low_dims = [sl.start for sl in in_slice]
    high_dims = [sl.stop for sl in in_slice]

    data_slice, extra_padding = _rectify_slice(data_dims, low_dims, high_dims,
                                               pad=pad_slice)

    in_slice_clipped = tuple(slice(*d) for d in data_slice)
    # Get the parts of the image that are in bounds
    data_clipped = data[in_slice_clipped]

    # Add any padding that is needed to behave like negative dims exist
    if sum(map(sum, extra_padding)) == 0:
        # The slice was completely in bounds
        data_sliced = data_clipped
    else:
        if len(data.shape) != len(extra_padding):
            extra_padding = extra_padding + [(0, 0)]
        data_sliced = np.pad(data_clipped, extra_padding, mode=pad_mode,
                             **padkw)

    st_dims = data_slice[0:ndim]
    pad_dims = extra_padding[0:ndim]

    st_dims = [(s - pad[0], t + pad[1])
               for (s, t), pad in zip(st_dims, pad_dims)]

    # TODO: return a better transform back to the original space
    transform = {
        'st_dims': st_dims,
        'st_offset': [d[0] for d in st_dims]
    }
    return data_sliced, transform


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

    data_slice = tuple(slice(low, high) for low, high in data_slice)
    if sum(map(sum, extra_padding)) == 0:
        extra_padding = None
    return data_slice, extra_padding


def _rectify_slice(data_dims, low_dims, high_dims, pad=None):
    """
    Given image dimensions, bounding box dimensions, and a padding get the
    corresponding slice from the image and any extra padding needed to achieve
    the requested window size.

    Args:
        data_dims (tuple): n-dimension data sizes (e.g. 2d height, width)
        low_dims (tuple): bounding box low values (e.g. 2d ymin, xmin)
        high_dims (tuple): bounding box high values (e.g. 2d ymax, xmax)
        pad (tuple): (List[int|Tuple]):
            pad applied to (left and right) / (both) sides of each slice dim

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
    if pad is None:
        pad = 0
    if isinstance(pad, int):
        pad = [pad] * len(data_dims)
    # Normalize to left/right pad value for each dim
    pad_slice = [p if ub.iterable(p) else [p, p] for p in pad]

    # Determine the real part of the image that can be sliced out
    for D_img, d_low, d_high, d_pad in zip(data_dims, low_dims, high_dims, pad_slice):
        if d_low > d_high:
            raise ValueError('d_low > d_high: {} > {}'.format(d_low, d_high))
        # Determine where the bounds would be if the image size was inf
        raw_low = d_low - d_pad[0]
        raw_high = d_high + d_pad[1]
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


def _rectify_slice2(requested_slice, data_dims, pad=None):
    low_dims = [s.start for s in requested_slice]
    high_dims = [s.stop for s in requested_slice]
    data_slice, extra_padding = _rectify_slice(data_dims, low_dims,
                                               high_dims, pad)
    data_slice = tuple(slice(low, high) for low, high in data_slice)
    if sum(map(sum, extra_padding)) == 0:
        extra_padding = None
    return data_slice, extra_padding


def _ensure_iterablen(scalar, n):
    try:
        iter(scalar)
    except TypeError:
        return [scalar] * n
    return scalar


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m ndsampler.coco_sampler
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
