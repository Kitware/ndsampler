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
    >>> import kwcoco
    >>> # First make a COCO dataset that refers to your images (and possibly annotations)
    >>> dataset = {
    >>>     'images': [{'id': i, 'file_name': fpath} for i, fpath in enumerate(image_paths)],
    >>>     'annotations': [],
    >>>     'categories': [],
    >>> }
    >>> coco_dset = kwcoco.CocoDataset(dataset)
    >>> print(coco_dset)
    <CocoDataset(tag=None, n_anns=0, n_imgs=3, ...n_cats=0)>
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
    ...
    >>> print(sorted(sample.keys()))
    ['annots', 'classes', 'im', 'kp_classes', 'params', 'tr']
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
import kwcoco
import warnings
from ndsampler import coco_regions
from ndsampler import coco_frames
from ndsampler import abstract_sampler
from ndsampler.utils import util_misc

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class CocoSampler(abstract_sampler.AbstractSampler, util_misc.HashIdentifiable,
                  ub.NiceRepr):
    """
    Samples patches of positives and negative detection windows from a COCO
    dataset. Can be used for training FCN or RPN based classifiers / detectors.

    Does data loading, padding, etc...

    Args:
        dset (kwcoco.CocoDataset): a coco-formatted dataset

        backend (str | Dict): either 'cog' or 'npy', or a dict with
            `{'type': str, 'config': Dict}`. See AbstractFrames for more
            details. Defaults to None, which does not do anything fancy.

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
    def demo(cls, key='shapes', workdir=None, backend=None, **kw):
        """
        Create a toy coco sampler for testing and demo puposes

        SeeAlso:
            * kwcoco.CocoDataset.demo
        """
        dset = kwcoco.CocoDataset.demo(key=key, **kw)
        if key == 'photos':
            toremove = [ann for ann in dset.anns.values() if 'bbox' not in ann]
            dset.remove_annotations(toremove)
            dset.add_category('background', id=0)
        if workdir is None:
            workdir = ub.ensure_app_cache_dir('ndsampler')
        self = CocoSampler(dset, workdir=workdir, backend=backend)
        return self

    def __init__(self, dset, workdir=None, autoinit=True, backend=None,
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
        if hasattr(self.dset, '_ensure_imgsize'):
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

    def new_sample_grid(self, task, window_dims, window_overlap=0):
        sample_grid = self.regions.new_sample_grid(
            task, window_dims, window_overlap)
        return sample_grid

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

    def load_positive(self, index=None, with_annots=True, tr=None, pad=None,
                      rng=None, **kw):
        """
        Load an item from the the positive pool of regions.

        Args:
            index (int): index of positive target

            pad (tuple): (height, width) extra context to add to each size.
                This helps prevent augmentation from producing boundary effects

            tr (Dict): Extra target arguments like window_dims.

            with_annots (bool | str, default=True):
                if True, also extracts information about any annotation that
                overlaps the region of interest (subject to visibility_thresh).
                Can also be a List[str] that specifies which specific subinfo
                should be extracted. Valid strings in this list are: boxes,
                keypoints, and segmentation.

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
            >>> sample = self.load_positive(pad=(10, 10), tr=dict(window_dims=(3, 3)))
            >>> assert sample['im'].shape[0] == 23
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(sample['im'])
            >>> kwplot.show_if_requested()
        """
        tr_ = self.regions.get_positive(index, rng=rng)
        if tr:
            tr_ = ub.dict_union(tr_, tr)
        sample = self.load_sample(tr_, with_annots=with_annots, pad=pad, **kw)
        return sample

    def load_negative(self, index=None, with_annots=True, tr=None, pad=None,
                      rng=None, **kw):
        """
        Load an item from the the negative pool of regions.

        Args:
            index (int): if specified loads a specific negative from the
                presampled pool, otherwise the next negative in the pool is
                returned.

            with_annots (bool | str, default=True):
                if True, also extracts information about any annotation that
                overlaps the region of interest (subject to visibility_thresh).
                Can also be a List[str] that specifies which specific subinfo
                should be extracted. Valid strings in this list are: boxes,
                keypoints, and segmentation.

            tr (Dict): Extra target arguments like window_dims.

            pad (tuple): (height, width) extra context to add to each size.
                This helps prevent augmentation from producing boundary effects

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
            >>> sample = self.load_negative(rng=rng, pad=(0, 0), tr=dict(window_dims=(64, 64)))
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> box = kwimage.Boxes(tr.reindex(['rel_cx', 'rel_cy', 'width', 'height']).values, 'cxywh')
            >>> kwplot.imshow(sample, fnum=1, doclf=True)
            >>> kwplot.draw_boxes(box)
            >>> kwplot.show_if_requested()
        """
        tr_ = self.regions.get_negative(index, rng=rng)
        if tr:
            tr_ = ub.dict_union(tr_, tr)
        sample = self.load_sample(tr_, with_annots=with_annots, pad=pad, **kw)
        return sample

    def load_sample(self, tr, with_annots=True, visible_thresh=0.0, pad=None,
                    padkw={'mode': 'constant'}, **kw):
        """
        Loads the volume data associated with the bbox and frame of a target

        Args:
            tr (dict): target dictionary indicating an nd source object (e.g.
                image or video) and the coordinate region to sample from.
                Unspecified coordinate regions default to the extent of the
                source object.

                For 2D image source objects, tr must contain or be able to
                infer the key `gid (int)`, to specify an image id.

                For 3D video source objects, tr must contain the key
                `vidid (int)`, to specify a video id. (NEW in 0.6.1)

                In general, coordinate regions can specified by the key
                `slices`, a numpy-like "fancy index" over each of the n
                dimensions. Usually this is a tuple of slices, e.g.
                (y1:y2, x1:x2) for images and (t1:t2, y1:y2, x1:x2) for videos.

                Spatial regions can be specified with keys:
                    * 'cx' and 'cy' as the center of the region in pixels.
                    * 'width' and 'height' are in pixels.
                    * 'window_dims' is a height, width tuple or can be a
                    special string key 'square', which overrides width and
                    height to both be the maximum of the two.

                Temporal regions are currently only specifiable by slices. This
                will change in the future.

                The `aid` key can be specified to indicate a specific
                annotation to load. This uses the annotation information to
                infer 'gid', 'cx', 'cy', 'width', and 'height' if they are not
                present. (NEW in 0.5.10)

                The `channels` key can be specified as a channel code or list
                of channel codes indicating a subset of channels to load.
                (NEW in 0.6.1)

            pad (tuple): (height, width) extra context to add to window dims.
                This helps prevent augmentation from producing boundary effects

            visible_thresh (float): does not return annotations with visibility
                less than this threshold.

            padkw (dict): kwargs for `numpy.pad`

            with_annots (bool | str, default=True):
                if True, also extracts information about any annotation that
                overlaps the region of interest (subject to visibility_thresh).
                Can also be a List[str] that specifies which specific subinfo
                should be extracted. Valid strings in this list are: boxes,
                keypoints, and segmentation.

            window_dims (tuple | str): (height, width) overrides the height/width
                in tr to determine the extracted window size. Can also be
                'extent' or 'square', which determines the final size using
                target information.

                DEPRECATED. IF DESIRED SPECIFY IN THE TARGET DICTIONARY

        Returns:
            Dict: sample: dict containing keys
                im (ndarray): image data
                tr (dict): contains the same input items as tr but additionally
                    specifies rel_cx and rel_cy, which gives the center
                    of the target w.r.t the returned **padded** sample.
                annots (dict): containing items:
                    frame_dets (List[kwimage.Detections]): a list of detection
                        objects containing the requested annotation info for each
                        frame.
                    aids (list): annotation ids DEPRECATED
                    cids (list): category ids DEPRECATED
                    rel_ssegs (ndarray): segmentations relative to the sample DEPRECATED
                    rel_kpts (ndarray): keypoints relative to the sample DEPRECATED

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
            ...
            >>> print('sample.shape = {!r}'.format(sample['im'].shape))
            sample.shape = (6, 6, 3)

        Example:
            >>> # Access direct annotation information
            >>> import ndsampler
            >>> sampler = ndsampler.CocoSampler.demo()
            >>> # Sample a region that contains at least one annotation
            >>> tr = {'gid': 1, 'cx': 5, 'cy': 2, 'width': 600, 'height': 600}
            >>> sample = sampler.load_sample(tr)
            >>> annotation_ids = sample['annots']['aids']
            >>> aid = annotation_ids[0]
            >>> # Method1: Access ann dict directly via the coco index
            >>> ann = sampler.dset.anns[aid]
            >>> # Method2: Access ann objects via annots method
            >>> dets = sampler.dset.annots(annotation_ids).detections
            >>> print('dets.data = {}'.format(ub.repr2(dets.data, nl=1)))

        Example:
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> tr = self.regions.get_positive(0)
            >>> pad = (25, 25)
            >>> tr['window_dims'] = 'square'
            >>> sample = self.load_sample(tr, pad=pad)
            >>> print('im.shape = {!r}'.format(sample['im'].shape))
            im.shape = (135, 135, 3)
            >>> pad = (0, 0)
            >>> tr['window_dims'] = None
            >>> sample = self.load_sample(tr, pad=pad)
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
            >>> tr['window_dims'] = (364, 364)
            >>> sample = self.load_sample(tr)
            >>> annots = sample['annots']
            >>> assert len(annots['aids']) > 0
            >>> #assert len(annots['rel_cxywh']) == len(annots['aids'])
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> abs_frame = self.frames.load_image(sample['tr']['gid'])[:]
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
            >>> pad = None
            >>> tr['window_dims'] = (300, 150)
            >>> sample = self.load_sample(tr, pad)
            >>> assert sample['im'].shape[0:2] == tr['window_dims']
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(sample['im'], colorspace='rgb')
            >>> kwplot.show_if_requested()

        Example:
            >>> # Multispectral video sample example
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo('vidshapes1-multispectral', num_frames=5)
            >>> sample_grid = self.new_sample_grid('video_detection', (3, 128, 128))
            >>> tr = sample_grid['positives'][0]
            >>> tr['channels'] = 'B1|B8'
            >>> sample = self.load_sample(tr)
            >>> assert sample['im'].shape == (3, 128, 128, 2)
            >>> tr['channels'] = '<all>'
            >>> sample = self.load_sample(tr)
            >>> assert sample['im'].shape == (3, 128, 128, 5)
        """
        if len(kw):
            raise Exception(
                'The load_sample API has deprecated arguments that should now '
                ' be given in `tr` itself. You specified {}'.format(list(kw)))

        sample = self._load_slice(tr, pad, padkw)

        if with_annots or ub.iterable(with_annots):
            self._populate_overlap(sample, visible_thresh, with_annots)

        sample['classes'] = self.classes
        sample['kp_classes'] = self.kp_classes
        return sample

    @profile
    def _infer_target_attributes(self, tr):
        """
        Infer unpopulated target attribues

        Example:
            >>> # sample using only an annotation id
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> tr = {'aid': 1}
            >>> tr_ = self._infer_target_attributes(tr)
            >>> assert tr_['gid'] == 1
            >>> assert all(k in tr_ for k in ['cx', 'cy', 'width', 'height'])
        """
        window_dims = None
        # we might modify the target
        tr_ = tr.copy()
        if 'aid' in tr_:
            # If the annotation id is specified, infer other unspecified fields
            aid = tr_['aid']
            try:
                ann = self.dset.anns[aid]
            except KeyError:
                pass
            else:
                if 'gid' not in tr_:
                    tr_['gid'] = ann['image_id']
                if len({'cx', 'cy', 'width', 'height'} & set(tr_)) != 4:
                    box = kwimage.Boxes([ann['bbox']], 'xywh')
                    cx, cy, width, height = box.to_cxywh().data[0]
                    if 'cx' not in tr_:
                        tr_['cx'] = cx
                    if 'cy' not in tr_:
                        tr_['cy'] = cy
                    if 'width' not in tr_:
                        tr_['width'] = width
                    if 'height' not in tr_:
                        tr_['height'] = height
                if 'category_id' not in tr_:
                    tr_['category_id'] = ann['category_id']

        ndims = 3 if 'vidid' in tr_ else 2

        # other spatial specifiers allowed if slices is not given
        alternate_keys = {'cx', 'cy', 'height', 'width'}
        has_alternate = bool(set(tr_) & alternate_keys)

        if 'slices' in tr_:
            # Slice was explicitly specified
            if has_alternate or window_dims:
                warnings.warn(ub.paragraph(
                    '''
                    data_slice was specified, but ignored keys are present
                    '''))
        elif not has_alternate:
            # No region specified. load everything.
            if ndims == 3:
                tr_['slices'] = (slice(0, None), slice(0, None), slice(0, None))
            else:
                tr_['slices'] = (slice(0, None), slice(0, None))
        else:
            if ndims == 3:
                raise NotImplementedError
            # A center / width / height was specified
            center = (tr_['cy'], tr_['cx'])
            # Determine the requested window size
            window_dims = tr_.get('window_dims', window_dims)
            if window_dims is None:
                window_dims = 'extent'

            if isinstance(window_dims, six.string_types):
                if window_dims == 'extent':
                    window_dims = (tr_['height'], tr_['width'])
                    window_dims = np.ceil(np.array(window_dims)).astype(np.int)
                    window_dims = tuple(window_dims.tolist())
                elif window_dims == 'square':
                    window_dims = (tr_['height'], tr_['width'])
                    window_dims = np.ceil(np.array(window_dims)).astype(np.int)
                    window_dims = tuple(window_dims.tolist())
                    maxdim = max(window_dims)
                    window_dims = (maxdim, maxdim)
                else:
                    raise KeyError(window_dims)
            tr_['window_dims'] = window_dims
            tr_['slices'] = _center_extent_to_slice(center, window_dims)

        if ndims == 3:
            vidid = tr['vidid']
            vid_gids = self.dset.index.vidid_to_gids[vidid]

        if any(sl.stop is None for sl in tr_['slices']):
            # Fix non-determined bounds
            if ndims == 2:
                gid = tr['gid']
                img = self.dset.index.imgs[gid]
                data_dims = img['height'], img['width']
            else:
                video = self.dset.index.videos[vidid]
                data_dims = len(vid_gids), video['height'], video['width']

            fixed = []
            for sl, D in zip(tr_['slices'], data_dims):
                stop = D if sl.stop is None else sl.stop
                fixed.append(slice(sl.start, stop, sl.step))
            tr_['slices'] = tuple(fixed)

        return tr_

    @profile
    def _load_slice(self, tr, pad=None, padkw={'mode': 'constant'}):
        """
        Example:
            >>> # sample an out of bounds target
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> tr = self.regions.get_positive(0)
            >>> sample = self._load_slice(tr)
            >>> print('sample = {!r}'.format(ub.map_vals(type, sample)))

            >>> # sample an out of bounds target
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo('vidshapes2')
            >>> tr = self._infer_target_attributes({'vidid': 1})
            >>> sample = self._load_slice(tr)
            >>> print('sample = {!r}'.format(ub.map_vals(type, sample)))
        """
        import skimage
        import kwarray
        from kwcoco import channel_spec
        from kwimage.transform import Affine
        import xarray as xr
        if pad is None:
            pad = 0

        tr_ = self._infer_target_attributes(tr)
        assert 'slices' in tr_
        requested_slice = tr_['slices']
        channels = tr_.get('channels', ub.NoParam)

        if channels == '<all>' or channels is ub.NoParam:
            # Do something special
            all_chan = True
        else:
            request_chanspec = channel_spec.ChannelSpec.coerce(channels)
            requeset_chan_coords = ub.oset(ub.flatten(request_chanspec.normalize().values()))
            all_chan = False

        vidid = tr_.get('vidid', None)
        if vidid is not None:
            ndim = 3  # number of space-time dimensions (ignore channel)
            pad = tuple(_ensure_iterablen(pad, ndim))

            # As of kwcoco 0.2.1 gids are ordered by frame index
            vid_gids = self.dset.index.vidid_to_gids[vidid]
            video = self.dset.index.videos[vidid]
            vid_width = video.get('width', None)
            vid_height = video.get('height', None)
            num_frames = len(vid_gids)
            if vid_height is None or vid_width is None:
                # Fallback on the first image
                img = self.dset.imgs[vid_gids[0]]
                vid_width = img['width']
                vid_height = img['height']

            vid_dsize = (vid_width, vid_height)
            data_dims = (num_frames, vid_height, vid_width)

            data_slice, extra_padding = kwarray.embed_slice(
                requested_slice, data_dims, pad)

            # TODO: frames should have better nd-support, hack it for now to
            # just load the 2d data for each image
            time_slice, *space_slice = data_slice
            space_slice = tuple(space_slice)
            time_gids = vid_gids[time_slice]
            space_frames = []

            slice_height = space_slice[0].stop - space_slice[0].start
            slice_width = space_slice[1].stop - space_slice[1].start

            # TODO: Handle channel encodings more ellegantly

            # HACKED AND NOT ELEGANT OR EFFICIENT.
            # MOST OF THIS LOGIC SHOULD BE IN WHATEVER THE TIME-SAMPLING VIDEO
            # MECHANISM IS
            for time_idx, gid in enumerate(time_gids):
                img = self.dset.imgs[gid]
                tf_img_to_vid = Affine.coerce(img['warp_img_to_vid'])

                alignable = self.frames._load_alignable(gid)
                frame_chan_names = list(alignable.pathinfo['channels'].keys())

                chan_frames = []
                for frame_chan_name in frame_chan_names:
                    frame_spec = channel_spec.ChannelSpec.coerce(frame_chan_name)
                    file_chan_coords = ub.oset(ub.flatten(frame_spec.normalize().values()))

                    if all_chan:
                        matching_coords = file_chan_coords
                    else:
                        matching_coords = file_chan_coords & requeset_chan_coords

                    if matching_coords:
                        # Load full image in "virtual" image space
                        img_full = alignable._load_delayed_channel(frame_chan_name)
                        vid_full = img_full.delayed_warp(tf_img_to_vid, dsize=vid_dsize)

                        vid_part = vid_full.delayed_crop(space_slice)

                        # TODO: only load some of the channels if that is an
                        # option
                        _vid_chan_frame = vid_part.finalize()

                        if 1:
                            # TODO: can we test if we can simplify this to None
                            # or a slice? Maybe we can write a function
                            # simplify slice?
                            subchan_idxs = [file_chan_coords.index(c)
                                            for c in matching_coords]
                            vid_chan_frame = _vid_chan_frame[..., subchan_idxs]

                        # TODO: we could add utm coords here
                        xr_chan_frame = xr.DataArray(
                            vid_chan_frame[None, ...],
                            dims=('t', 'y', 'x', 'c'),
                            coords={
                                't': np.array([time_idx]),
                                # 'y': np.arange(vid_chan_frame.shape[0]),
                                # 'x': np.arange(vid_chan_frame.shape[1]),
                                'c': list(matching_coords),
                            }
                        )
                        chan_frames.append(xr_chan_frame)

                if len(chan_frames):
                    xr_frame = xr.concat(chan_frames, dim='c')
                else:
                    # TODO: we could add utm coords here
                    xr_frame = xr.DataArray(
                        np.empty((1, slice_height, slice_width, 0)),
                        dims=('t', 'y', 'x', 'c'),
                        coords={
                            't': np.array([time_idx]),
                            'c': np.empty((0,), dtype=str),
                        }
                    )
                space_frames.append(xr_frame)

            # Concat aligned frames together (add nans for non-existing
            # channels)

            _data_clipped = xr.concat(space_frames, dim='t')
            if all_chan:
                # This is not the right thing to do for rgb data
                c = sorted(_data_clipped.coords['c'].values.tolist())
                _data_clipped = _data_clipped.sel(c=c)
            else:
                # have = set(_data_clipped.coords['c'].values.tolist())
                # if len(have & requeset_chan_coords):
                #     c = list(requeset_chan_coords)
                #     _data_clipped = _data_clipped.sel(c=c)
                # else:
                exist = ub.oset(_data_clipped.coords['c'].values.tolist())
                missing = requeset_chan_coords - exist
                if len(missing):
                    _data_clipped = _data_clipped.pad({'c': (0, len(missing))})
                    _data_clipped.coords['c'] = list(exist) + list(missing)
                c = list(requeset_chan_coords)
                _data_clipped = _data_clipped.sel(c=c)

            # Hack to return some info about dims and not returning the xarray
            # itself. In the future we will likely return the xarray itself.
            tr_['_coords'] = _data_clipped.coords
            tr_['_dims'] = _data_clipped.dims

            data_clipped = _data_clipped.values

            # TODO: gids should be padded if it goes oob.
            tr_['_data_gids'] = time_gids
        else:
            ndim = 2  # number of space-time dimensions (ignore channel)
            pad = tuple(_ensure_iterablen(pad, ndim))
            gid = tr_['gid']
            # Determine the image extent
            img = self.dset.imgs[gid]
            data_dims = (img['height'], img['width'])

            data_slice, extra_padding = kwarray.embed_slice(
                requested_slice, data_dims, pad)

            # Load the image data
            # frame = self.frames.load_image(gid)  # TODO: lazy load on slice
            # im = frame[data_slice]

            data_clipped = self.frames.load_region(
                image_id=gid, region=data_slice, channels=channels)

        # Apply the padding
        if sum(map(sum, extra_padding)) == 0:
            # No padding was requested
            data_sliced = data_clipped
        else:
            trailing_dims = len(data_clipped.shape) - len(extra_padding)
            if trailing_dims > 0:
                extra_padding = extra_padding + ([(0, 0)] * trailing_dims)
            data_sliced = np.pad(data_clipped, extra_padding, **padkw)

        st_dims = [(sl.start - pad_[0], sl.stop + pad_[1])
                   for sl, pad_ in zip(data_slice, extra_padding)]

        (y_start, y_stop), (x_start, x_stop) = st_dims[-2:]

        sample_tlbr = kwimage.Boxes([x_start, y_start, x_stop, y_stop], 'ltrb')
        offset = np.array([-x_start, -y_start])
        tf_rel_to_abs = skimage.transform.AffineTransform(
            translation=-offset
        ).params

        if 0:
            print('data_sliced.shape = {!r}'.format(data_sliced.shape))
            print('data_sliced.dtype = {!r}'.format(data_sliced.dtype))

        sample = {
            'im': data_sliced,
            'tr': tr_,
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

    @profile
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

        if '_data_gids' in tr:
            gids = tr['_data_gids']
        else:
            gids = [tr['gid']]

        params = sample['params']
        sample_tlbr = params['sample_tlbr']
        offset = params['offset']
        data_dims = params['data_dims']
        space_dims = data_dims[-2:]

        # accumulate information over all frames
        frame_dets = []

        for rel_frame_idx, gid in enumerate(gids):

            # Check to see if there is a transform between the image-space and
            # the sampling-space (currently this can only be done by a video,
            # but in the future the user might be able to adjust sample scale)
            if tr.get('vidid', None) is not None:
                # hack to align annots from image space to video space
                img = self.dset.imgs[gid]
                from kwimage.transform import Affine
                tf_img_to_abs = Affine.coerce(img.get('warp_img_to_vid', None))
                tf_abs_to_img = tf_img_to_abs.inv()
            else:
                tf_img_to_abs = None

            # check overlap in image space
            if tf_img_to_abs is None:
                sample_tlbr_ = sample_tlbr
            else:
                sample_tlbr_ = sample_tlbr.warp(tf_abs_to_img.matrix).quantize()

            # Find which bounding boxes are visible in this region
            overlap_aids = self.regions.overlapping_aids(
                gid, sample_tlbr_, visible_thresh=visible_thresh)

            # Get info about all annotations inside this window

            overlap_anns = [self.dset.anns[aid] for aid in overlap_aids]
            overlap_cids = [ann['category_id'] for ann in overlap_anns]
            abs_boxes = kwimage.Boxes(
                [ann['bbox'] for ann in overlap_anns], 'xywh')

            # Handle segmentations and keypoints if they exist
            coco_dset = self.dset
            kp_classes = self.kp_classes
            classes = self.classes
            sseg_list = []
            kpts_list = []
            for ann in overlap_anns:
                # TODO: it should probably be the regions's responsibilty to load
                # and return these kwimage data structures.
                abs_points = None
                abs_sseg = None
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
                if 'segmentation' in with_annots:
                    coco_sseg = ann.get('segmentation', None)
                    if coco_sseg is not None:
                        # x = _coerce_coco_segmentation(coco_sseg, space_dims)
                        # abs_sseg = kwimage.Mask.coerce(coco_sseg, dims=space_dims)
                        abs_sseg = kwimage.MultiPolygon.coerce(coco_sseg, dims=space_dims)
                sseg_list.append(abs_sseg)
                kpts_list.append(abs_points)

            abs_ssegs = kwimage.PolygonList(sseg_list)
            abs_kpts = kwimage.PointsList(kpts_list)
            abs_kpts.meta['classes'] = self.kp_classes

            # Construct a detections object containing absolute annotation
            # positions
            abs_dets = kwimage.Detections(
                aids=np.array(overlap_aids),
                cids=np.array(overlap_cids),
                boxes=abs_boxes,
                segmentations=abs_ssegs,
                keypoints=abs_kpts,
                classes=classes,
                rel_frame_index=rel_frame_idx,
                gid=gid,
                datakeys=['aids', 'cids'],
                metakeys=['gid', 'rel_frame_index']
            )

            # Translate the absolute detections to relative sample coordinates
            if tf_img_to_abs is not None:
                # hack to align annots from image space to video space
                tf_abs_to_rel = Affine.translate(offset) @ tf_img_to_abs
                rel_dets = abs_dets.warp(tf_abs_to_rel.matrix)
            else:
                rel_dets = abs_dets.translate(offset)

            frame_dets.append(rel_dets)

        # if len(frame_dets) == 1:
        #     annots = frame_dets[0].data
        # else:
        # Hack to get multi-frame annots without too much developer effort.
        # Could be more efficient
        annots = {
            'aids': np.hstack([x.data['aids'] for x in frame_dets]),
            'cids': np.hstack([x.data['cids'] for x in frame_dets]),
            'rel_frame_index': np.hstack([[x.meta['rel_frame_index']] * len(x) for x in frame_dets]),
            'rel_boxes': kwimage.Boxes.concatenate([x.data['boxes'] for x in frame_dets]),
            'rel_ssegs': kwimage.PolygonList(list(ub.flatten([x.data['segmentations'].data for x in frame_dets]))),
            'rel_kpts': kwimage.PointsList(list(ub.flatten([x.data['keypoints'].data for x in frame_dets]))),
            'frame_dets': frame_dets,
            # Removed:
            # 'rel_cxywh': np.vstack([x['rel_cxywh'] for x in frame_accum]),
            # 'abs_cxywh': np.vstack([x['abs_cxywh'] for x in frame_accum]),
        }
        annots['rel_kpts'].meta['classes'] = self.kp_classes

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


def _center_extent_to_slice(center, window_dims):
    """
    Transforms a center and window dimensions into a start/stop slice

    Args:
        center (Tuple[float]): center location (cy, cx)
        window_dims (Tuple[int]): window size (height, width)

    Returns:
        Tuple[slice, ...]: the slice corresponding to the centered window

    Example:
        >>> center = (2, 5)
        >>> window_dims = (6, 6)
        >>> slices = _center_extent_to_slice(center, window_dims)
        >>> assert slices == (slice(-1, 5), slice(2, 8))

     Example:
        >>> center = (2, 5)
        >>> window_dims = (64, 64)
        >>> slices = _center_extent_to_slice(center, window_dims)
        >>> assert slices == (slice(-30, 34, None), slice(-27, 37, None))

    Example:
        >>> # Test floating point error case
        >>> center = (500.5, 974.9999999999999)
        >>> window_dims  = (100, 100)
        >>> slices = _center_extent_to_slice(center, window_dims)
        >>> assert slices == (slice(450, 550, None), slice(924, 1024, None))
    """
    # Compute lower and upper coordinates of the window bounding box
    low_dims = [int(np.floor(c - d_win / 2.0))
                for c, d_win in zip(center, window_dims)]
    high_dims = [int(np.floor(c + d_win / 2.0))
                 for c, d_win in zip(center, window_dims)]

    # Floating point errors can cause the slice window size to be different
    # from the requested one. We check and correct for this.
    for idx, tup in enumerate(zip(window_dims, low_dims, high_dims)):
        d_win, d_low, d_high = tup
        d_win_got = d_high - d_low
        delta = d_win - d_win_got
        if delta:
            high_dims[idx] += delta

    if __debug__:
        for d_win, d_low, d_high in zip(window_dims, low_dims, high_dims):
            d_win_got = d_high - d_low
            assert d_win_got == d_win, 'slice has incorrect window size'

    slices = tuple([slice(s, t) for s, t in zip(low_dims, high_dims)])
    return slices


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
