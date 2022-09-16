"""
The CocoSampler is the ndsampler interface for efficiently sampling windowed
data from a :class:`kwcoco.CocoDataset`.

CommandLine:
    xdoctest -m ndsampler.coco_sampler __doc__ --show

Example:
    >>> # Imagine you have some images
    >>> import kwimage
    >>> image_paths = [
    >>>     kwimage.grab_test_image_fpath('astro'),
    >>>     kwimage.grab_test_image_fpath('carl'),
    >>>     kwimage.grab_test_image_fpath('airport'),
    >>> ]  # xdoctest: +IGNORE_WANT
    ['~/.cache/kwimage/demodata/KXhKM72.png',
     '~/.cache/kwimage/demodata/flTHWFD.png',
     '~/.cache/kwimage/demodata/Airport.jpg']
    >>> # And you want to randomly load subregions of them in O(1) time
    >>> import ndsampler
    >>> import kwcoco
    >>> # First make a COCO dataset that refers to your images
    >>> dataset = {
    >>>     'images': [{'id': i, 'file_name': fpath} for i, fpath in enumerate(image_paths)],
    >>>     'annotations': [],
    >>>     'categories': [],
    >>> }
    >>> coco_dset = kwcoco.CocoDataset(dataset)
    >>> # (and possibly annotations)
    >>> category_id = coco_dset.ensure_category('face')
    >>> image_id = 0
    >>> coco_dset.add_annotation(image_id=image_id, category_id=category_id, bbox=kwimage.Boxes([[140, 10, 180, 180]], 'xywh'))
    >>> print(coco_dset)
    <CocoDataset(tag=None, n_anns=1, n_imgs=3, ... n_cats=1)>
    >>> # Now pass the dataset to a sampler and tell it where it can store temporary files
    >>> workdir = ub.ensure_app_cache_dir('ndsampler/demo')
    >>> sampler = ndsampler.CocoSampler(coco_dset, workdir=workdir)
    >>> # Now you can load arbirary samples by specifing a target dictionary
    >>> # with an image_id (gid) center location (cx, cy) and width, height.
    >>> target = {'gid': 0, 'cx': 220, 'cy': 100, 'width': 300, 'height': 300}
    >>> sample = sampler.load_sample(target)
    >>> # The sample contains the image data, any visible annotations, a reference
    >>> # to the original target, and params of the transform used to sample this
    >>> # patch
    ...
    >>> print(sorted(sample.keys()))
    ['annots', 'classes', 'im', 'kp_classes', 'params', 'target', 'tr']
    >>> im = sample['im']
    >>> print(f'im.shape={im.shape}')
    im.shape=(300, 300, 3)
    >>> dets = sample['annots']['frame_dets'][0]
    >>> print(f'dets={dets}')
    >>> print('dets.data = {}'.format(ub.repr2(dets.data, nl=1, sv=1)))
    dets=<Detections(1)>
    dets.data = {
        'aids': [1],
        'boxes': <Boxes(xywh, array([[ 70.,  60., 180., 180.]]))>,
        'cids': [1],
        'keypoints': <PointsList(n=1)>,
        'segmentations': <SegmentationList(n=1)>,
    }
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> kwplot.imshow(im)
    >>> dets.draw(labels=False)
    >>> kwplot.show_if_requested()
    >>> # The load sample function is at the core of what ndsampler does
    >>> # There are other helper functions like load_positive / load_negative
    >>> # which deal with annotations. See those for more details.
    >>> # For random negative sampling see coco_regions.
"""
import ubelt as ub
import numpy as np
import kwimage
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
        #print
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
            >>> import ndsampler
            >>> self = ndsampler.CocoSampler.demo()
            >>> img, anns = self.load_image_with_annots(1)
            >>> dets = kwimage.Detections.from_coco_annots(anns, dset=self.dset)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(img['imdata'][:], doclf=1)
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

    def load_item(self, index, with_annots=True, target=None, rng=None, **kw):
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

            with_annots (bool | str, default=True):
                if True, also extracts information about any annotation that
                overlaps the region of interest (subject to visibility_thresh).
                Can also be a List[str] that specifies which specific subinfo
                should be extracted. Valid strings in this list are: boxes,
                keypoints, and segmenation.

            target (Dict): Extra target arguments that update the positive target,
                like window_dims, pad, etc.... See :func:`load_sample` for
                details on allowed keywords.

            rng (None | int | RandomState):
                a seed or seeded random number generator.

            **kw : other arguments that can be passed to :func:`CocoSampler.load_sample`

        Returns:
            Dict: sample: dict containing keys
                im (ndarray): image data
                target (dict): contains the same input items as the input
                    target but additionally specifies inferred information like
                    rel_cx and rel_cy, which gives the center of the target
                    w.r.t the returned **padded** sample.
                annots (dict): Dict of aids, cids, and rel/abs boxes
        """
        if index < self.n_positives:
            sample = self.load_positive(index, target=target,
                                        with_annots=with_annots, rng=rng, **kw)
        else:
            index = index - self.n_positives
            sample = self.load_negative(index, target=target,
                                        with_annots=with_annots, rng=rng, **kw)
        return sample

    def load_positive(self, index=None, with_annots=True, target=None, rng=None,
                      **kw):
        """
        Load an item from the the positive pool of regions.

        Args:
            index (int): index of positive target

            with_annots (bool | str, default=True):
                if True, also extracts information about any annotation that
                overlaps the region of interest (subject to visibility_thresh).
                Can also be a List[str] that specifies which specific subinfo
                should be extracted. Valid strings in this list are: boxes,
                keypoints, and segmentation.

            target (Dict): Extra target arguments that update the positive target,
                like window_dims, pad, etc.... See :func:`load_sample` for
                details on allowed keywords.

            rng (None | int | RandomState):
                a seed or seeded random number generator.

            **kw : other arguments that can be passed to :func:`CocoSampler.load_sample`

        Returns:
            Dict: sample: dict containing keys
                im (ndarray): image data
                tr (dict): contains the same input items as tr but additionally
                    specifies rel_cx and rel_cy, which gives the center
                    of the target w.r.t the returned **padded** sample.
                annots (dict): Dict of aids, cids, and rel/abs boxes

        Example:
            >>> import ndsampler
            >>> self = ndsampler.CocoSampler.demo()
            >>> sample = self.load_positive(pad=(10, 10), tr=dict(window_dims=(3, 3)))
            >>> assert sample['im'].shape[0] == 23
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(sample['im'], doclf=1)
            >>> kwplot.show_if_requested()
        """
        if 'tr' in kw:
            ub.schedule_deprecation('ndsampler', 'tr', 'keyword arg of load_positive',
                                    migration='use target instead',
                                    deprecate='0.7.0', error='1.0.0', remove='1.1.0')
            if target is not None:
                warnings.warn('deprecated tr is overriding target')
            target = kw.pop('tr')
        target_ = self.regions.get_positive(index, rng=rng)
        if target:
            target_ = ub.dict_union(target_, target)
        sample = self.load_sample(target_, with_annots=with_annots, **kw)
        return sample

    def load_negative(self, index=None, with_annots=True, target=None,
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

            target (Dict): Extra target arguments that update the positive target,
                like window_dims, pad, etc.... See :func:`load_sample` for
                details on allowed keywords.

            rng (None | int | RandomState):
                a seed or seeded random number generator.

        Returns:
            Dict: sample: dict containing keys
                im (ndarray): image data
                tr (dict): contains the same input items as tr but additionally
                    specifies rel_cx and rel_cy, which gives the center
                    of the target w.r.t the returned **padded** sample.
                annots (dict): Dict of aids, cids, and rel/abs boxes

        Example:
            >>> import ndsampler
            >>> self = ndsampler.CocoSampler.demo()
            >>> rng = None
            >>> sample = self.load_negative(rng=rng, pad=(0, 0))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> import kwimage
            >>> kwplot.autompl()
            >>> abs_sample_box = sample['params']['sample_tlbr']
            >>> tf_rel_from_abs = kwimage.Affine.coerce(sample['params']['tf_rel_to_abs']).inv()
            >>> wh, ww = sample['target']['window_dims']
            >>> abs_window_box = kwimage.Boxes([[sample['target']['cx'], sample['target']['cy'], ww, wh]], 'cxywh')
            >>> rel_window_box = abs_window_box.warp(tf_rel_from_abs)
            >>> rel_sample_box = abs_sample_box.warp(tf_rel_from_abs)
            >>> kwplot.imshow(sample['im'], fnum=1, doclf=True)
            >>> rel_sample_box.draw(color='kw_green', lw=10)
            >>> rel_window_box.draw(color='kw_blue', lw=8)
            >>> kwplot.show_if_requested()

        Example:
            >>> import ndsampler
            >>> self = ndsampler.CocoSampler.demo()
            >>> rng = None
            >>> sample = self.load_negative(rng=rng, pad=(10, 20), target=dict(window_dims=(64, 64)))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> import kwimage
            >>> kwplot.autompl()
            >>> abs_sample_box = sample['params']['sample_tlbr']
            >>> tf_rel_from_abs = kwimage.Affine.coerce(sample['params']['tf_rel_to_abs']).inv()
            >>> wh, ww = sample['target']['window_dims']
            >>> abs_window_box = kwimage.Boxes([[sample['target']['cx'], sample['target']['cy'], ww, wh]], 'cxywh')
            >>> rel_window_box = abs_window_box.warp(tf_rel_from_abs)
            >>> rel_sample_box = abs_sample_box.warp(tf_rel_from_abs)
            >>> kwplot.imshow(sample['im'], fnum=1, doclf=True)
            >>> rel_sample_box.draw(color='kw_green', lw=10)
            >>> rel_window_box.draw(color='kw_blue', lw=8)
            >>> kwplot.show_if_requested()
        """
        if 'tr' in kw:
            ub.schedule_deprecation('ndsampler', 'tr', 'keyword arg of load_negative',
                                    migration='use target instead',
                                    deprecate='0.7.0', error='1.0.0', remove='1.1.0')
            if target is not None:
                warnings.warn('deprecated tr is overriding target')
            target = kw.pop('tr')
        target_ = self.regions.get_negative(index, rng=rng)
        if target:
            target_ = ub.dict_union(target_, target)
        sample = self.load_sample(target_, with_annots=with_annots, **kw)
        return sample

    def load_sample(self, target=None, with_annots=True, visible_thresh=0.0,
                    **kwargs):
        """
        Loads the volume data associated with the bbox and frame of a target

        Args:
            target (dict): target dictionary (often abbreviated as tr)
                indicating an nd source object (e.g.  image or video) and the
                coordinate region to sample from.  Unspecified coordinate
                regions default to the extent of the source object.

                For 2D image source objects, target must contain or be able to
                infer the key `gid (int)`, to specify an image id.

                For 3D video source objects, target must contain the key
                `vidid (int)`, to specify a video id (NEW in 0.6.1) or
                `gids List[int]`, as a list of images in a video (NEW in 0.6.2)

                In general, coordinate regions can specified by the key
                `slices`, a numpy-like "fancy index" over each of the n
                dimensions. Usually this is a tuple of slices, e.g.
                (y1:y2, x1:x2) for images and (t1:t2, y1:y2, x1:x2) for videos.

                You may also specify:
                `space_slice` as (y1:y2, x1:x2) for both 2D images and 3D
                videos and `time_slice` as t1:t2 for 3D videos.

                Spatial regions can be specified with keys:
                    * 'cx' and 'cy' as the center of the region in pixels.
                    * 'width' and 'height' are in pixels.
                    * 'window_dims' is a height, width tuple or can be a
                    special string key 'square', which overrides width and
                    height to both be the maximum of the two.

                Temporal regions are specifiable by `slices`, `time_slice` or
                an explicit list of `gids`.

                The `aid` key can be specified to indicate a specific
                annotation to load. This uses the annotation information to
                infer 'gid', 'cx', 'cy', 'width', and 'height' if they are not
                present. (NEW in 0.5.10)

                The `channels` key can be specified as a channel code or
                    :class:`kwcoco.ChannelSpec` object.  (NEW in 0.6.1)

                as_xarray (bool, default=False):
                    if True, return the image data as an xarray object

                interpolation (str, default='auto'):
                    type of resample interpolation

                antialias (str, default='auto'):
                    antialias sample or not

                nodata: override function level nodata

                use_native_scale (bool): If True, the "im" field is returned
                    as a jagged list of data that are as close to native
                    resolution as possible while still maintaining alignment up
                    to a scale factor. Currently only available for video
                    sampling.

                scale (float | Tuple[float, float]):
                    if specified, the same window is sampled, but the data is
                    returned warped by the extra scale factor. This augments
                    the existing image or video scale factor. Any annotations
                    are also warped according to this factor such that they
                    align with the returned data.

                pad (tuple): (height, width) extra context to add to window dims.
                    This helps prevent augmentation from producing boundary effects

                padkw (dict): kwargs for `numpy.pad`.
                    Defaults to {'mode': 'constant'}.

                dtype (type | None):
                    Cast the loaded data to this type. If unspecified returns the
                    data as-is.

                nodata (int | None, default=None):
                    If specified, for integer data with nodata values, this is
                    passed to kwcoco delayed image finalize. The data is converted
                    to float32 and nodata values are replaced with nan. These
                    nan values are handled correctly in subsequent warping
                    operations.

            with_annots (bool | str, default=True):
                if True, also extracts information about any annotation that
                overlaps the region of interest (subject to visibility_thresh).
                Can also be a List[str] that specifies which specific subinfo
                should be extracted. Valid strings in this list are: boxes,
                keypoints, and segmentation.

            visible_thresh (float): does not return annotations with visibility
                less than this threshold.

            **kwargs : handles deprecated arguments which are now specified
                in the target dictionary itself.

        Returns:
            Dict: sample: dict containing keys
                im (ndarray | DataArray): image / video data
                target (dict): contains the same input items as the input
                    target but additionally specifies inferred information like
                    rel_cx and rel_cy, which gives the center of the target
                    w.r.t the returned **padded** sample.
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
            >>> import ndsampler
            >>> self = ndsampler.CocoSampler.demo()
            >>> # The target (target) lets you specify an arbitrary window
            >>> target = {'gid': 1, 'cx': 5, 'cy': 2, 'width': 6, 'height': 6}
            >>> sample = self.load_sample(target)
            ...
            >>> print('sample.shape = {!r}'.format(sample['im'].shape))
            sample.shape = (6, 6, 3)

        Example:
            >>> # Access direct annotation information
            >>> import ndsampler
            >>> sampler = ndsampler.CocoSampler.demo()
            >>> # Sample a region that contains at least one annotation
            >>> target = {'gid': 1, 'cx': 5, 'cy': 2, 'width': 600, 'height': 600}
            >>> sample = sampler.load_sample(target)
            >>> annotation_ids = sample['annots']['aids']
            >>> aid = annotation_ids[0]
            >>> # Method1: Access ann dict directly via the coco index
            >>> ann = sampler.dset.anns[aid]
            >>> # Method2: Access ann objects via annots method
            >>> dets = sampler.dset.annots(annotation_ids).detections
            >>> print('dets.data = {}'.format(ub.repr2(dets.data, nl=1)))

        Example:
            >>> import ndsampler
            >>> self = ndsampler.CocoSampler.demo()
            >>> target = self.regions.get_positive(0)
            >>> target['window_dims'] = 'square'
            >>> target['pad'] = (25, 25)
            >>> sample = self.load_sample(target)
            >>> print('im.shape = {!r}'.format(sample['im'].shape))
            im.shape = (135, 135, 3)
            >>> target['window_dims'] = None
            >>> target['pad'] = (0, 0)
            >>> sample = self.load_sample(target)
            >>> print('im.shape = {!r}'.format(sample['im'].shape))
            im.shape = (52, 85, 3)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(sample['im'])
            >>> kwplot.show_if_requested()

        Example:
            >>> # sample an out of bounds target
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo('vidshapes8')
            >>> test_vidspace = 1
            >>> target = self.regions.get_positive(0)
            >>> # Toggle to see if this test works in both cases
            >>> space = 'image'
            >>> if test_vidspace:
            >>>     space = 'video'
            >>>     target = target.copy()
            >>>     target['gids'] = [target.pop('gid')]
            >>>     target['scale'] = 1.3
            >>>     #target['scale'] = 0.8
            >>>     #target['use_native_scale'] = True
            >>>     #target['realign_native'] = 'largest'
            >>> target['window_dims'] = (364, 364)
            >>> sample = self.load_sample(target)
            >>> annots = sample['annots']
            >>> assert len(annots['aids']) > 0
            >>> #assert len(annots['rel_cxywh']) == len(annots['aids'])
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> tf_rel_to_abs = sample['params']['tf_rel_to_abs']
            >>> rel_dets = annots['frame_dets'][0]
            >>> abs_dets = rel_dets.warp(tf_rel_to_abs)
            >>> # Draw box in original image context
            >>> #abs_frame = self.frames.load_image(sample['target']['gid'], space=space)[:]
            >>> abs_frame = self.dset.coco_image(sample['target']['gid']).delay(space=space).finalize()
            >>> kwplot.imshow(abs_frame, pnum=(1, 2, 1), fnum=1)
            >>> abs_dets.data['boxes'].translate([-.5, -.5]).draw()
            >>> abs_dets.data['keypoints'].draw(color='green', radius=10)
            >>> abs_dets.data['segmentations'].draw(color='red', alpha=.5)
            >>> # Draw box in relative sample context
            >>> if test_vidspace:
            >>>     kwplot.imshow(sample['im'][0], pnum=(1, 2, 2), fnum=1)
            >>> else:
            >>>     kwplot.imshow(sample['im'], pnum=(1, 2, 2), fnum=1)
            >>> rel_dets.data['boxes'].translate([-.5, -.5]).draw()
            >>> rel_dets.data['segmentations'].draw(color='red', alpha=.6)
            >>> rel_dets.data['keypoints'].draw(color='green', alpha=.4, radius=10)
            >>> kwplot.show_if_requested()

        Example:
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo('photos')
            >>> target = self.regions.get_positive(1)
            >>> target['window_dims'] = (300, 150)
            >>> target['pad'] = None
            >>> sample = self.load_sample(target)
            >>> assert sample['im'].shape[0:2] == target['window_dims']
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(sample['im'], colorspace='rgb')
            >>> kwplot.show_if_requested()

        Example:
            >>> # Multispectral video sample example
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo('vidshapes1-multispectral', num_frames=5)
            >>> sample_grid = self.new_sample_grid('video_detection', (3, 128, 128))
            >>> target = sample_grid['positives'][0]
            >>> target['channels'] = 'B1|B8'
            >>> target['as_xarray'] = False
            >>> sample = self.load_sample(target)
            >>> print(ub.repr2(sample['target'], nl=1))
            >>> print(sample['im'].shape)
            >>> assert sample['im'].shape == (3, 128, 128, 2)
            >>> target['channels'] = '<all>'
            >>> sample = self.load_sample(target)
            >>> assert sample['im'].shape == (3, 128, 128, 5)

        Example:
            >>> # Multispectral-multisensor jagged video sample example
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo('vidshapes1-msi-multisensor', num_frames=5)
            >>> sample_grid = self.new_sample_grid('video_detection', (3, 128, 128))
            >>> target = sample_grid['positives'][0]
            >>> target['channels'] = 'B1|B8'
            >>> target['as_xarray'] = False
            >>> sample1 = self.load_sample(target)
            >>> target['scale'] = 2
            >>> sample2 = self.load_sample(target)
            >>> target['use_native_scale'] = True
            >>> sample3 = self.load_sample(target)
            >>> ####
            >>> assert sample1['im'].shape == (3, 128, 128, 2)
            >>> assert sample2['im'].shape == (3, 256, 256, 2)
            >>> box1 = sample1['annots']['frame_dets'][0].boxes
            >>> box2 = sample2['annots']['frame_dets'][0].boxes
            >>> box3 = sample3['annots']['frame_dets'][0].boxes
            >>> assert np.allclose((box2.width / box1.width), 2)
            >>> # Jagged annotations are still in video space
            >>> assert np.allclose((box3.width / box1.width), 2)
            >>> jagged_shape = [[p.shape for p in f] for f in sample3['im']]
            >>> jagged_align = [[a for a in m['align']] for m in sample3['params']['jagged_meta']]
        """
        if target is None:
            target = kwargs.pop('tr', None)
        if target is None:
            raise ValueError('The target dictionary must be specified')

        target_ = self._infer_target_attributes(target, **kwargs)
        sample = self._load_slice(target_)

        if with_annots or ub.iterable(with_annots):
            self._populate_overlap(sample, visible_thresh, with_annots)

        sample['classes'] = self.classes
        sample['kp_classes'] = self.kp_classes
        return sample

    @profile
    def _infer_target_attributes(self, target, **kwargs):
        """
        Infer unpopulated target attribues

        Example:
            >>> # sample using only an annotation id
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> target = {'aid': 1, 'as_xarray': True}
            >>> target_ = self._infer_target_attributes(target)
            >>> print('target_ = {}'.format(ub.repr2(target_, nl=1)))
            >>> assert target_['gid'] == 1
            >>> assert all(k in target_ for k in ['cx', 'cy', 'width', 'height'])

            >>> self = CocoSampler.demo('vidshapes8-multispectral')
            >>> target = {'aid': 1, 'as_xarray': True}
            >>> target_ = self._infer_target_attributes(target)
            >>> assert target_['gid'] == 1
            >>> assert all(k in target_ for k in ['cx', 'cy', 'width', 'height'])

            >>> target = {'vidid': 1, 'as_xarray': True}
            >>> target_ = self._infer_target_attributes(target)
            >>> print('target_ = {}'.format(ub.repr2(target_, nl=1)))
            >>> assert 'gids' in target_

            >>> target = {'gids': [1, 2], 'as_xarray': True}
            >>> target_ = self._infer_target_attributes(target)
            >>> print('target_ = {}'.format(ub.repr2(target_, nl=1)))
        """
        # we might modify the target
        target_ = target.copy()

        # Handle moving old keyword arguments into the target dictionary
        deprecated_kwargs_defaults = ub.udict(
            pad=None, padkw={'mode': 'constant'}, dtype=None, nodata=None)
        for key, default in deprecated_kwargs_defaults.items():
            if key in kwargs:
                ub.schedule_deprecation(
                    'ndsampler', key, 'keyword arg to load_slice',
                    migration=f'specify {key} as an item in the target dictionary',
                    deprecate='0.7.0', error='1.0.0', remove='1.1.0')
                if key in target_:
                    warnings.warn(
                        f'{key} was specified in both kwargs and the target dictionary, '
                        'the deprecated kwarg will take precedence')
                target_[key] = kwargs[key]

        if 'aid' in target_:
            # If the annotation id is specified, infer other unspecified fields
            aid = target_['aid']
            try:
                ann = self.dset.anns[aid]
            except KeyError:
                pass
            else:
                if 'gid' not in target_:
                    target_['gid'] = ann['image_id']
                if len({'cx', 'cy', 'width', 'height'} & set(target_)) != 4:
                    box = kwimage.Boxes([ann['bbox']], 'xywh')
                    cx, cy, width, height = box.to_cxywh().data[0]
                    if 'cx' not in target_:
                        target_['cx'] = cx
                    if 'cy' not in target_:
                        target_['cy'] = cy
                    if 'width' not in target_:
                        target_['width'] = width
                    if 'height' not in target_:
                        target_['height'] = height
                if 'category_id' not in target_:
                    target_['category_id'] = ann['category_id']

        gid = target_.get('gid', None)
        vidid = target_.get('vidid', None)
        gids = target_.get('gids', None)
        slices = target_.get('slices', None)
        time_slice = target_.get('time_slice', None)
        space_slice = target_.get('space_slice', None)
        window_dims = target_.get('window_dims', None)
        vid_gids = None
        ndim = None

        if vidid is not None or gids is not None:
            # Video sample
            if vidid is None:
                if gids is None:
                    raise ValueError('ambiguous image or video object id(s)')
                _vidids = self.dset.images(gids).lookup('video_id', None)
                if __debug__:
                    if not ub.allsame(_vidids):
                        warnings.warn('sampled gids from different videos')
                vidid = ub.peek(_vidids)
                target_['vidid'] = vidid
            # assert vidid == target_['vidid']
            ndim = 3
        elif gid is not None:
            # Image sample
            ndim = 2
        else:
            raise ValueError('no source object id(s)')

        # Fix non-determined bounds
        if ndim == 2:
            img = self.dset.index.imgs[gid]
            space_dims = (img['height'], img['width'])
            data_dims = space_dims
        elif ndim == 3:
            if vidid is not None:
                video = self.dset.index.videos[vidid]
                space_dims = (video['height'], video['width'])
                vid_gids = self.dset.index.vidid_to_gids[vidid]
                data_dims = (len(vid_gids),) + space_dims
            else:
                space_dims = None
                data_dims = None
        else:
            raise NotImplementedError

        target_['space_dims'] = space_dims
        target_['data_dims'] = data_dims

        # other spatial specifiers allowed if slices is not given
        alternate_keys = {'cx', 'cy', 'height', 'width'}
        has_alternate = bool(set(target_) & alternate_keys)

        if slices is not None:
            if space_slice is None:
                if ndim == 3:
                    space_slice = target_['space_slice'] = slices[1:3]
                elif ndim == 2:
                    space_slice = target_['space_slice'] = slices[0:2]
                else:
                    raise NotImplementedError
            if ndim == 3 and gids is None and time_slice is None:
                time_slice = target_['time_slice'] = slices[0]

        if space_slice is None:
            if has_alternate:
                # A center / width / height was specified
                center = (target_['cy'], target_['cx'])
                # Determine the requested window size
                if window_dims is None:
                    window_dims = 'extent'

                if isinstance(window_dims, str):
                    if window_dims == 'extent':
                        window_dims = (target_['height'], target_['width'])
                        window_dims = np.ceil(np.array(window_dims)).astype(np.int)
                        window_dims = tuple(window_dims.tolist())
                    elif window_dims == 'square':
                        window_dims = (target_['height'], target_['width'])
                        window_dims = np.ceil(np.array(window_dims)).astype(np.int)
                        window_dims = tuple(window_dims.tolist())
                        maxdim = max(window_dims)
                        window_dims = (maxdim, maxdim)
                    else:
                        raise KeyError(window_dims)
                target_['window_dims'] = window_dims
                space_slice = _center_extent_to_slice(center, window_dims)
            else:
                height, width = space_dims
                space_slice = (slice(0, height), slice(0, width))
            target_['space_slice'] = space_slice

        if ndim == 2:
            target_['slices'] = slices = space_slice
        elif ndim == 3:
            if gids is None:
                if time_slice is None:
                    if vid_gids is None:
                        raise ValueError('no gids or ability to infer them')
                    else:
                        gids = target_['gids'] = vid_gids
                else:
                    gids = target_['gids'] = vid_gids[time_slice]
            if time_slice is None:
                time_slice = target_['time_slice'] = slice(0, len(gids))
            target_['slices'] = slices = (time_slice,) + space_slice
        else:
            raise NotImplementedError(ndim)
        return target_

    @profile
    def _load_slice(self, target):
        """
        Example:
            >>> # sample an out of bounds target
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo()
            >>> target = self.regions.get_positive(0)
            >>> target = self._infer_target_attributes(target)
            >>> target['as_xarray'] = True
            >>> sample = self._load_slice(target)
            >>> print('sample = {!r}'.format(ub.map_vals(type, sample)))

            >>> # sample an out of bounds target
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo('vidshapes2')
            >>> target = self._infer_target_attributes({'vidid': 1})
            >>> target = self._infer_target_attributes(target)
            >>> target['as_xarray'] = True
            >>> sample = self._load_slice(target)
            >>> print('sample = {!r}'.format(ub.map_vals(type, sample)))

            >>> target = self._infer_target_attributes({'gids': [1, 2]})
            >>> target['as_xarray'] = True
            >>> sample = self._load_slice(target)
            >>> print('sample = {!r}'.format(ub.map_vals(type, sample)))

        CommandLine:
            xdoctest -m ndsampler.coco_sampler CocoSampler._load_slice --profile

        Ignore:
            from ndsampler.coco_sampler import *  # NOQA
            from ndsampler.coco_sampler import _center_extent_to_slice, _ensure_iterablen
            import ndsampler
            import xdev
            globals().update(xdev.get_func_kwargs(ndsampler.CocoSampler._load_slice))

        Example:
            >>> # Multispectral video sample example
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo('vidshapes1-multispectral', num_frames=5)
            >>> sample_grid = self.new_sample_grid('video_detection', (3, 128, 128))
            >>> target = sample_grid['positives'][0]
            >>> target = self._infer_target_attributes(target)
            >>> target['channels'] = 'B1|B8'
            >>> target['as_xarray'] = False
            >>> sample = self.load_sample(target)
            >>> print(ub.repr2(sample['target'], nl=1))
            >>> print(sample['im'].shape)
            >>> assert sample['im'].shape == (3, 128, 128, 2)
            >>> target['channels'] = '<all>'
            >>> sample = self.load_sample(target)
            >>> assert sample['im'].shape == (3, 128, 128, 5)

        Example:
            >>> # Multispectral video sample example
            >>> from ndsampler.coco_sampler import *
            >>> self = CocoSampler.demo('vidshapes1-multisensor-msi', num_frames=5)
            >>> sample_grid = self.new_sample_grid('video_detection', (3, 128, 128))
            >>> target = sample_grid['positives'][0]
            >>> target = self._infer_target_attributes(target)
            >>> target['channels'] = 'B1|B8'
            >>> target['as_xarray'] = False
            >>> target['space_slice'] = (slice(-64, 64), slice(-64, 64))
            >>> sample = self.load_sample(target)
            >>> print(ub.repr2(sample['target'], nl=1))
            >>> print(sample['im'].shape)
            >>> assert sample['im'].shape == (3, 128, 128, 2)
            >>> target['channels'] = '<all>'
            >>> sample = self.load_sample(target)
            >>> assert sample['im'].shape[2] > 5  # probably 16

            >>> # Test jagged native scale sampling
            >>> target['use_native_scale'] = True
            >>> target['as_xarray'] = True
            >>> target['channels'] = 'B1|B8|r|g|b|disparity|gauss'
            >>> sample = self.load_sample(target)
            >>> jagged_meta = sample['params']['jagged_meta']
            >>> frames = sample['im']
            >>> jagged_shape = [[p.shape for p in f] for f in frames]
            >>> jagged_chans = [[p.coords['c'].values.tolist() for p in f] for f in frames]
            >>> jagged_chans2 = [m['chans'] for m in jagged_meta]
            >>> jagged_align = [[a.concise() for a in m['align']] for m in jagged_meta]
            >>> # all frames should have the same number of channels
            >>> assert len(frames) == 3
            >>> assert all(sum(p.shape[2] for p in f) == 7 for f in frames)
            >>> frames[0] == 3
            >>> print('jagged_chans = {}'.format(ub.repr2(jagged_chans, nl=1)))
            >>> print('jagged_shape = {}'.format(ub.repr2(jagged_shape, nl=1)))
            >>> print('jagged_chans2 = {}'.format(ub.repr2(jagged_chans2, nl=1)))
            >>> print('jagged_align = {}'.format(ub.repr2(jagged_align, nl=1)))

            >>> # Test realigned native scale sampling
            >>> target['use_native_scale'] = True
            >>> target['realign_native'] = 'largest'
            >>> target['as_xarray'] = True
            >>> gid = None
            >>> for coco_img in self.dset.images().coco_images:
            >>>     if coco_img.channels & 'r|g|b':
            >>>         gid = coco_img.img['id']
            >>>         break
            >>> assert gid is not None, 'need specific image'
            >>> target['gids'] = [gid]
            >>> # Test channels that are good early fused groups
            >>> target['channels'] = 'r|g|b'
            >>> sample1 = self.load_sample(target)
            >>> target['channels'] = 'B8|B11'
            >>> sample2 = self.load_sample(target)
            >>> target['channels'] = 'r|g|b|B11'
            >>> sample3 = self.load_sample(target)
            >>> shape1 = sample1['im'].shape[1:3]
            >>> shape2 = sample2['im'].shape[1:3]
            >>> shape3 = sample3['im'].shape[1:3]
            >>> print(f'shape1={shape1}')
            >>> print(f'shape2={shape2}')
            >>> print(f'shape3={shape3}')
            >>> assert shape1 != shape2
            >>> assert shape2 == shape3
        """
        gids = target.get('gids', None)
        if gids is not None:
            sample = self._load_slice_3d(target)
        else:
            sample = self._load_slice_2d(target)

        legacy_target = target.get('legacy_target', None)
        if legacy_target is None:
            legacy_target = True

        if legacy_target:
            ub.schedule_deprecation(
                'ndsampler', 'tr', 'key in the returned sample dictionary',
                migration=ub.paragraph(
                    '''
                    Opt-in to new behavior by specifying legacy_target=False in
                    the target dictionary.
                    '''),
                deprecate='0.7.0', error='1.0.0', remove='1.1.0',
            )
            sample['tr'] = sample['target']
        return sample

    @profile
    def _load_slice_3d(self, target):
        """
        Breakout the 2d vs 3d logic so they can evolve somewhat independently.

        TODO: the 2D logic needs to be updated to be more consistent with 3d
        logic

        Or at least the differences between them are more clear.
        """
        target_ = target
        pad = target_.get('pad', None)
        dtype = target_.get('dtype', None)
        nodata = target_.get('nodata', None)

        assert 'space_slice' in target_
        data_dims = target_['data_dims']
        requested_slice = target_['slices']

        # Resolve any Nones in the requested slice (without the padding)
        # Probably could do this more efficienty
        import kwarray
        if data_dims is None:
            _req_real_slice = requested_slice
            _req_extra_pad = [(0, 0), (0, 0)]
        else:
            _req_real_slice, _req_extra_pad = kwarray.embed_slice(requested_slice, data_dims)
        resolved_slice = tuple([
            slice(s.start - p_lo, s.stop + p_hi, s.step)
            for s, (p_lo, p_hi) in zip(_req_real_slice, _req_extra_pad)
        ])

        channels = target_.get('channels', ub.NoParam)

        if channels is ub.NoParam or isinstance(channels, str) and channels == '<all>':
            # Do something special
            request_chanspec = None
        else:
            request_chanspec = kwcoco.ChannelSpec.coerce(channels)

        # TODO: disable function level nodata param
        interpolation = target_.get('interpolation', 'auto')
        antialias = target_.get('antialias', 'auto')
        if interpolation == 'auto':
            interpolation = 'linear'
        if antialias == 'auto':
            antialias = interpolation not in {'nearest'}

        # Special arg that will let us use "native resolution" or as
        # close to it.
        extra_scale = target_.get('scale', None)  # extra scaling
        use_native_scale = target_.get('use_native_scale', False)
        realign_native = target_.get('realign_native', False)

        # vidid = target_.get('vidid', None)
        # if vidid is None:
        #     raise AssertionError

        ndim = 3  # number of space-time dimensions (ignore channel)

        # padkw = target_.get('padkw', {'mode': 'constant'})  # TODO
        pad_slice = _coerce_pad(pad, ndim)

        data_time_dim = None if data_dims is None else data_dims[0]
        # data_space_dims = data_dims[1:]
        requested_time_slice = resolved_slice[0]
        requested_space_slice = resolved_slice[1:]
        space_pad = pad_slice[1:]
        time_pad = pad_slice[0]

        # if time_pad.start is None:

        if time_pad[0] != 0 or time_pad[1] != 0 or requested_time_slice.start < 0 or (data_time_dim is not None and requested_time_slice.stop > data_time_dim):
            raise NotImplementedError(ub.paragraph(
                '''
                padding in time is not yet supported, but is an easy TODO
                '''))

        # TODO: we may want to build a efficient temporal sampling
        # data structure when there are a lot of timesteps

        # As of kwcoco 0.2.1 gids are ordered by frame index
        # TODO: frames should have better nd-support, hack it for now to
        # just load the 2d data for each image
        time_gids = target_['gids']
        space_frames = []
        jagged_meta = []

        as_xarray = target_.get('as_xarray', False)

        space = 'video'
        # space = 'asset'

        coco_img_list = self.dset.images(time_gids).coco_images
        if request_chanspec is None:
            # Hobble together channels if they aren't given
            unique_chan = list(ub.unique([g.channels for g in coco_img_list], key=lambda c: c.spec))
            if len(unique_chan):
                request_chanspec = kwcoco.FusedChannelSpec.coerce(sorted(sum(unique_chan).fuse().unique()))

        # New: for each frame, we need to keep track of the transform from the
        # grid space to sample space, possibly for each channel
        frame_sample_from_grid_warps = []

        # Start off by making the transform assuming no special scaling.
        request_space_box = kwimage.Boxes.from_slice(
            requested_space_slice, clip=False, wrap=False).to_ltrb()
        hpad, wpad = space_pad
        final_space_box = request_space_box.pad(
            x_left=wpad[0], y_top=hpad[0], x_right=wpad[1], y_bot=hpad[1])
        sample_tlbr = final_space_box
        st_dims = [(s.start, s.stop) for s in final_space_box.to_slices()[0]]
        x_start = sample_tlbr.tl_x.ravel()[0]
        y_start = sample_tlbr.tl_y.ravel()[0]
        offset = np.array([-x_start, -y_start])
        tf_abs_from_rel = kwimage.Affine.affine(offset=-offset)
        tf_rel_from_abs = tf_abs_from_rel.inv()

        for time_idx, coco_img in enumerate(coco_img_list):

            # Build up the transform from the grid to the final sample
            # TODO: We should be able to use delayed image to do this.
            # It may require a reference image, or reference delayed operation
            # points (which could be a load). For now just do it here.

            warp_sample_from_grid = tf_rel_from_abs

            delayed_frame = coco_img.delay(
                channels=request_chanspec, space=space,
                interpolation=interpolation, nodata_method=nodata,
                antialias=antialias, mode=1
            )

            delayed_crop = delayed_frame.crop(requested_space_slice,
                                              clip=False, wrap=False,
                                              pad=space_pad)

            delayed_crop = delayed_crop.prepare()
            delayed_crop = delayed_crop.optimize()

            frame_use_native_scale = use_native_scale
            if frame_use_native_scale:
                undone_parts, jagged_align = delayed_crop.undo_warps(
                    remove=['scale'], squash_nans=True,
                    return_warps=True)

                if realign_native:
                    # User requested to realign all parts back up to a
                    # comparable scale.
                    if realign_native == 'largest':
                        old_dsize = delayed_crop.dsize

                        dsizes = []
                        for part in undone_parts:
                            dsizes.append(part.dsize)
                        max_dsize = max(dsizes, key=lambda t: t[0] * t[1])
                        rescaled_parts = []
                        for part in undone_parts:
                            rescale_factor = np.array(max_dsize) / np.array(part.dsize)
                            rescaled = part.warp({'scale': rescale_factor}, dsize=max_dsize)
                            rescaled_parts.append(rescaled)
                        from kwcoco.util.delayed_ops.delayed_nodes import DelayedChannelConcat
                        delayed_crop = DelayedChannelConcat(rescaled_parts, dsize=max_dsize)
                        delayed_crop = delayed_crop.optimize()

                        # Find the scale factor to add into the sample_from_grid
                        # transform
                        # TODO: can get a finer-grained scale factor here if we
                        # avoid dsize divides and instead track the scale
                        # factors used in the delayed tree. For now just
                        # get something working.
                        new_dsize = delayed_crop.dsize
                        realign_scale = np.array(new_dsize) / np.array(old_dsize)
                        warp_sample_from_grid = kwimage.Affine.scale(realign_scale) @ warp_sample_from_grid
                    else:
                        raise NotImplementedError
                    # disable this for the rest of the frame
                    frame_use_native_scale = False

            if frame_use_native_scale:
                # print(undone_parts)
                jagged_parts = []
                jagged_chans = []
                jagged_warps = []
                for part in undone_parts:
                    if extra_scale is not None:
                        part = part.warp({'scale': extra_scale})
                        warp_grid_to_part = kwimage.Affine.scale(extra_scale) @ warp_sample_from_grid
                    else:
                        warp_grid_to_part = warp_sample_from_grid
                    if as_xarray:
                        frame = part.optimize().as_xarray().finalize()
                    else:
                        frame = part.optimize().finalize()
                    jagged_parts.append(frame)
                    jagged_chans.append(part.channels)
                    jagged_warps.append(warp_grid_to_part)
                space_frames.append(jagged_parts)
                jagged_meta.append({
                    'align': jagged_align,
                    'chans': jagged_chans,
                    'grid_to_part_warps': jagged_warps,
                })
            else:
                if extra_scale is not None:
                    tf_scale = kwimage.Affine.scale(extra_scale)
                    delayed_crop = delayed_crop.warp({'scale': extra_scale})
                    warp_sample_from_grid = tf_scale @ warp_sample_from_grid
                    # warp_sample_from_grid =  warp_sample_from_grid @ tf_scale
                if as_xarray:
                    frame = delayed_crop.as_xarray().finalize()
                else:
                    delayed_crop = delayed_crop.optimize()
                    frame = delayed_crop.finalize()
                if dtype is not None:
                    frame = frame.astype(dtype)
                space_frames.append(frame)
                frame_sample_from_grid_warps.append(warp_sample_from_grid)

        # Concat aligned frames together (add nans for non-existing
        # channels)
        if frame_use_native_scale:
            data_sliced = space_frames
        else:
            if as_xarray:
                import xarray as xr
                data_sliced = xr.concat(space_frames, dim='t')
            else:
                data_sliced = np.stack(space_frames, axis=0)

        # TODO: gids should be padded if it goes oob.
        # target_['_data_gids'] = time_gids

        if frame_sample_from_grid_warps:
            # only works when there is one image
            tf_abs_from_rel = frame_sample_from_grid_warps[0].inv()

        sample = {
            'im': data_sliced,
            'target': target_,
            'params': {
                # TODO: some of these params need to be deprecated or have
                # their behavior changed to respect the "window/grid space" and
                # "input/sample space".
                'offset': offset,
                'tf_rel_to_abs': tf_abs_from_rel.matrix,  # doesnt make sense here
                'sample_tlbr': sample_tlbr,
                'st_dims': st_dims,
                'data_dims': data_dims,
                'pad': pad,
                'request_chanspec': request_chanspec,
                'frame_sample_from_grid_warps': frame_sample_from_grid_warps,
            },
        }

        if use_native_scale:
            sample['params']['jagged_meta'] = jagged_meta
        return sample

    @profile
    def _load_slice_2d(self, target):
        """
        Breakout the 2d vs 3d logic so they can evolve somewhat independently.

        TODO: the 2D logic needs to be updated to be more consistent with 3d
        logic

        Or at least the differences between them are more clear.
        """
        import skimage
        import kwarray
        target_ = target
        pad = target_.get('pad', None)
        padkw = target_.get('padkw', {'mode': 'constant'})
        nodata = target_.get('nodata', None)

        if pad is None:
            pad = 0

        assert 'space_slice' in target_
        data_dims = target_['data_dims']

        requested_slice = target_['slices']
        channels = target_.get('channels', ub.NoParam)

        # TODO: disable function level nodata param
        nodata = target_.get('nodata', nodata)
        interpolation = target_.get('interpolation', 'auto')
        antialias = target_.get('antialias', 'auto')
        if interpolation == 'auto':
            interpolation = 'linear'
        if antialias == 'auto':
            antialias = interpolation not in {'nearest'}

        # Special arg that will let us use "native resolution" or as
        # close to it.
        scale = target_.get('scale', None)  # extra scaling
        use_native_scale = target_.get('use_native_scale', False)
        assert not use_native_scale

        vidid = target_.get('vidid', None)
        if vidid is not None:
            raise Exception

        gid = target_['gid']
        ndim = 2  # number of space-time dimensions (ignore channel)
        pad = tuple(_ensure_iterablen(pad, ndim))
        data_slice, extra_padding = kwarray.embed_slice(
            requested_slice, data_dims, pad)

        if scale is not None:
            raise NotImplementedError

        # TODO: ensure we are using the kwcoco mechanisms here
        data_clipped = self.frames.load_region(
            image_id=gid, region=data_slice, channels=channels)

        if target_.get('as_xarray', False):
            import xarray as xr
            # TODO: respect the channels arg in target_
            if len(data_clipped.shape) == 1:
                num_bands = 1
            else:
                num_bands = data_clipped.shape[2]

            xrkw = {}
            if num_bands == 1:
                xrkw['c'] = ['gray']
            elif num_bands == 3:
                xrkw['c'] = ['r', 'g', 'b']

            # hack to respect xarray
            data_clipped = xr.DataArray(
                data_clipped, dims=('y', 'x', 'c'), coords=xrkw)

        #####
        # TODO: the 2D logic needs to be updated to be more consistent with
        # 3d logic

        # Apply the padding
        if sum(map(sum, extra_padding)) == 0:
            # No padding was requested
            data_sliced = data_clipped
        else:
            trailing_dims = len(data_clipped.shape) - len(extra_padding)
            if trailing_dims > 0:
                extra_padding = extra_padding + ([(0, 0)] * trailing_dims)
            if target_.get('as_xarray', False):
                coord_pad = dict(zip(data_clipped.dims, extra_padding))
                # print('data_clipped.dims = {!r}'.format(data_clipped.dims))
                if 'constant_values' not in padkw:
                    if nodata in {'float', 'auto'}:
                        padkw['constant_values'] = np.nan
                    else:
                        # hack for consistency
                        padkw['constant_values'] = 0
                data_sliced = data_clipped.pad(coord_pad, **padkw)
            else:
                # print('data_clipped.dims = {!r}'.format(data_clipped.dims))
                if 'constant_values' not in padkw:
                    if nodata in {'float', 'auto'}:
                        padkw['constant_values'] = np.nan
                    else:
                        # hack for consistency
                        padkw['constant_values'] = 0
                # TODO: if the data out of kwcoco is masked, mask the padded
                # value.
                data_sliced = np.pad(data_clipped, extra_padding, **padkw)

        st_dims = [(sl.start - pad_[0], sl.stop + pad_[1])
                   for sl, pad_ in zip(data_slice, extra_padding)]
        (y_start, y_stop), (x_start, x_stop) = st_dims[-2:]
        sample_tlbr = kwimage.Boxes([x_start, y_start, x_stop, y_stop], 'ltrb')
        offset = np.array([-x_start, -y_start])
        tf_rel_to_abs = skimage.transform.AffineTransform(
            translation=-offset
        ).params

        sample = {
            'im': data_sliced,
            'target': target_,
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
            >>> import ndsampler
            >>> self = ndsampler.CocoSampler.demo()
            >>> target = self.regions.get_item(0)
            >>> target = self._infer_target_attributes(target)
            >>> sample = self._load_slice(target)
            >>> sample = self._populate_overlap(sample)
            >>> print('sample = {}'.format(ub.repr2(ub.util_dict.dict_diff(sample, ['im']), nl=-1)))
        """

        if with_annots is True:
            with_annots = ['segmentation', 'keypoints', 'boxes']
        elif isinstance(with_annots, str):
            with_annots = with_annots.split('+')

        if __debug__:
            for k in with_annots:
                assert k in ['segmentation', 'keypoints', 'boxes'], 'k={!r}'.format(k)

        target = sample['target']

        if 'gids' in target:
            gids = target['gids']
        else:
            gids = [target['gid']]

        params = sample['params']

        # The sample box is in "absolute sample space",
        # which is either video or image space.
        sample_box: kwimage.Boxes = params['sample_tlbr']
        offset = params['offset']
        data_dims = params['data_dims']
        space_dims = data_dims[-2:]

        coco_dset = self.dset
        kp_classes = self.kp_classes
        classes = self.classes

        # accumulate information over all frames
        frame_dets = []
        scale = sample['target'].get('scale', None)  # extra scaling

        frame_sample_from_grid_warps = params.get('frame_sample_from_grid_warps', [])

        for rel_frame_idx, gid in enumerate(gids):

            # Check to see if there is a transform between the image-space and
            # the sampling-space (currently this can only be done by a video,
            # but in the future the user might be able to adjust sample scale)
            if target.get('vidid', None) is not None:
                in_video_space = True
                # hack to align annots from image space to video space
                coco_img = coco_dset.coco_image(gid)
                tf_abs_from_img = coco_img.warp_vid_from_img
                tf_img_from_abs = tf_abs_from_img.inv()
            else:
                in_video_space = False
                tf_abs_from_img = None

            # check overlap in image space
            if tf_abs_from_img is None:
                imgspace_sample_box = sample_box
            else:
                imgspace_sample_box = sample_box.warp(tf_img_from_abs.matrix).quantize()

            # Find which bounding boxes are visible in this region
            overlap_aids = self.regions.overlapping_aids(
                gid, imgspace_sample_box, visible_thresh=visible_thresh)

            # Get info about all annotations inside this window
            overlap_anns = [coco_dset.anns[aid] for aid in overlap_aids]
            overlap_cids = [ann['category_id'] for ann in overlap_anns]
            abs_boxes = kwimage.Boxes(
                [ann['bbox'] for ann in overlap_anns], 'xywh')

            # Handle segmentations and keypoints if they exist
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
                        abs_sseg = kwimage.MultiPolygon.coerce(coco_sseg, dims=space_dims)
                sseg_list.append(abs_sseg)
                kpts_list.append(abs_points)

            abs_ssegs = kwimage.PolygonList(sseg_list)
            abs_kpts = kwimage.PointsList(kpts_list)
            abs_kpts.meta['classes'] = self.kp_classes

            # Construct a detections object containing absolute annotation
            # positions
            imgspace_dets = kwimage.Detections(
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
            if scale is None:
                scale = 1.0

            if in_video_space:
                # hack to align annots from image space to video space
                tf_vid_from_img = tf_abs_from_img
                if frame_sample_from_grid_warps:
                    tf_rel_from_vid = frame_sample_from_grid_warps[rel_frame_idx]
                else:
                    # FIXME: the only case where frame_sample_from_grid_warps
                    # would be zero is the jagged case, in which case we should
                    # probably align these to something different, right now it
                    # just aligns it to the scaled window (but does not account
                    # for jagged scaling)
                    tf_rel_from_vid = kwimage.Affine.affine(scale=scale) @ kwimage.Affine.affine(offset=offset)
                tf_rel_from_img = tf_rel_from_vid @ tf_vid_from_img
                rel_dets = imgspace_dets.warp(tf_rel_from_img.matrix)
            else:
                rel_dets = imgspace_dets.translate(offset)
                if scale != 1.0:
                    rel_dets = rel_dets.scale(scale)

            frame_dets.append(rel_dets)

        annots = {}

        legacy_annots = target.get('legacy_annots', None)
        if legacy_annots is None:
            legacy_annots = True

        if legacy_annots:
            ub.schedule_deprecation(
                'ndsampler', 'non-frame_dets', 'return information in sample["annots"]',
                migration=ub.paragraph(
                    '''
                    Opt-in to new behavior by specifying legacy_annots=False in
                    the target dictionary.
                    '''),
                deprecate='0.7.0', error='1.0.0', remove='1.1.0',
            )

            annots.update({
                ###
                # TODO: deprecate everything except for frame dets
                'aids': np.hstack([x.data['aids'] for x in frame_dets]),
                'cids': np.hstack([x.data['cids'] for x in frame_dets]),
                'rel_frame_index': np.hstack([[x.meta['rel_frame_index']] * len(x) for x in frame_dets]),
                'rel_boxes': kwimage.Boxes.concatenate([x.data['boxes'] for x in frame_dets]),
                'rel_ssegs': kwimage.PolygonList(list(ub.flatten([x.data['segmentations'].data for x in frame_dets]))),
                'rel_kpts': kwimage.PointsList(list(ub.flatten([x.data['keypoints'].data for x in frame_dets]))),
                ###
            })
            annots['rel_kpts'].meta['classes'] = self.kp_classes

            main_aid = target.get('aid', None)
            if main_aid is not None:
                # Determine which (if any) index in "annots" corresponds to the
                # main aid (if we even have a main aid)
                cand_idxs = np.where(annots['aids'] == main_aid)[0]
                if len(cand_idxs) == 0:
                    target['annot_idx'] = -1
                elif len(cand_idxs) == 1:
                    target['annot_idx'] = cand_idxs[0]
                else:
                    raise AssertionError('impossible state: len(cand_idxs)={}'.format(len(cand_idxs)))
            else:
                target['annot_idx'] = -1

        annots.update({
            'frame_dets': frame_dets,
        })

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


def _coerce_pad(pad, ndims):
    if pad is None:
        pad_slice = [(0, 0)] * ndims
    elif isinstance(pad, int):
        pad_slice = [(pad, pad)] * ndims
    else:
        # Normalize to left/right pad value for each dim
        pad_slice = [p if ub.iterable(p) else [p, p] for p in pad]

    if len(pad_slice) != ndims:
        # We could "fix" it, but the user probably made a mistake
        # n_trailing = ndims - len(pad)
        # if n_trailing > 0:
        #     pad = list(pad) + [(0, 0)] * n_trailing
        raise ValueError('pad and data_dims must have the same length')
    return pad_slice


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m ndsampler.coco_sampler
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
