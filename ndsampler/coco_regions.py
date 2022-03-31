"""
Maintains information about groundtruth targets. Positives are specified
explicitly, and negatives are mined.

The Targets class maintains the "Positive Population".

A Positive is a bounding box that belongs to an image or video with a class
label and potentially other attributes. Negatives are similar except they are
boxes that do not significantly intersect positives. A pool of positives can
also be selected from the population such that only a subset of data is used
per epoch.


Cases to Handle:
    - [ ] Annotations are significantly smaller than images
        * Annotations are typically very far apart
        * Annotations can be clustered tightly together
        * Annotations are at massively different scales
    - [ ] Annotations are about the same size as the images

"""
import itertools as it
import kwarray
import kwimage
import numpy as np
import ubelt as ub
from ndsampler.utils import util_misc
from ndsampler import isect_indexer


try:
    from xdev import profile
except Exception:
    profile = ub.identity


class MissingNegativePool(AssertionError):
    pass


class Targets(object):
    """
    Abstract API
    """

    def get_negative(self, index=None, rng=None):
        pass

    def get_positive(self, index=None, rng=None):
        pass

    def overlapping_aids(self, gid, box):
        raise NotImplementedError

    def preselect(self, n_pos=None, n_neg=None, neg_to_pos_ratio=None,
                  window_dims=None, rng=None, verbose=0):
        """
        Shuffle selection of positive and negative samples

        TODO:
            [X] Basic, window around positive annotation algorithm
            [ ] Sliding window algorithm from bioharn
        """
        if neg_to_pos_ratio is not None and n_neg is not None:
            raise ValueError('Cannot specify both neg_to_pos_ratio and n_neg')

        n_pos = self._preselect_positives(num=n_neg, rng=rng,
                                          window_dims=window_dims,
                                          verbose=verbose)

        if n_neg is None:
            if neg_to_pos_ratio is None:
                neg_to_pos_ratio = 0
            n_neg = int(neg_to_pos_ratio * n_pos)

        n_neg = self._preselect_negatives(
            num=n_neg, window_dims=window_dims, rng=rng, verbose=verbose)
        return n_pos, n_neg


class CocoRegions(Targets, util_misc.HashIdentifiable, ub.NiceRepr):
    """
    Converts Coco-Style datasets into a table for efficient on-line work

    Perhaps rename this class to regions, and then have targets be
    an attribute of regions.

    Args:
        dset (ndsampler.CocoAPI): a dataset in coco format
        workdir (PathLike): a temporary directory where we can cache stuff
        verbose (int): verbosity level

    Example:
        >>> from ndsampler.coco_regions import *
        >>> self = CocoRegions.demo()
        >>> pos_tr = self.get_positive(rng=0)
        >>> neg_tr = self.get_negative(rng=0)
        >>> print(ub.repr2(pos_tr, precision=2))
        >>> print(ub.repr2(neg_tr, precision=2))
    """

    def __init__(self, dset, workdir=None, verbose=1):
        super(CocoRegions, self).__init__()
        self.dset = dset
        self.workdir = workdir
        self.verbose = verbose  # caching verbosity

        # Lazilly computed values
        self._targets = None
        self._isect_index = None
        self._neg_anchors = None

        self._positive_pool = None

        # A list of the selected positive (possibily oversampled) indices
        self._pos_select_idxs = None

        self._negative_pool = None
        self._negative_idx = None

        self.classes = self.dset.object_categories()
        # currently hacked in
        self.BACKGROUND_CLASS_ID = self.classes.node_to_id.get('background', 0)

        # A dictionary specifying which on-disk caches are enabled.  By default
        # all are enabled, but the user can turn these off if they are causing
        # issues.
        self._enabled_caches = {
            'isect_index': True,
            'targets': True,
            'neg_anchors': True,
            '_negative_pool': True,
            '_pos_select_idxs': True,
        }

    @property
    def catgraph(self):
        # Old alias for classes
        return self.classes

    @property
    def n_negatives(self):
        return len(self._negative_pool) if self._negative_pool is not None else 0

    @property
    def n_positives(self):
        return len(self._pos_select_idxs) if self._pos_select_idxs is not None else len(self.targets)
        # return len(self._positive_pool) if self._positive_pool is not None else len(self.targets)

    @property
    def n_samples(self):
        return self.n_positives + self.n_negatives

    @property
    def class_ids(self):
        return list(self.dset.cats.keys())

    @property
    def image_ids(self):
        return sorted(self.dset.imgs.keys())

    @property
    def n_annots(self):
        return len(self.dset.anns)

    @property
    def n_images(self):
        return len(self.dset.imgs)

    @property
    def n_categories(self):
        return len(self.dset.cats)

    def lookup_class_name(self, class_id):
        return self.dset.cats[class_id]['name']

    def lookup_class_id(self, class_name):
        return self.dset.name_to_cat[class_name]['id']

    def __nice__(self):
        n_targets = len(self._targets) if self._targets is not None else None
        n_pos = len(self._pos_select_idxs) if self._pos_select_idxs is not None else None
        n_neg = len(self._negative_pool) if self._negative_pool is not None else None
        n_dset = (ub.repr2(self.dset.basic_stats(), nl=0, sep='', si=True,
                           explicit=1, nobr=1) if self.dset is not None else None)
        nice = ('{{nPos={pos}, nNeg={neg}}} ∼ '
                '{{nViable={targets}}} ⊆ '
                '{{{dset}}}').format(
                    dset=n_dset, targets=n_targets, pos=n_pos, neg=n_neg)
        return nice

    @classmethod
    def demo(CocoRegions):
        import kwcoco
        dset = kwcoco.CocoDataset.demo('shapes8')
        dset._ensure_imgsize()
        self = CocoRegions(dset)
        return self

    def _make_hashid(self):
        _hashid = getattr(self.dset, 'hashid', None)
        if _hashid is None:
            if self.verbose > 1:
                print('Constructing regions hashid (ignoring pixel data)')
            self.dset._build_hashid(hash_pixels=False)
            _hashid = getattr(self.dset, 'hashid', None)
        return _hashid, None

    @property
    def isect_index(self):
        """
        Lazy access to a disk-cached intersection index for this dataset
        """
        return self._lazy_isect_index()

    def _lazy_isect_index(self, verbose=None):
        if self._isect_index is None:
            # FIXME! Any use of cacher here should be wrapped in an
            # InterProcessLock! The sampler often exists in a multiprocessing
            # context!
            cacher = self._cacher('isect_index', verbose=verbose)
            _isect_index = cacher.tryload(on_error='clear')
            if _isect_index is None:
                _isect_index = isect_indexer.FrameIntersectionIndex.from_coco(self.dset)
                cacher.save(_isect_index)
            self._isect_index = _isect_index
        return self._isect_index

    @property
    def targets(self):
        """
        All viable positive annotations targets in a flat table.

        The main idea is that this is the population of all positives that we
        could sample from. Often times we will simply use all of them.

        This function takes a subset of annotations in the coco dataset that
        can be considered "viable" positives. We may subsample these futher,
        but this serves to collect the annotations that could feasibly be used
        by the network. Essentailly we remove annotations without bounding
        boxes. I'm not sure I 100% like the way this works though. Shouldn't
        filtering be done before we even get here? Perhaps but perhaps not.
        This design needs a bit more thought.
        """
        if self._targets is None:
            if self.verbose:
                print('Building targets from coco dataset')
            cacher = self._cacher('targets')
            _targets = cacher.tryload(on_error='clear')
            if _targets is None:
                _targets = tabular_coco_targets(self.dset)
                cacher.save(_targets)
            self._targets = _targets
        return self._targets

    @property
    def neg_anchors(self):
        if self._neg_anchors is None:
            cacher = self._cacher('neg_anchors')
            neg_anchors = cacher.tryload(on_error='clear')
            if neg_anchors is None:
                if False:
                    # FIXME: this is not the right way to normalize anchors
                    neg_anchors = np.vstack([
                        self.targets['width'] / self.targets['img_width'],
                        self.targets['height'] / self.targets['img_height']
                    ]).T
                else:
                    z = np.minimum(self.targets['img_width'],
                                   self.targets['img_height'])
                    neg_anchors = np.vstack([self.targets['width'],
                                             self.targets['height']]).T
                    neg_anchors = neg_anchors / z[:, None]

                rng = kwarray.ensure_rng(0)
                rng.shuffle(neg_anchors)
                neg_anchors = neg_anchors[0:1000]

                if False:
                    # TODO: perhaps use k-means to cluster
                    import sklearn
                    import sklearn.cluster
                    n_clusters = len(self.dset.dataset['categories']) * 3
                    algo = sklearn.cluster.KMeans(
                        n_clusters=n_clusters, n_init=20, max_iter=10000,
                        tol=1e-6, algorithm='elkan', verbose=0)
                    algo.fit(neg_anchors)
                    neg_anchors = algo.cluster_centers_

                cacher.save(neg_anchors)

            self._neg_anchors = neg_anchors
        return self._neg_anchors

    @profile
    def overlapping_aids(self, gid, region, visible_thresh=0.0):
        """
        Finds the other annotations in this image that overlap a region

        Args:
            gid (int): image id
            region (kwimage.Boxes): bounding box
            visible_thresh (float): does not return annotations with visibility
                less than this threshold.

        Returns:
            List[int]: annotation ids
        """
        overlap_aids = self.isect_index.overlapping_aids(gid, region)
        if visible_thresh > 0 and len(overlap_aids) > 0:
            # Get info about all annotations inside this window
            if 0:
                overlap_annots = self.dset.annots(overlap_aids)
                abs_boxes = overlap_annots.boxes
            else:
                overlap_anns = [self.dset.anns[aid] for aid in overlap_aids]
                abs_boxes = kwimage.Boxes(
                    [ann['bbox'] for ann in overlap_anns], 'xywh')
            # Remove annotations that are not mostly invisible
            if len(abs_boxes) > 0:
                eps = 1e-6
                isect_area = region[None, :].isect_area(abs_boxes)[0]
                other_area = abs_boxes.area.T[0]
                visibility = isect_area / (other_area + eps)
                is_visible = visibility > visible_thresh
                abs_boxes = abs_boxes[is_visible]
                overlap_aids = list(it.compress(overlap_aids, is_visible))
                # overlap_annots = self.dset.annots(overlap_aids)
        return overlap_aids

    def get_segmentations(self, aids):
        """
        Returns the segmentations corresponding to a set of annotation ids

        Example:
            >>> from ndsampler.coco_regions import *
            >>> from ndsampler import coco_sampler
            >>> self = coco_sampler.CocoSampler.demo().regions
            >>> aids = [1, 2]
        """
        sseg_list = []
        for aid in aids:
            ann = self.dset.anns[aid]
            coco_sseg = ann.get('segmentation', None)
            if coco_sseg is None:
                sseg = None
            else:
                sseg = kwimage.MultiPolygon.coerce(coco_sseg)
            sseg_list.append(sseg)
        return sseg_list

    def get_negative(self, index=None, rng=None):
        """
        Get localization information for a negative region

        Args:
            index (int or None): indexes into the current negative pool
                or if None returns a random negative

            rng (RandomState): used only if index is None

        Returns:
            Dict: tr: target info dictionary

        CommandLine:
            xdoctest -m ndsampler.coco_regions CocoRegions.get_negative

        Example:
            >>> from ndsampler.coco_regions import *
            >>> from ndsampler import coco_sampler
            >>> rng = kwarray.ensure_rng(0)
            >>> self = coco_sampler.CocoSampler.demo().regions
            >>> tr = self.get_negative(rng=rng)
            >>> # xdoctest: +IGNORE_WANT
            >>> assert 'category_id' in tr
            >>> assert 'aid' in tr
            >>> assert 'cx' in tr
            >>> print(ub.repr2(tr, precision=2))
            {
                'aid': -1,
                'category_id': 0,
                'cx': 190.71,
                'cy': 95.83,
                'gid': 1,
                'height': 140.00,
                'img_height': 600,
                'img_width': 600,
                'width': 68.00,
            }
        """

        if index is None:
            # Random negative
            if self._negative_pool is None:
                self._preselect_negatives(1, rng=rng)

            # sample sequentially from the presampled pool
            tr = self._negative_pool.iloc[self._negative_idx]
            self._negative_idx += 1

            # resample the pool once we reach the end
            if self._negative_idx == len(self._negative_pool):
                self._negative_pool = None
        else:
            # the negative pool must have been initialized
            if self._negative_pool is None:
                raise MissingNegativePool((
                    'A presampled negative pool does not exist for this '
                    'target, but it is required to access the specific '
                    'negative index={}. Did you forget to call '
                    '_preselect_negatives?'
                ).format(index))
            tr = self._negative_pool.iloc[index]
        return tr

    def get_positive(self, index=None, rng=None):
        """
        Get localization information for a positive region

        Args:
            index (int or None): indexes into the current positive pool
                or if None returns a random negative

            rng (RandomState): used only if index is None

        Returns:
            Dict: tr: target info dictionary

        Example:
            >>> from ndsampler import coco_sampler
            >>> rng = kwarray.ensure_rng(0)
            >>> self = coco_sampler.CocoSampler.demo().regions
            >>> tr = self.get_positive(0, rng=rng)
            >>> print(ub.repr2(tr, precision=2))
        """
        if index is None:
            rng = kwarray.ensure_rng(rng)
            index = rng.randint(0, self.n_positives)

        if self._pos_select_idxs is not None:
            index = self._pos_select_idxs[index]

        tr = self.targets.iloc[index]
        return tr

    def get_item(self, index, rng=None):
        """
        Loads from positives and then negatives.
        """
        if index is None:
            rng = kwarray.ensure_rng(rng)
            index = rng.randint(0, self.n_samples)

        if index < self.n_positives:
            sample = self.get_positive(index, rng=rng)
        else:
            index = index - self.n_positives
            sample = self.get_negative(index, rng=rng)
        return sample

    def _random_negatives(self, num, exact=False, neg_anchors=None,
                          window_size=None, rng=None, thresh=0.0):
        """
        Samples multiple negatives at once for efficiency

        Args:
            num (int): number of negatives to sample

            exact (bool): if True, we will try to find exactly `num` negatives,
                otherwise the number returned is approximate.

            neg_anchors (): prior normalized aspect ratios for negative boxes.
                Mutually exclusive with `window_size`.

            window_size (Tuple): absolute box size (width, height)
                used to sample negative regions. If not specified the relative
                anchor strategy will be used to randomly choose potentially
                non-square regions relative to the image size.

            thresh (float): overlap area threshold as a percentage of the
                negative box size. When thresh=0.0, that means negatives cannot
                overlap any positive, when threh=1.0, there are no constrains
                on negative placement.

        Returns:
            DataFrameArray: targets - contains negative target information

        Example:
            >>> from ndsampler.coco_regions import *
            >>> from ndsampler import coco_sampler
            >>> self = coco_sampler.CocoSampler.demo().regions
            >>> num = 100
            >>> rng = kwarray.ensure_rng(0)
            >>> targets = self._random_negatives(num, rng=rng)
            >>> assert len(targets) <= num
            >>> targets = self._random_negatives(num, exact=True)
            >>> assert len(targets) == num
        """
        rng = kwarray.ensure_rng(rng)

        # Choose some number of centers in each dimension
        if neg_anchors is None and window_size is None:
            neg_anchors = self.neg_anchors

        gids, boxes = self.isect_index.random_negatives(
            num, anchors=neg_anchors, window_size=window_size, exact=exact,
            rng=rng, thresh=thresh)

        targets = kwarray.DataFrameArray()
        targets = kwarray.DataFrameArray(columns=['gid', 'aid', 'cx', 'cy',
                                                  'width', 'height',
                                                  'img_width', 'img_height'])
        targets['gid'] = gids
        targets['aid'] = [-1] * len(gids)
        targets['category_id'] = [self.BACKGROUND_CLASS_ID] * len(gids)
        if len(boxes) > 0:
            cxywh = boxes.to_cxywh().data
            targets['cx'] = cxywh.T[0]
            targets['cy'] = cxywh.T[1]
            targets['width'] = cxywh.T[2]
            targets['height'] = cxywh.T[3]

        if 0:
            targets['img_width'] = self.dset.images(gids).width
            targets['img_height'] = self.dset.images(gids).height
        else:
            imgs = [self.dset.imgs[gid] for gid in gids]
            targets['img_width'] = [img['width'] for img in imgs]
            targets['img_height'] = [img['height'] for img in imgs]
        return targets

    def new_sample_grid(self, task, window_dims, window_overlap=0, **kwargs):
        """
        New experimental method to replace preselect positives / negatives

        Args:
            task (str): can be
                video_detection
                image_detection
                # video_classification
                # image_classification

            **kwargs :
                passed to `new_video_sample_grid` or `new_image_sample_grid`

        Example:
            >>> from ndsampler.coco_regions import *
            >>> from ndsampler import coco_sampler
            >>> self = coco_sampler.CocoSampler.demo('vidshapes1').regions
            >>> self.dset.conform()
            >>> sample_grid = self.new_sample_grid('video_detection', window_dims=(2, 100, 100))
        """
        dset = self.dset
        if task == 'video_detection':
            sample_grid = new_video_sample_grid(dset, window_dims,
                                                window_overlap, **kwargs)
        elif task == 'image_detection':
            sample_grid = new_image_sample_grid(dset, window_dims,
                                                window_overlap, **kwargs)
        else:
            raise NotImplementedError(task)

        return sample_grid

    def _preselect_positives(self, num=None, window_dims=None, rng=None,
                             verbose=None):
        """"
        preload a bunch of positives

        Example:
            >>> from ndsampler.coco_regions import *
            >>> from ndsampler import coco_sampler
            >>> self = coco_sampler.CocoSampler.demo().regions
            >>> window_dims = (64, 64)
            >>> self._preselect_positives(window_dims=window_dims, verbose=4)
        """
        if verbose is None:
            verbose = self.verbose

        # HACK: USE A MONKEY PATCHED CUSTOM SAMPLER
        if hasattr(self, 'custom_preselect_positives'):
            self._pos_select_idxs = self.custom_preselect_positives(
                self, window_dims, rng=rng)
        else:
            if True:
                self._pos_select_idxs = np.arange(len(self.targets))
            else:
                # OLD: NMS-based positive selection
                # TODO: get a more varied positive sample
                # TODO: generalize the type of positive sampling.  The setcover
                # approach with window_dims, only makes sense in the context of
                # detection.
                disable = rng is None
                extra_deps = ub.odict()
                extra_deps['window_size'] = window_dims
                extra_deps['num'] = num
                extra_deps['rng'] = rng
                cacher = self._cacher('_pos_select_idxs',
                                      extra_deps=extra_deps, disable=disable,
                                      verbose=verbose)
                _pos_select_idxs = cacher.tryload(on_error='clear')
                if _pos_select_idxs is None:
                    _pos_select_idxs = select_positive_regions(
                        self.targets, window_dims=window_dims, rng=rng
                    )
                    cacher.save(_pos_select_idxs)
                self._pos_select_idxs = _pos_select_idxs

        n_pos = len(self._pos_select_idxs)
        if verbose:
            print('Preselected {} positives'.format(n_pos))

        return n_pos

    def _preselect_negatives(self, num, window_dims=None, thresh=0.3, rng=None,
                             verbose=None):
        """
        Preselect a set of random regions to be used as negative examples.

        Args:
            num (int): number of desired negatives to preselect. In some cases
                achieving this number may not be possible.

            window_dims (Tuple): absolute dimensions (height, width)
                used to sample negative regions. If not specified the relative
                anchor strategy will be used to randomly choose potentially
                non-square regions relative to the image size.

            thresh (float): overlap area threshold as a percentage of the
                negative box size. When thresh=0.0, that means negatives cannot
                overlap any positive, when threh=1.0, there are no constrains
                on negative placement.

            rng (int | RandomState): random seed / state

            verbose (int): verbosity level

        Returns:
            int : number of negatives actually chosen

        Example:
            >>> from ndsampler.coco_regions import *
            >>> import ndsampler
            >>> self = ndsampler.CocoSampler.demo().regions
            >>> num = 100
            >>> self._preselect_negatives(num, window_dims=(30, 30))
        """
        if verbose is None:
            verbose = self.verbose

        if verbose > 2:
            print('Preselect {} negatives'.format(num))

        # Explicitly disable caching if rng is not seeded
        disable = rng is None

        rng = kwarray.ensure_rng(rng)
        if window_dims is not None:
            window_size = window_dims[::-1]
            neg_anchors = None
        else:
            window_size = None
            neg_anchors = self.neg_anchors

        # Setup extra dependencies
        extra_deps = ub.odict()
        extra_deps['window_size'] = window_size
        extra_deps['neg_anchors'] = neg_anchors
        extra_deps['num'] = num
        extra_deps['rng'] = rng
        extra_deps['thresh'] = thresh

        cacher = self._cacher('_negative_pool', extra_deps=extra_deps,
                              verbose=verbose, disable=disable)
        _negative_pool = cacher.tryload(on_error='clear')
        if _negative_pool is None:
            _negative_pool = self._random_negatives(
                num, window_size=window_size, neg_anchors=neg_anchors,
                thresh=thresh, exact='warn', rng=rng)
            cacher.save(_negative_pool)
        self._negative_pool = _negative_pool
        self._negative_idx = 0
        num_neg = len(self._negative_pool)
        if verbose > 0:
            print('Preselected {} negatives'.format(num_neg))
        return num_neg

    def _cacher(self, fname, extra_deps=None, disable=False, verbose=None):
        """
        Create a cacher for a known lazy computation using a common hashid.

        If `self.workdir` or `self.hashid` is None, then caches are disabled by
        default. Caches can be explicitly disabled by setting the appropriate
        value in the `self._enabled_caches` dictionary.

        Args:
            fname (str): name of the property we are caching
            extra_deps (OrderedDict): extra data to contribute to the hashid
            disable (bool): explicitly disable cache if True, otherwise do
                normal checks to see if enabled.
            verbose (bool, default=None): if specified overrides `self.verbose`.

        Returns:
            ub.Cacher: cacher - if enabled this cacher will minimally depend
                on the `self.hashid`, but may also depend on extra info.
        """
        if verbose is None:
            verbose = self.verbose

        if not disable and self.hashid and self.workdir:
            enabled = self._enabled_caches.get(fname, True)
            dpath = ub.ensuredir((self.workdir, '_cache', fname))
        else:
            dpath = None
            enabled = False  # forced disable

        depends = None
        if enabled:
            if extra_deps is None:
                extra_deps = ub.odict()
            elif not isinstance(extra_deps, ub.odict):
                raise TypeError('Extra dependencies must be an OrderedDict')
            # always include `self.hashid`
            extra_deps['self_hashid'] = self.hashid
            depends = extra_deps

        cacher = ub.Cacher(fname, depends=depends, dpath=dpath,
                           verbose=self.verbose, enabled=enabled)
        return cacher


@profile
def tabular_coco_targets(dset):
    """
    Transforms COCO box annotations into a tabular form

    _ = xdev.profile_now(tabular_coco_targets)(dset)
    """
    import warnings
    # TODO: better handling of non-bounding box annotations; ignore for now

    if hasattr(dset, 'tabular_targets'):
        # In the SQL case, we can write a single query that
        # builds the table more efficiently.
        return dset.tabular_targets()

    img_items = list(dset.imgs.items())
    gid_to_width = {gid: img['width'] for gid, img in img_items}
    gid_to_height = {gid: img['height'] for gid, img in img_items}

    try:
        anns = dset.dataset['annotations']
        if not isinstance(anns, list):
            anns = list(anns)

        xywh = [ann['bbox'] for ann in anns]
        xywh = np.array(xywh, dtype=np.float32)
    except Exception:
        has_bbox = [ann.get('bbox', None) is not None for ann in anns]
        if not all(has_bbox):
            n_missing = len(has_bbox) - sum(has_bbox)
            warnings.warn('CocoDataset is missing boxes '
                          'for {} annotations'.format(n_missing))
        anns = list(ub.compress(anns, has_bbox))
        xywh = [ann['bbox'] for ann in anns]
        xywh = np.array(xywh, dtype=np.float32)

    boxes = kwimage.Boxes(xywh, 'xywh')
    cxywhs = boxes.to_cxywh().data.reshape(-1, 4)

    aids = [ann['id'] for ann in anns]
    gids = [ann['image_id'] for ann in anns]
    cids = [ann['category_id'] for ann in anns]

    img_width = [gid_to_width[gid] for gid in gids]
    img_height = [gid_to_height[gid] for gid in gids]

    aids = np.array(aids, dtype=np.int32)
    gids = np.array(gids, dtype=np.int32)
    cids = np.array(cids, dtype=np.int32)

    table = {
        # Annotation / Image / Category ids
        'aid': aids,
        'gid': gids,
        'category_id': cids,
        # Subpixel box localizations wrt parent image
        'cx': cxywhs.T[0],
        'cy': cxywhs.T[1],
        'width': cxywhs.T[2],
        'height': cxywhs.T[3],
    }

    # Parent image id and width / height
    table['img_width'] = np.array(img_width, dtype=np.int32)
    table['img_height'] = np.array(img_height, dtype=np.int32)

    # table = ub.map_vals(np.asarray, table)
    targets = kwarray.DataFrameArray(table)
    return targets


@profile
def select_positive_regions(targets, window_dims=(300, 300), thresh=0.0,
                            rng=None, verbose=0):
    """
    Reduce positive example redundency by selecting disparate positive samples

    Example:
        >>> from ndsampler.coco_regions import *
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('shapes8')
        >>> targets = tabular_coco_targets(dset)
        >>> window_dims = (300, 300)
        >>> selected = select_positive_regions(targets, window_dims)
        >>> print(len(selected))
        >>> print(len(dset.anns))
    """
    unique_gids, groupxs = kwarray.group_indices(targets['gid'])
    gid_to_groupx = dict(zip(unique_gids, groupxs))
    wh, ww = window_dims
    rng = kwarray.ensure_rng(rng)
    selection = []

    # Get all the bounding boxes
    cxs, cys = ub.take(targets, ['cx', 'cy'])
    n = len(targets)
    cxs = cxs.astype(np.float32)
    cys = cys.astype(np.float32)
    wws = np.full(n, ww, dtype=np.float32)
    whs = np.full(n, wh, dtype=np.float32)
    cxywh = np.hstack([a[:, None] for a in [cxs, cys, wws, whs]])
    boxes = kwimage.Boxes(cxywh, 'cxywh').to_tlbr()

    iter_ = ub.ProgIter(gid_to_groupx.items(),
                        enabled=verbose,
                        label='select positive regions',
                        total=len(gid_to_groupx), adjust=0, freq=32)

    for gid, groupx in iter_:
        # Select all candiate windows in this image
        cand_windows = boxes.take(groupx, axis=0)
        # Randomize which candidate windows have the highest scores so the
        # selection can vary each epoch.
        cand_scores = rng.rand(len(cand_windows))
        cand_dets = kwimage.Detections(boxes=cand_windows, scores=cand_scores)
        # Non-max supresssion is really similar to set-cover
        keep = cand_dets.non_max_supression(thresh=thresh)
        selection.extend(groupx[keep])

    selection = np.array(sorted(selection))
    return selection


def new_video_sample_grid(dset, window_dims=None, window_overlap=0.0,
                          space_dims=None, time_dim=None,  # TODO
                          classes_of_interest=None, ignore_coverage_thresh=0.6,
                          negative_classes={'ignore', 'background'},
                          use_annots=True, legacy=True, verbose=1):
    """
    Create a space time-grid to sample with

    Returns:

        Dict: sample_grid

            contains "targets", and if use_annots=True then also
                contains "positives_indexes" and "negatives_indexes" indicating
                which annotations contain positive/negative samples.

            The "positives" and "negatives" lists are deprecated and will be
            removed.

    Example:
        >>> from ndsampler.coco_regions import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral', num_frames=5)
        >>> dset.conform()
        >>> window_dims = (2, 224, 224)
        >>> sample_grid = new_video_sample_grid(dset, window_dims)
        >>> print('sample_grid = {}'.format(ub.repr2(sample_grid, nl=2)))
        >>> # Now try to load a sample
        >>> tr = sample_grid['positives'][0]
        >>> import ndsampler
        >>> sampler = ndsampler.CocoSampler(dset)
        >>> tr_ = sampler._infer_target_attributes(tr)
        >>> print('tr_ = {}'.format(ub.repr2(tr_, nl=1)))
        >>> sample = sampler.load_sample(tr)
        >>> assert sample['im'].shape == (2, 224, 224, 5)

    Example:
        >>> from ndsampler.coco_regions import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral', num_frames=5)
        >>> dset.conform()
        >>> window_dims = (2, 224, 224)
        >>> sample_grid = new_video_sample_grid(dset, window_dims, use_annots=False)

    Ignore:
        import timerit
        ti = timerit.Timerit(10, bestof=3, verbose=2)
        for timer in ti.reset('vid use_annots=True'):
            with timer:
                new_video_sample_grid(dset, window_dims, use_annots=True, verbose=0)

        for timer in ti.reset('vid use_annots=False'):
            with timer:
                new_video_sample_grid(dset, window_dims, use_annots=False, verbose=0)

        import timerit
        ti = timerit.Timerit(10, bestof=3, verbose=2)
        for timer in ti.reset('img use_annots=True'):
            with timer:
                new_image_sample_grid(dset, window_dims[1:], use_annots=True, verbose=0)

        for timer in ti.reset('img use_annots=False'):
            with timer:
                new_image_sample_grid(dset, window_dims[1:], use_annots=False, verbose=0)

    Ignore:
        import xdev
        globals().update(xdev.get_func_kwargs(new_video_sample_grid))
    """
    import kwarray
    from ndsampler import isect_indexer
    keepbound = True

    if classes_of_interest:
        raise NotImplementedError

    # Create a sliding window object for each specific image (because they may
    # have different sizes, technically we could memoize this)
    vidid_to_slider = {}
    for vidid, video in dset.index.videos.items():
        gids = dset.index.vidid_to_gids[vidid]
        num_frames = len(gids)
        full_dims = [num_frames, video['height'], video['width']]
        window_dims_ = full_dims if window_dims == 'full' else window_dims
        slider = kwarray.SlidingWindow(full_dims, window_dims_,
                                       overlap=window_overlap,
                                       keepbound=keepbound,
                                       allow_overshoot=True)

        vidid_to_slider[vidid] = slider

    @ub.memoize
    def _lut_warp(gid):
        warp_img_to_vid = dset.index.imgs[gid].get('warp_img_to_vid', None)
        return kwimage.Affine.coerce(warp_img_to_vid)

    _isect_index = isect_indexer.FrameIntersectionIndex.from_coco(dset)

    positives = []
    negatives = []
    targets = []
    positive_idxs = []
    negative_idxs = []
    for vidid, slider in vidid_to_slider.items():
        video_regions = list(slider)
        gids = dset.index.vidid_to_gids[vidid]
        for vid_region in video_regions:
            t_sl, y_sl, x_sl = vid_region
            vid_box = kwimage.Boxes.from_slice((y_sl, x_sl))
            region_gids = gids[t_sl]

            if use_annots:
                region_aids = []
                for gid in region_gids:
                    warp_img_to_vid = _lut_warp(gid)
                    # Check to see what annotations this window-box overlaps with
                    # (in image space!)
                    img_box = vid_box.warp(warp_img_to_vid.inv())
                    aids = _isect_index.overlapping_aids(gid, img_box)
                    region_aids.extend(aids)

                pos_aids = sorted(region_aids)
            else:
                pos_aids = None

            time_slice = vid_region[0]
            space_slice = vid_region[1:3]
            tr = {
                'vidid': vidid,
                'time_slice': time_slice,
                'space_slice': space_slice,
                # 'slices': region,
                'gids': region_gids,
                'aids': pos_aids,
            }
            targets.append(tr)
            if pos_aids:
                positive_idxs.append(len(targets))
                if legacy:
                    positives.append(tr)
            else:
                negative_idxs.append(len(targets))
                if legacy:
                    negatives.append(tr)

    if verbose:
        print('Found {} targets'.format(len(targets)))
        if use_annots:
            print('Found {} positives'.format(len(positive_idxs)))
            print('Found {} negatives'.format(len(negative_idxs)))

    sample_grid = {
        'targets': targets,
        'positives_indexes': positive_idxs,
        'negatives_indexes': negative_idxs,
    }
    if legacy:
        sample_grid.update({
            # Deprecated:
            'positives': positives,
            'negatives': negatives,
        })
    return sample_grid


def new_image_sample_grid(dset, window_dims, window_overlap=0.0,
                          classes_of_interest=None, ignore_coverage_thresh=0.6,
                          negative_classes={'ignore', 'background'},
                          use_annots=True, legacy=True, verbose=1):
    """
    Create a space time-grid to sample with

    Returns:
        Dict: sample_grid

            contains "targets", and if use_annots=True then also
                contains "positives_indexes" and "negatives_indexes" indicating
                which annotations contain positive/negative samples.

            The "positives" and "negatives" lists are deprecated and will be
            removed.

    Example:
        >>> from ndsampler.coco_regions import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('shapes8')
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> window_dims = (224, 224)
        >>> sample_grid1 = new_image_sample_grid(dset, window_dims, use_annots=False)
        >>> sample_grid = new_image_sample_grid(dset, window_dims)
        >>> # Now try to load a sample
        >>> idx = sample_grid['positives_indexes'][0]
        >>> tr = sample_grid['targets'][idx]
        >>> import ndsampler
        >>> sampler = ndsampler.CocoSampler(dset)
        >>> tr['channels'] = '<all>'
        >>> tr_ = sampler._infer_target_attributes(tr)
        >>> print('tr_ = {}'.format(ub.repr2(tr_, nl=1)))
        >>> sample = sampler.load_sample(tr)
        >>> assert sample['im'].shape == (224, 224, 5)

    Ignore:
        import xdev
        globals().update(xdev.get_func_kwargs(new_image_sample_grid))
    """
    # import netharn as nh
    import kwarray
    from ndsampler import isect_indexer
    keepbound = True

    # Create a sliding window object for each specific image (because they may
    # have different sizes, technically we could memoize this)
    gid_to_slider = {}
    for img in dset.imgs.values():
        full_dims = [img['height'], img['width']]
        window_dims_ = full_dims if window_dims == 'full' else window_dims
        slider = kwarray.SlidingWindow(full_dims, window_dims_,
                                       overlap=window_overlap, keepbound=keepbound,
                                       allow_overshoot=True)
        gid_to_slider[img['id']] = slider

    if use_annots:
        _isect_index = isect_indexer.FrameIntersectionIndex.from_coco(dset)

    positives = []
    negatives = []
    targets = []
    positive_idxs = []
    negative_idxs = []
    for gid, slider in gid_to_slider.items():

        # For each image, create a box for each spatial region in the slider
        boxes = []
        regions = list(slider)
        for region in regions:
            y_sl, x_sl = region
            boxes.append([x_sl.start,  y_sl.start, x_sl.stop, y_sl.stop])
        boxes = kwimage.Boxes(np.array(boxes), 'ltrb')

        for region, box in zip(regions, boxes):

            if use_annots:
                # Check to see what annotations this window-box overlaps with
                aids = _isect_index.overlapping_aids(gid, box)

                # Look at the categories within this region
                catnames = [
                    dset.cats[dset.anns[aid]['category_id']]['name'].lower()
                    for aid in aids
                ]

                if ignore_coverage_thresh:
                    ignore_flags = [catname == 'ignore' for catname in catnames]
                    if any(ignore_flags):
                        # If the almost the entire window is marked as ignored then
                        # just skip this window.
                        ignore_aids = list(ub.compress(aids, ignore_flags))
                        ignore_boxes = dset.annots(ignore_aids).boxes

                        # Get an upper bound on coverage to short circuit extra
                        # computation in simple cases.
                        box_area = box.area.sum()
                        coverage_ub = ignore_boxes.area.sum() / box_area
                        if coverage_ub  > ignore_coverage_thresh:
                            max_coverage = ignore_boxes.iooas(box).max()
                            if max_coverage > ignore_coverage_thresh:
                                continue
                            elif len(ignore_boxes) > 1:
                                # We have to test the complex case
                                try:
                                    from shapely.ops import cascaded_union
                                    ignore_shape = cascaded_union(ignore_boxes.to_shapley())
                                    region_shape = box[None, :].to_shapley()[0]
                                    coverage_shape = ignore_shape.intersection(region_shape)
                                    real_coverage = coverage_shape.area / box_area
                                    if real_coverage > ignore_coverage_thresh:
                                        continue
                                except Exception as ex:
                                    import warnings
                                    warnings.warn(
                                        'ignore region select had non-critical '
                                        'issue ex = {!r}'.format(ex))

                if classes_of_interest:
                    # If there are CoIs then only count a region as positive if one
                    # of those is in this region
                    interest_flags = np.array([
                        catname in classes_of_interest for catname in catnames])
                    pos_aids = list(ub.compress(aids, interest_flags))
                elif negative_classes:
                    # Don't count negative classes as positives
                    nonnegative_flags = np.array([
                        catname not in negative_classes for catname in catnames])
                    pos_aids = list(ub.compress(aids, nonnegative_flags))
                else:
                    pos_aids = aids
            else:
                aids = None
                pos_aids = None

            # aids = sampler.regions.overlapping_aids(gid, box, visible_thresh=0.001)
            tr = {
                'gid': gid,
                'slices': region,
                'aids': aids,
            }
            targets.append(tr)
            if pos_aids:
                positive_idxs.append(len(targets))
                if legacy:
                    positives.append(tr)
            else:
                negative_idxs.append(len(targets))
                if legacy:
                    negatives.append(tr)

    if verbose:
        print('Found {} targets'.format(len(targets)))
        if use_annots:
            print('Found {} positives'.format(len(positive_idxs)))
            print('Found {} negatives'.format(len(negative_idxs)))

    sample_grid = {
        'targets': targets,
        'positives_indexes': positive_idxs,
        'negatives_indexes': negative_idxs,
    }
    if legacy:
        sample_grid.update({
            # Deprecated:
            'positives': positives,
            'negatives': negatives,
        })
    return sample_grid
