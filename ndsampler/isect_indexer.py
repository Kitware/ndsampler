import numpy as np
import ubelt as ub
import itertools as it
import warnings
import kwarray
import kwimage
import os

try:
    USE_RTREE = os.environ.get('USE_RTREE', '').lower()
    USE_RTREE = USE_RTREE in {'0', 'false'} or USE_RTREE
    if not USE_RTREE:
        raise ImportError
    import rtree
    pyqtree = None
except ImportError:
    import pyqtree
    rtree = None

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class FrameIntersectionIndex(ub.NiceRepr):
    """
    Build spatial tree for each frame so we can quickly determine if a random
    negative is too close to a positive. For each frame/image we built a qtree.

    Example:
        >>> from ndsampler.isect_indexer import *
        >>> import kwcoco
        >>> import ubelt as ub
        >>> dset = kwcoco.CocoDataset.demo()
        >>> dset._ensure_imgsize()
        >>> dset.remove_annotations([ann for ann in dset.anns.values()
        >>>                          if 'bbox' not in ann])
        >>> # Build intersection index around coco dataset
        >>> self = FrameIntersectionIndex.from_coco(dset)
        >>> gid = 1
        >>> box = kwimage.Boxes([0, 10, 100, 100], 'xywh')
        >>> isect_aids, ious = self.ious(gid, box)
        >>> print(ub.urepr(ious.tolist(), nl=0, precision=4))
        [0.0507]
    """
    def __init__(self):
        self.qtrees = None
        self.all_gids = None

    def __nice__(self):
        if self.all_gids is None:
            return 'None'
        else:
            return len(self.all_gids)

    @classmethod
    def from_coco(cls, dset, verbose=0):
        """
        Args:
            dset (kwcoco.CocoDataset): positive annotation data

        Returns:
            FrameIntersectionIndex
        """
        self = cls()
        self.qtrees = self._build_index(dset, verbose=verbose)
        self.all_gids = sorted(self.qtrees.keys())
        return self

    @classmethod
    def demo(cls, *args, **kwargs):
        """
        Create a demo intersection index.

        Args:
            *args: see kwcoco.CocoDataset.demo
            **kwargs: see kwcoco.CocoDataset.demo

        Returns:
            FrameIntersectionIndex
        """
        import kwcoco
        dset = kwcoco.CocoDataset.demo(*args, **kwargs)
        dset._ensure_imgsize()
        dset.remove_annotations([ann for ann in dset.anns.values()
                                 if 'bbox' not in ann])
        self = cls.from_coco(dset)
        return self

    @staticmethod
    @profile
    def _build_index(dset, verbose=0):
        """
        """
        if verbose:
            print('Building isect index')
        if rtree is not None:
            # Try using trees first
            qtrees = {}
            for img in ub.ProgIter(dset.dataset['images'], desc='init rtrees', verbose=verbose):
                gid = img['id']
                qtree = rtree.Index()
                qtree.width = img['width']
                qtree.height = img['height']
                qtrees[gid] = qtree
                # qtrees[gid].bounds = [0, 0, img['width'], img['height']]
            for qtree in qtrees.values():
                qtree.aid_to_ltrb = {}  # Add extra index to track boxes

            if dset.index is not None:
                for gid, aids in ub.ProgIter(dset.index.gid_to_aids.items(),
                                             total=len(dset.index.gid_to_aids),
                                             desc='populate qtrees',
                                             verbose=verbose):
                    annots = dset.annots(aids)
                    qtree = qtrees[gid]
                    ltrb_boxes = annots.boxes.to_ltrb().data
                    qtree.aid_to_ltrb = ub.dzip(aids, ltrb_boxes)
                    for aid, ltrb_box in qtree.aid_to_ltrb.items():
                        qtree.insert(aid, ltrb_box)
            else:

                for ann in ub.ProgIter(dset.dataset['annotations'],
                                       desc='populate qtrees', verbose=verbose):

                    bbox = ann.get('bbox', None)
                    if bbox is not None:
                        aid = ann['id']
                        qtree = qtrees[ann['image_id']]
                        xywh_box = kwimage.Boxes(bbox, 'xywh')
                        ltrb_box = xywh_box.to_ltrb().data
                        qtree.insert(aid, ltrb_box)
                        qtree.aid_to_ltrb[aid] = ltrb_box
        else:
            qtrees = {
                img['id']: pyqtree.Index((0, 0, img['width'], img['height']))
                for img in ub.ProgIter(dset.dataset['images'],
                                       desc='init qtrees', verbose=verbose)
            }
            for qtree in qtrees.values():
                qtree.aid_to_ltrb = {}  # Add extra index to track boxes
            for ann in ub.ProgIter(dset.dataset['annotations'],
                                   desc='populate qtrees', verbose=verbose):
                bbox = ann.get('bbox', None)
                if bbox is not None:
                    aid = ann['id']
                    qtree = qtrees[ann['image_id']]
                    # Orig variant:
                    # xywh_box = kwimage.Boxes(bbox, 'xywh')
                    # ltrb_box = xywh_box.to_ltrb().data
                    # Optimized Code:
                    x, y, w, h = bbox
                    ltrb_box = [x, y, x + w, y + h]
                    ltrb_box = np.array(ltrb_box)
                    qtree.insert(aid, ltrb_box)
                    qtree.aid_to_ltrb[aid] = ltrb_box
        return qtrees

    @profile
    def overlapping_aids(self, gid, box):
        """
        Find all annotation-ids within an image that have some overlap with a
        bounding box.

        Args:
            gid (int): an image id
            box (kwimage.Boxes): the specified region

        Returns:
            List[int]: list of annotation ids

        CommandLine:
            USE_RTREE=0 xdoctest -m ndsampler.isect_indexer FrameIntersectionIndex.overlapping_aids
            USE_RTREE=1 xdoctest -m ndsampler.isect_indexer FrameIntersectionIndex.overlapping_aids

        Example:
            >>> from ndsampler.isect_indexer import *  # NOQA
            >>> self = FrameIntersectionIndex.demo('shapes128')
            >>> for gid, qtree in self.qtrees.items():
            >>>     box = kwimage.Boxes([0, 0, qtree.width, qtree.height], 'xywh')
            >>>     print(self.overlapping_aids(gid, box))
        """
        boxes1 = box[None, :] if len(box.shape) == 1 else box
        qtree = self.qtrees[gid]
        query = boxes1.to_ltrb(copy=False).data[0]
        if rtree is None:
            isect_aids = sorted(set(qtree.intersect(query)))
        else:
            isect_aids = sorted(set(qtree.intersection(query)))
        return isect_aids

    def ious(self, gid, box):
        """
        Find overlapping annotations in a specific image and their intersection
        over union with a a query box.

        Args:
            gid (int): an image id
            box (kwimage.Boxes): the specified region

        Returns:
            Tuple[List[int], ndarray]:
                isect_aids: list of annotation ids
                ious: jaccard score for each returned annotation id
        """
        boxes1 = box[None, :] if len(box.shape) == 1 else box
        isect_aids = self.overlapping_aids(gid, box)
        if len(isect_aids):
            boxes1 = box[None, :]
            boxes2 = [self.qtrees[gid].aid_to_ltrb[aid] for aid in isect_aids]
            boxes2 = kwimage.Boxes(np.array(boxes2), 'ltrb')
            ious = boxes1.ious(boxes2)[0]
        else:
            ious = np.empty(0)
        return isect_aids, ious

    def iooas(self, gid, box):
        """
        Intersection over other's area

        Args:
            gid (int): an image id
            box (kwimage.Boxes): the specified region

        Like iou, but non-symetric, returned number is a percentage of the
        other's  (groundtruth) area. This means we dont care how big the
        (negative) `box` is.
        """
        boxes1 = box[None, :] if len(box.shape) == 1 else box
        isect_aids = self.overlapping_aids(gid, box)
        if len(isect_aids):
            boxes2 = [self.qtrees[gid].aid_to_ltrb[aid] for aid in isect_aids]
            boxes2 = kwimage.Boxes(np.array(boxes2), 'ltrb')
            isect = boxes1.isect_area(boxes2)
            denom = boxes2.area.T
            eps = 1e-6
            iomas = isect / (denom[0] + eps)
        else:
            iomas = np.empty(0)
        return isect_aids, iomas

    def random_negatives(self, num, anchors=None, window_size=None, gids=None,
                         thresh=0.0, exact=True, rng=None, patience=None):
        """
        Finds random boxes that don't have a large overlap with positive
        instances.

        Args:
            num (int): number of negative boxes to generate (actual number of
                boxes returned may be less unless `exact=True`)

            anchors (ndarray): prior normalized aspect ratios for negative
                boxes. Mutually exclusive with `window_size`.

            window_size (ndarray): absolute (W, H) sizes to use for negative
                boxes.  Mutually exclusive with `anchors`.

            gids (List[int]): image-ids to generate negatives for,
                if not specified generates for all images.

            thresh (float): overlap area threshold as a percentage of
                the negative box size. When thresh=0.0, that means
                negatives cannot overlap any positive, when threh=1.0, there
                are no constrains on negative placement.

            exact (bool): if True, ensure that we generate exactly `num` boxes

            rng (RandomState): random number generator

        Example:
            >>> from ndsampler.isect_indexer import *
            >>> import ndsampler
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> self = FrameIntersectionIndex.from_coco(dset)
            >>> anchors = np.array([[.35, .15], [.2, .2], [.1, .1]])
            >>> #num = 25
            >>> num = 5
            >>> rng = kwarray.ensure_rng(None)
            >>> neg_gids, neg_boxes = self.random_negatives(
            >>>     num, anchors, gids=[1], rng=rng, thresh=0.01, exact=1)
            >>> # xdoc: +REQUIRES(--show)
            >>> gid = sorted(set(neg_gids))[0]
            >>> boxes = neg_boxes.compress(neg_gids == gid)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> img = kwimage.imread(dset.imgs[gid]['file_name'])
            >>> kwplot.imshow(img, doclf=True, fnum=1, colorspace='bgr')
            >>> support = self._support(gid)
            >>> kwplot.draw_boxes(support, color='blue')
            >>> kwplot.draw_boxes(boxes, color='orange')

        Example:
            >>> from ndsampler.isect_indexer import *
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> self = FrameIntersectionIndex.from_coco(dset)
            >>> #num = 25
            >>> num = 5
            >>> rng = kwarray.ensure_rng(None)
            >>> window_size = (50, 50)
            >>> neg_gids, neg_boxes = self.random_negatives(
            >>>     num, window_size=window_size, gids=[1], rng=rng,
            >>>     thresh=0.01, exact=1)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> gid = sorted(set(neg_gids))[0]
            >>> boxes = neg_boxes.compress(neg_gids == gid)
            >>> img = kwimage.imread(dset.imgs[gid]['file_name'])
            >>> kwplot.imshow(img, doclf=True, fnum=1, colorspace='bgr')
            >>> support = self._support(gid)
            >>> support.draw(color='blue')
            >>> boxes.draw(color='orange')
        """

        if not ((window_size is None) ^ (anchors is None)):
            raise ValueError('window_size and anchors are mutually exclusive')

        rng = kwarray.ensure_rng(rng)
        all_gids = self.all_gids if gids is None else gids

        def _generate_rel(n):
            # Generate n candidate boxes in the normalized 0-1 domain
            cand_boxes = kwimage.Boxes.random(num=n, scale=1.0, format='ltrb',
                                              anchors=anchors, anchor_std=0,
                                              rng=rng)

            chosen_gids = np.array(sorted(rng.choice(all_gids, size=n)))
            gid_to_boxes = kwarray.group_items(cand_boxes, chosen_gids, axis=0)

            neg_gids = []
            neg_boxes = []
            for gid, img_boxes in gid_to_boxes.items():
                qtree = self.qtrees[gid]
                # scale from normalized coordinates to image coordinates
                img_boxes = img_boxes.scale((qtree.width, qtree.height))
                for box in img_boxes:
                    # isect_aids, overlaps = self.ious(gid, box)
                    isect_aids, overlaps = self.iooas(gid, box)
                    if len(overlaps) == 0 or overlaps.max() < thresh:
                        neg_gids.append(gid)
                        neg_boxes.append(box.data)
            return neg_gids, neg_boxes

        def _generate_abs(n):
            # Randomly choose images to generate boxes for
            chosen_gids = np.array(sorted(rng.choice(all_gids, size=n)))
            gid_to_nboxes = ub.dict_hist(chosen_gids)

            neg_gids = []
            neg_boxes = []
            for gid, nboxes in gid_to_nboxes.items():
                qtree = self.qtrees[gid]
                scale = (qtree.width, qtree.height)
                anchors_ = np.array([window_size]) / np.array(scale)
                if np.any(anchors_ > 1.0):
                    continue
                img_boxes = kwimage.Boxes.random(
                    num=nboxes, scale=1.0, format='ltrb', anchors=anchors_,
                    anchor_std=0, rng=rng)
                img_boxes = img_boxes.scale(scale)
                for box in img_boxes:
                    # isect_aids, overlaps = self.ious(gid, box)
                    isect_aids, overlaps = self.iooas(gid, box)
                    if len(overlaps) == 0 or overlaps.max() < thresh:
                        neg_gids.append(gid)
                        neg_boxes.append(box.data)
            return neg_gids, neg_boxes

        if window_size is not None:
            _generate = _generate_abs
        elif anchors is not None:
            _generate = _generate_rel
        else:
            raise ValueError('must specify at least one window_size or anchors')

        if exact:
            # TODO: Dont attempt to sample negatives from images where the
            # positives cover more than a threshold percent. (Handle the case
            # of chip detections)

            factor = 2  # oversample factor
            if patience is None:
                patience = int(np.sqrt(num * 10) + 1)
            remaining_patience = patience
            timer = ub.Timer().tic()
            # Generate boxes until we have enough
            neg_gids, neg_boxes = _generate(n=int(num * factor))
            n_tries = 1
            for n_tries in it.count(n_tries):
                want = num - len(neg_boxes)
                if want <= 0:
                    break
                extra_gids, extra_boxes = _generate(n=int(want * factor))
                neg_gids.extend(extra_gids)
                neg_boxes.extend(extra_boxes)
                if len(neg_boxes) < num:
                    # If we haven't found a significant number of boxes our
                    # patience decreases (if the wall time is getting large)
                    if len(extra_boxes) <= (num // 10) and timer.toc() > 1.0:
                        remaining_patience -= 1
                        if remaining_patience == 0:
                            break

            if len(neg_boxes) < num:
                # but throw an error if we don't make any progress
                message = (
                    'Cannot make a negative sample with thresh={} '
                    'in under {} tries. Found {} but need {}'.format(
                        thresh, n_tries, len(neg_boxes), num))
                if exact == 'warn':
                    warnings.warn(message)
                else:
                    raise Exception(message)
            print('n_tries = {!r}'.format(n_tries))

            neg_gids = neg_gids[:num]
            neg_boxes = neg_boxes[:num]
        else:
            neg_gids, neg_boxes = _generate(n=num)

        neg_gids = np.array(neg_gids)
        neg_boxes = kwimage.Boxes(np.array(neg_boxes), 'ltrb')
        return neg_gids, neg_boxes

    def _debug_index(self):
        from shapely.ops import cascaded_union

        def _to_shapely(boxes):
            from shapely.geometry import Polygon
            from kwimage.structs.boxes import _cat
            x1, y1, x2, y2 = boxes.to_ltrb(copy=False).components
            a = _cat([x1, y1]).tolist()
            b = _cat([x1, y2]).tolist()
            c = _cat([x2, y2]).tolist()
            d = _cat([x2, y1]).tolist()
            polygons = [Polygon(points) for points in zip(a, b, c, d, a)]
            return polygons

        for gid, qtree in self.qtrees.items():
            boxes = kwimage.Boxes(np.array(list(qtree.aid_to_ltrb.values())), 'ltrb')
            polygons = _to_shapely(boxes)

            bounds = kwimage.Boxes([[0, 0, qtree.width, qtree.height]], 'ltrb')
            bounds = _to_shapely(bounds)[0]
            merged_polygon = cascaded_union(polygons)
            uncovered = (bounds - merged_polygon)
            print('uncovered.area = {!r}'.format(uncovered.area))

            # plot these two polygons separately
            if 1:
                from descartes import PolygonPatch
                from matplotlib import pyplot as plt
                import kwplot
                kwplot.autompl()
                fig = plt.figure(gid)
                ax = fig.add_subplot(111)
                ax.cla()
                # ax.add_patch(
                #     PolygonPatch(bounds, alpha=0.5, zorder=2, fc='blue')
                # )
                # ax.add_patch(
                #     PolygonPatch(merged_polygon, alpha=0.5, zorder=2, fc='red')
                # )
                ax.add_patch(
                    PolygonPatch(uncovered, alpha=0.5, zorder=2, fc='green')
                )
                ax.set_xlim(0, qtree.width)
                ax.set_ylim(0, qtree.height)
                ax.set_aspect(1)

    def _support(self, gid):
        qtree = self.qtrees[gid]
        support_boxes = kwimage.Boxes(list(qtree.aid_to_ltrb.values()), 'ltrb')

        return support_boxes
