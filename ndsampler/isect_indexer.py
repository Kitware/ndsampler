# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import kwil
import numpy as np
import pyqtree
import ubelt as ub
import itertools as it
import warnings


class FrameIntersectionIndex(object):
    """
    Build spatial tree for each frame so we can quickly determine if a random
    negative is too close to a positive. For each frame/image we built a qtree.


    Example:
        >>> from ndsampler.isect_indexer import *
        >>> import ndsampler
        >>> import ubelt as ub
        >>> dset = ndsampler.CocoDataset.demo()
        >>> dset._ensure_imgsize()
        >>> dset.remove_annotations([ann for ann in dset.anns.values()
        >>>                          if 'bbox' not in ann])
        >>> # Build intersection index aroung coco dataset
        >>> self = FrameIntersectionIndex.from_coco(dset)
        >>> gid = 1
        >>> box = kwil.Boxes([0, 10, 100, 100], 'xywh')
        >>> isect_aids, ious = self.ious(gid, box)
        >>> print(ub.repr2(ious.tolist(), nl=0, precision=4))
        [0.0507]
    """
    def __init__(self):
        self.qtrees = None
        self.all_gids = None

    @classmethod
    def from_coco(cls, dset):
        """
        Args:
            dset (ndsampler.CocoDataset): positive annotation data
        """
        self = cls()
        self.qtrees = self._build_index(dset)
        self.all_gids = sorted(self.qtrees.keys())
        return self

    @staticmethod
    def _build_index(dset):
        qtrees = {
            img['id']: pyqtree.Index((0, 0, img['width'], img['height']))
            for img in dset.dataset['images']
        }
        for qtree in qtrees.values():
            qtree.aid_to_tlbr = {}  # Add extra index to track boxes
        for ann in dset.dataset['annotations']:
            if 'bbox' in ann:
                aid = ann['id']
                qtree = qtrees[ann['image_id']]
                tlwh_box = kwil.Boxes(ann['bbox'], 'tlwh')
                tlbr_box = tlwh_box.to_tlbr().data
                qtree.insert(aid, tlbr_box)
                qtree.aid_to_tlbr[aid] = tlbr_box
        return qtrees

    def overlapping_aids(self, gid, box):
        qtree = self.qtrees[gid]
        query = box.to_tlbr().data
        isect_aids = sorted(qtree.intersect(query))
        return isect_aids

    def ious(self, gid, box):
        isect_aids = self.overlapping_aids(gid, box)
        if len(isect_aids):
            boxes1 = box[None, :]
            boxes2 = [self.qtrees[gid].aid_to_tlbr[aid] for aid in isect_aids]
            boxes2 = kwil.Boxes(np.array(boxes2), 'tlbr')
            ious = boxes1.ious(boxes2)[0]
        else:
            ious = np.empty(0)
        return isect_aids, ious

    def iooas(self, gid, box):
        """
        Intersection over other's area

        Like iou, but non-symetric, returned number is a percentage of the
        other's  (groundtruth) area. This means we dont care how big the
        (negative) `box` is.
        """
        isect_aids = self.overlapping_aids(gid, box)
        if len(isect_aids):
            boxes1 = box[None, :]
            boxes2 = [self.qtrees[gid].aid_to_tlbr[aid] for aid in isect_aids]
            boxes2 = kwil.Boxes(np.array(boxes2), 'tlbr')
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
            window_size (ndarray): absolute window sizes.
            gids (List[int]): image-ids to generate negatives for,
                if not specified generates for all images.
            thresh (float): overlap area threshold as a percentage of
                the negative box size. When thresh=0.0, that means
                negatives cannot overlap any positive, when threh=1.0, there
                are no constrains on negative placement.
            exact (bool): if True, ensure that we generate exactly `num` boxes
            rng (np.random.RandomState): random number generator

        Example:
            >>> from ndsampler.isect_indexer import *
            >>> from ndsampler import toydata
            >>> import ndsampler
            >>> dset = ndsampler.CocoDataset(toydata.demodata_toy_dset())
            >>> self = FrameIntersectionIndex.from_coco(dset)
            >>> anchors = np.array([[.35, .15], [.2, .2], [.1, .1]])
            >>> #num = 25
            >>> num = 5
            >>> rng = kwil.ensure_rng(None)
            >>> neg_gids, neg_boxes = self.random_negatives(
            >>>     num, anchors, gids=[1], rng=rng, thresh=0.01, exact=1)
            >>> # xdoc: +REQUIRES(--show)
            >>> gid = sorted(set(neg_gids))[0]
            >>> boxes = neg_boxes.compress(neg_gids == gid)
            >>> kwil.autompl()
            >>> img = kwil.imread(dset.imgs[gid]['file_name'])
            >>> kwil.imshow(img, doclf=True, fnum=1, colorspace='bgr')
            >>> support = self._support(gid)
            >>> kwil.draw_boxes(support, color='blue')
            >>> kwil.draw_boxes(boxes, color='orange')

        Example:
            >>> from ndsampler.isect_indexer import *
            >>> from ndsampler import toydata
            >>> import ndsampler
            >>> dset = ndsampler.CocoDataset(toydata.demodata_toy_dset())
            >>> self = FrameIntersectionIndex.from_coco(dset)
            >>> #num = 25
            >>> num = 5
            >>> rng = kwil.ensure_rng(None)
            >>> window_size = (50, 50)
            >>> neg_gids, neg_boxes = self.random_negatives(
            >>>     num, window_size=window_size, gids=[1], rng=rng,
            >>>     thresh=0.01, exact=1)
            >>> # xdoc: +REQUIRES(--show)
            >>> gid = sorted(set(neg_gids))[0]
            >>> boxes = neg_boxes.compress(neg_gids == gid)
            >>> kwil.autompl()
            >>> img = kwil.imread(dset.imgs[gid]['file_name'])
            >>> kwil.imshow(img, doclf=True, fnum=1, colorspace='bgr')
            >>> support = self._support(gid)
            >>> kwil.draw_boxes(support, color='blue')
            >>> kwil.draw_boxes(boxes, color='orange')
        """

        if not ((window_size is None) ^ (anchors is None)):
            raise ValueError('window_size and anchors are mutually exclusive')

        rng = kwil.ensure_rng(rng)
        all_gids = self.all_gids if gids is None else gids

        def _generate_rel(n):
            # Generate n candidate boxes in the normalized 0-1 domain
            cand_boxes = kwil.Boxes.random(num=n, scale=1.0, format='tlbr',
                                              anchors=anchors, anchor_std=0,
                                              rng=rng)

            chosen_gids = np.array(sorted(rng.choice(all_gids, size=n)))
            gid_to_boxes = kwil.group_items(cand_boxes, chosen_gids, axis=0)

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
                img_boxes = kwil.Boxes.random(
                    num=nboxes, scale=1.0, format='tlbr', anchors=anchors_,
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
        neg_boxes = kwil.Boxes(np.array(neg_boxes), 'tlbr')
        return neg_gids, neg_boxes

    def _debug_index(self):
        import kwil
        from shapely.ops import cascaded_union

        def _to_shapely(boxes):
            from shapely.geometry import Polygon
            from kwil.structs.boxes import _cat
            x1, y1, x2, y2 = boxes.to_tlbr(copy=False).components
            a = _cat([x1, y1]).tolist()
            b = _cat([x1, y2]).tolist()
            c = _cat([x2, y2]).tolist()
            d = _cat([x2, y1]).tolist()
            polygons = [Polygon(points) for points in zip(a, b, c, d, a)]
            return polygons

        for gid, qtree in self.qtrees.items():
            boxes = kwil.Boxes(np.array(list(qtree.aid_to_tlbr.values())), 'tlbr')
            polygons = _to_shapely(boxes)

            bounds = kwil.Boxes([[0, 0, qtree.width, qtree.height]], 'tlbr')
            bounds = _to_shapely(bounds)[0]
            merged_polygon = cascaded_union(polygons)
            uncovered = (bounds - merged_polygon)
            print('uncovered.area = {!r}'.format(uncovered.area))

            # plot these two polygons separately
            if 1:
                from descartes import PolygonPatch
                from matplotlib import pyplot as plt
                kwil.autompl()
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
        support_boxes = kwil.Boxes(list(qtree.aid_to_tlbr.values()), 'tlbr')

        return support_boxes
