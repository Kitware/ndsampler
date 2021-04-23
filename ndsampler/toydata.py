# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ndsampler import abstract_sampler
from ndsampler import category_tree
import networkx as nx
import kwarray
import kwimage

# Moved to kwcoco
from kwcoco.demo.toypatterns import CategoryPatterns
from kwcoco.demo.toydata import demodata_toy_img, demodata_toy_dset  # NOQA


class DynamicToySampler(abstract_sampler.AbstractSampler):
    """
    Generates positive and negative samples on the fly.

    Note:
        Its probably more robust to generate a static fixed-size dataset with
        'demodata_toy_dset' or `kwcoco.CocoDataset.demo`. However, if you
        need a sampler that dynamically generates toydata, this is for you.

    Ignore:
        >>> from ndsampler.toydata import *
        >>> self = DynamicToySampler()
        >>> window_dims = (96, 96)

        img, anns = self.load_positive(window_dims=window_dims)
        kwplot.autompl()
        kwplot.imshow(img['imdata'])

        img, anns = self.load_negative(window_dims=window_dims)
        kwplot.autompl()
        kwplot.imshow(img['imdata'])

    CommandLine:
        xdoctest -m ndsampler.toydata DynamicToySampler --show

    Example:
        >>> # Test that this sampler works with the dataset
        >>> from ndsampler.toydata import *
        >>> self = DynamicToySampler(1e3)
        >>> imgs = [self.load_positive()['im'] for _ in range(9)]
        >>> # xdoctest: +REQUIRES(--show)
        >>> stacked = kwimage.stack_images_grid(imgs, overlap=-10)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(stacked)
        >>> kwplot.show_if_requested()
    """

    def __init__(self, n_positives=1e5, seed=None, gsize=(416, 416),
                 categories=None):

        self.catpats = CategoryPatterns.coerce(categories)
        self.kp_catnames = self.catpats.kp_catnames

        self.cname_to_cid = {
            cat['name']: cat['id'] for cat in self.catpats
        }
        self.cname_to_cid['background'] = 0
        self.cid_to_cname = {
            cid: cname for cname, cid in self.cname_to_cid.items()
        }

        graph = nx.DiGraph()
        graph.add_nodes_from(self.cname_to_cid.keys())
        nx.set_node_attributes(graph, self.cname_to_cid, name='id')
        self.catgraph = category_tree.CategoryTree(graph)

        self.BACKGROUND_CLASS_ID = self.catgraph.node_to_id['background']

        self._full_imgsize = gsize
        self._n_positives = int(n_positives)
        self._n_images = 50
        self.seed = seed

        self.gray = True
        self._n_annots_pos = (1, 100)
        self._n_annots_neg = (0, 10)
        # self.catpats = CategoryPatterns.coerce(self.catpats)
        # fg_scale=fg_scale, fg_intensity=fg_intensity, rng=rng)

    def load_item(self, index, pad=None, window_dims=None):
        """
        Loads from positives and then negatives.
        """
        if index < self._n_positives:
            sample = self.load_positive(index, pad=pad, window_dims=window_dims)
        else:
            index = index - self._n_positives
            sample = self.load_negative(index, pad=pad, window_dims=window_dims)
        return sample

    def __len__(self):
        return self._n_positives * 2

    def _depends(self):
        return [self._n_positives]

    @property
    def class_ids(self):
        return list(self.cid_to_cname.keys())

    @property
    def n_positives(self):
        return self._n_positives

    @property
    def n_annots(self):
        raise NotImplementedError

    @property
    def n_images(self):
        raise NotImplementedError

    def image_ids(self):
        # All images are dummy images
        return list(range(10))

    def lookup_class_name(self, class_id):
        return self.cid_to_cname[class_id]

    def lookup_class_id(self, class_name):
        return self.cname_to_cid[class_name]

    def _lookup_kpnames(self, class_id):
        cname = self.lookup_class_name(class_id)
        return self.catpats.cname_to_kp[cname]
        # if cname is not None:
        #     return ['center']

    @property
    def n_categories(self):
        return len(self.catpats) + 1

    def preselect(self, n_pos=None, n_neg=None, neg_to_pos_ratio=None,
                  window_dims=None, rng=None, verbose=0):
        n_pos = self._n_positives
        if n_neg is None:
            n_neg = int(n_pos * neg_to_pos_ratio)
        if verbose:
            print('no need to presample dynamic sampler')
        return n_pos, n_neg

    def load_image(self, image_id=None, rng=None):
        img, anns = self.load_image_with_annots(image_id=image_id, rng=rng)
        return img['imdata']

    def load_image_with_annots(self, image_id=None, rng=None):
        """ Returns a random image and its annotations """
        if image_id is not None and self.seed is not None:
            rng = kwarray.ensure_rng(self.seed * len(self) + image_id)
        rng = kwarray.ensure_rng(rng)
        img, anns = demodata_toy_img(gsize=self._full_imgsize,
                                     categories=self.catpats,
                                     gray=self.gray,
                                     rng=rng, n_annots=(0, 10))
        _node_to_id = self.catgraph.node_to_id
        img['id'] = int(rng.rand() * self._n_images)
        img['file_name'] = '{}.png'.format(img['id'])
        for ann in anns:
            ann['category_id'] = _node_to_id[ann['category_name']]
        return img, anns

    def load_sample(self, tr, pad=None, window_dims=None):
        raise NotImplementedError

    def _load_toy_sample(self, window_dims, pad, rng, centerobj, n_annots):
        rng = kwarray.ensure_rng(rng)

        gid = int(rng.rand() * 500)
        # guuid = ub.hash_data(rng)

        if window_dims is None:
            window_dims = self._full_imgsize[::-1]

        gsize = np.array(window_dims[::-1])
        if pad is not None:
            gsize += 2 * np.array(pad[::-1])

        img, anns = demodata_toy_img(gsize=gsize,
                                     rng=rng,
                                     categories=self.catpats,
                                     gray=self.gray,
                                     centerobj=centerobj, n_annots=n_annots)
        im = img['imdata']
        if centerobj == 'neg':
            cid = self.BACKGROUND_CLASS_ID
            aid = -1
        else:
            ann = anns[0]
            cname = ann['category_name']
            cid = self.cname_to_cid[cname]
            aid = 1

        tr_ = {
            'aid': aid,
            # 'gid': guuid,
            # 'id': gid,
            'gid': gid,
            'rel_cx': gsize[0] / 2,
            'rel_cy': gsize[1] / 2,
            'cx': gsize[0] / 2,
            'cy': gsize[1] / 2,
            'width': window_dims[1],
            'height': window_dims[0],
            'category_id': cid,
        }

        # Handle segmentations and keypoints if they exist
        sseg_list = []
        kpts_list = []

        kp_catnames = self.catpats.kp_catnames
        for ann in anns:
            coco_sseg = ann.get('segmentation', None)
            coco_kpts = ann.get('keypoints', None)
            if coco_kpts is not None:
                cid = self.lookup_class_id(ann['category_name'])
                # kpnames = self._lookup_kpnames(cid)
                rel_points = kwimage.Points._from_coco(coco_kpts,
                                                       classes=kp_catnames)
                # coco_xyf = np.array(coco_kpts).reshape(-1, 3)
                # flags = (coco_xyf.T[2] > 0)
                # xy_pts = coco_xyf[flags, 0:2]
                # kpnames = list(ub.compress(kpnames, flags))
                # kp_class_idxs = np.array([kp_catnames.index(n) for n in kpnames])
                # rel_points = kwimage.Points(xy=xy_pts,
                #                             class_idxs=kp_class_idxs,
                #                             classes=kp_catnames)
            else:
                rel_points = None

            if coco_sseg is not None:
                # TODO: implement MultiPolygon coerce instead
                data_dims = gsize[::-1]
                abs_sseg = kwimage.Mask.coerce(coco_sseg, dims=data_dims)
                rel_sseg = abs_sseg.to_multi_polygon()
            else:
                rel_sseg = None

            kpts_list.append(rel_points)
            sseg_list.append(rel_sseg)

        rel_ssegs = kwimage.PolygonList(sseg_list)
        rel_kpts = kwimage.PolygonList(kpts_list)
        rel_kpts.meta['classes'] = self.catpats.kp_catnames

        rel_boxes = kwimage.Boxes([a['bbox'] for a in anns], 'xywh').to_cxywh()

        annots = {
            'aids': np.arange(len(anns)),
            'cids': np.array([self.lookup_class_id(a['category_name']) for a in anns]),
            'rel_cxywh': rel_boxes.data,

            'rel_boxes': rel_boxes,
            'rel_kpts': rel_kpts,
            'rel_ssegs': rel_ssegs,
        }

        # TODO: if window_dims was None, then crop the image to the size of the
        # annotation and remove non-visible other annots!
        sample = {'im': im, 'tr': tr_, 'annots': annots}
        return sample

    def load_positive(self, index=None, pad=None, window_dims=None, rng=None):
        """
        Note: window_dims is height / width

        Example:
            >>> from ndsampler.toydata import *
            >>> self = DynamicToySampler(1e2)
            >>> sample = self.load_positive()
            >>> annots = sample['annots']
            >>> assert len(annots['aids']) > 0
            >>> assert len(annots['rel_cxywh']) == len(annots['aids'])
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> # Draw box in relative sample context
            >>> kwplot.imshow(sample['im'], pnum=(1, 1, 1), fnum=1)
            >>> annots['rel_boxes'].translate([-.5, -.5]).draw()
            >>> annots['rel_ssegs'].draw(color='red', alpha=.6)
            >>> annots['rel_kpts'].draw(color='green', alpha=.8, radius=4)
        """
        if index is not None and self.seed is not None:
            rng = self.seed * len(self) + index

        sample = self._load_toy_sample(window_dims, pad, rng,
                                       centerobj='pos',
                                       n_annots=self._n_annots_pos)
        return sample

    def load_negative(self, index=None, pad=None, window_dims=None, rng=None):

        if index is not None and self.seed is not None:
            rng = kwarray.ensure_rng(self.seed * len(self) + index)

        sample = self._load_toy_sample(window_dims, pad, rng,
                                       centerobj='neg',
                                       n_annots=self._n_annots_neg)
        return sample
