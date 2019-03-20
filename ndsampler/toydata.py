# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import glob
import numpy as np
import ubelt as ub
import kwarray
import kwimage
import skimage
import skimage.morphology  # NOQA
from ndsampler import abstract_sampler
from ndsampler import category_tree
from ndsampler.toypatterns import CategoryPatterns
import networkx as nx


class DynamicToySampler(abstract_sampler.AbstractSampler):
    """
    Generates positive and negative samples on the fly.

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

    def __init__(self, n_positives=1e5, seed=None, gsize=(416, 416), categories=None):
        self.categories = CategoryPatterns.coerce(categories)

        self.cname_to_cid = {}
        self.cname_to_cid = {
            cat['name']: cid for cid, cat in enumerate(self.categories, start=1)
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
        # self.catpats = CategoryPatterns.coerce(self.categories)
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
        return self.categories.cname_to_kp[cname]
        # if cname is not None:
        #     return ['center']

    @property
    def n_categories(self):
        return len(self.categories) + 1

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
                                     categories=self.categories,
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
                                     categories=self.categories,
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

        kp_catnames = self.categories.kp_catnames
        for ann in anns:
            coco_sseg = ann.get('segmentation', None)
            coco_kpts = ann.get('keypoints', None)
            if coco_kpts is not None:
                cid = self.lookup_class_id(ann['category_name'])
                kpnames = self._lookup_kpnames(cid)
                coco_xyf = np.array(coco_kpts).reshape(-1, 3)
                flags = (coco_xyf.T[2] > 0)
                xy_pts = coco_xyf[flags, 0:2]
                kpnames = list(ub.compress(kpnames, flags))
                kp_class_idxs = np.array([kp_catnames.index(n) for n in kpnames])
                rel_points = kwimage.Points(xy=xy_pts,
                                            class_idxs=kp_class_idxs,
                                            classes=kp_catnames)
                # rel_points = abs_points.translate(offset)
            else:
                rel_points = None

            if coco_sseg is not None:
                # TODO: implement MultiPolygon coerce instead
                data_dims = gsize[::-1]
                abs_sseg = kwimage.Mask.coerce(coco_sseg, shape=data_dims)
                rel_sseg = abs_sseg.to_multi_polygon()
            else:
                rel_sseg = None

            kpts_list.append(rel_points)
            sseg_list.append(rel_sseg)

        rel_ssegs = kwimage.PolygonList(sseg_list)
        rel_kpts = kwimage.PolygonList(kpts_list)
        rel_kpts.meta['classes'] = self.categories.kp_catnames

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
                                       centerobj='pos', n_annots=(1, 100))
        return sample

    def load_negative(self, index=None, pad=None, window_dims=None, rng=None):

        if index is not None and self.seed is not None:
            rng = kwarray.ensure_rng(self.seed * len(self) + index)

        sample = self._load_toy_sample(window_dims, pad, rng,
                                       centerobj='neg', n_annots=(0, 10))
        return sample


def demodata_toy_img(anchors=None, gsize=(104, 104), categories=None,
                     n_annots=(0, 50), fg_scale=0.5, bg_scale=0.8,
                     bg_intensity=0.1, fg_intensity=0.9,
                     gray=True, centerobj=None, exact=False, rng=None):
    r"""
    Generate a single image with non-overlapping toy objects of available
    categories.

    Args:
        anchors (ndarray): Nx2 base width / height of boxes

        gsize (Tuple[int, int]): width / height of the image

        categories (List[str]): list of category names

        n_annots (Tuple | int): controls how many annotations are in the image.
            if it is a tuple, then it is interpreted as uniform random bounds

        fg_scale (float): standard deviation of foreground intensity

        bg_scale (float): standard deviation of background intensity

        bg_intensity (float): mean of background intensity

        fg_intensity (float): mean of foreground intensity

        centerobj (bool): if 'pos', then the first annotation will be in the
            center of the image, if 'neg', then no annotations will be in the
            center.

        exact (bool): if True, ensures that exactly the number of specified
            annots are generated.

        rng (RandomState): the random state used to seed the process

    CommandLine:
        xdoctest -m ndsampler.toydata demodata_toy_img:0 --profile
        xdoctest -m ndsampler.toydata demodata_toy_img:1 --show

    Example:
        >>> from ndsampler.toydata import *  # NOQA
        >>> img, anns = demodata_toy_img(gsize=(32, 32), anchors=[[.3, .3]], rng=0)
        >>> img['imdata'] = '<ndarray shape={}>'.format(img['imdata'].shape)
        >>> print('img = {}'.format(ub.repr2(img)))
        >>> print('anns = {}'.format(ub.repr2(anns, nl=2, cbr=True)))
        >>> # xdoctest: +IGNORE_WANT
        img = {
            'height': 32,
            'imdata': '<ndarray shape=(32, 32, 3)>',
            'width': 32,
        }
        anns = [{'bbox': [15, 10, 9, 8],
          'category_name': 'star',
          'keypoints': [19.5, 14.0, 2],
          'segmentation': {'counts': '[`06j0000O20N1000e8', 'size': [32, 32]},},
         {'bbox': [11, 20, 7, 7],
          'category_name': 'eff',
          'keypoints': [14.5, 23.5, 2],
          'segmentation': {'counts': 'd<7i0O1LWO0j00VOOk012O]=', 'size': [32, 32]},},
         {'bbox': [4, 4, 8, 6],
          'category_name': 'star',
          'keypoints': [8.0, 7.0, 2],
          'segmentation': {'counts': 'U54l00O2O01N10kd0', 'size': [32, 32]},},
         {'bbox': [3, 20, 6, 7],
          'category_name': 'star',
          'keypoints': [6.0, 23.5, 2],
          'segmentation': {'counts': 'g31m04N000002L[f0', 'size': [32, 32]},},]

    Example:
        >>> # xdoctest: +REQUIRES(--show)
        >>> img, anns = demodata_toy_img(gsize=(172, 172), rng=None)
        >>> print('anns = {}'.format(ub.repr2(anns, nl=1)))
        >>> kwplot.autompl()
        >>> kwplot.imshow(img['imdata'])
        >>> kwplot.show_if_requested()

    Ignore:
        from ndsampler.toydata import *
        import xinspect
        globals().update(xinspect.get_kwargs(demodata_toy_img))

    """
    if anchors is None:
        anchors = [[.20, .20]]
    anchors = np.asarray(anchors)

    rng = kwarray.ensure_rng(rng)
    # if isinstance(categories, CategoryPatterns) or hasattr(categories, 'obj_catnames'):
    #     catpats = categories
    # else:
    catpats = CategoryPatterns.coerce(categories, fg_scale=fg_scale,
                                      fg_intensity=fg_intensity, rng=rng)

    if n_annots is None:
        n_annots = (0, 50)

    if isinstance(n_annots, tuple):
        num = rng.randint(*n_annots)
    else:
        num = n_annots

    assert centerobj in {None, 'pos', 'neg'}
    if exact:
        raise NotImplementedError

    while True:
        boxes = kwimage.Boxes.random(
            num=num, scale=1.0, format='xywh', rng=rng, anchors=anchors)
        boxes = boxes.scale(gsize)
        bw, bh = boxes.components[2:4]
        ar = np.maximum(bw, bh) / np.minimum(bw, bh)
        flags = ((bw > 1) & (bh > 1) & (ar < 4))
        boxes = boxes[flags.ravel()]

        if centerobj != 'pos' or len(boxes):
            # Ensure we generate at least one box when centerobj is true
            # TODO: if an exact number of boxes is specified, we
            # should ensure that that number is generated.
            break

    if centerobj:
        if centerobj == 'pos':
            assert len(boxes) > 0, 'oops, need to enforce at least one'
        if len(boxes) > 0:
            # Force the first box to be in the center
            cxywh = boxes.to_cxywh()
            cxywh.data[0, 0:2] = np.array(gsize) / 2
            boxes = cxywh.to_tlbr()

    # Make sure the first box is always kept.
    box_priority = np.arange(boxes.shape[0])[::-1].astype(np.float32)
    boxes.ious(boxes)

    if len(boxes) > 1:
        tlbr_data = boxes.to_tlbr().data
        try:
            keep = kwimage.non_max_supression(
                tlbr_data, scores=box_priority, thresh=0.0, impl='cpu')
        except KeyError:
            import warnings
            warnings.warn('missing cpu impl, trying numpy')
            keep = kwimage.non_max_supression(
                tlbr_data, scores=box_priority, thresh=0.0, impl='py')
        boxes = boxes[keep]

    if centerobj == 'neg':
        # The center of the image should be negative so remove the center box
        boxes = boxes[1:]

    boxes = boxes.scale(.8).translate(.1 * min(gsize))
    boxes.data = boxes.data.astype(np.int)

    # Hack away zero width objects
    boxes = boxes.to_xywh(copy=False)
    boxes.data[..., 2:4] = np.maximum(boxes.data[..., 2:4], 1)

    gw, gh = gsize
    dims = (gh, gw)

    # This is 2x as fast for gsize=(300,300)
    if gray:
        gshape = (gh, gw, 1)
        imdata = kwarray.standard_normal(gshape, mean=bg_intensity, std=bg_scale,
                                           rng=rng, dtype=np.float32)
    else:
        gshape = (gh, gw, 3)
        # imdata = kwarray.standard_normal(gshape, mean=bg_intensity, std=bg_scale,
        #                                    rng=rng, dtype=np.float32)
        # hack because 3 channels is slower
        imdata = kwarray.uniform(0, 1, gshape, rng=rng, dtype=np.float32)

    np.clip(imdata, 0, 1, out=imdata)

    catnames = []

    tlbr_boxes = boxes.to_tlbr().data
    xywh_boxes = boxes.to_xywh().data.tolist()

    # Construct coco-style annotation dictionaries
    anns = []
    for tlbr, xywh in zip(tlbr_boxes, xywh_boxes):
        tl_x, tl_y, br_x, br_y = tlbr
        chip_index = tuple([slice(tl_y, br_y), slice(tl_x, br_x)])
        chip = imdata[chip_index]
        xy_offset = (tl_x, tl_y)
        info = catpats.random_category(chip, xy_offset, dims)
        fgdata = info['data']
        if gray:
            fgdata = fgdata.mean(axis=2, keepdims=True)

        catnames.append(info['name'])
        imdata[tl_y:br_y, tl_x:br_x, :] = fgdata
        ann = {
            'category_name': info['name'],
            'segmentation': info['segmentation'],
            'keypoints': info['keypoints'],
            'bbox': xywh,
        }
        anns.append(ann)

    if 0:
        imdata.mean(axis=2, out=imdata[:, :, 0])
        imdata[:, :, 1] = imdata[:, :, 0]
        imdata[:, :, 2] = imdata[:, :, 0]

    imdata = (imdata * 255).astype(np.uint8)
    imdata = kwimage.atleast_3channels(imdata)

    img = {
        'width': gw,
        'height': gh,
        'imdata': imdata,
    }
    return img, anns


def demodata_toy_dset(gsize=(600, 600), n_imgs=5):
    """
    Create a toy detection problem

    Returns:
        dict: dataset in mscoco format

    Example:
        >>> from ndsampler.toydata import *
        >>> import ndsampler
        >>> dataset = demodata_toy_dset(gsize=(300, 300))
        >>> dpath = ub.ensure_app_cache_dir('ndsampler', 'toy_dset')
        >>> dset = ndsampler.CocoDataset(dataset)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> dset.show_image(gid=1)
        >>> ub.startfile(dpath)
    """
    dpath = ub.ensure_app_cache_dir('ndsampler', 'toy_dset')

    # gsize = (104, 104)
    # gsize = (300, 300)
    # gsize = (600, 600)

    rng = np.random.RandomState(0)

    catpats = CategoryPatterns.coerce([
        # 'box',
        # 'circle',
        'star',
        'superstar',
        'eff',
        # 'octagon',
        # 'diamond'
    ])

    anchors = np.array([
        [1, 1], [2, 2], [1.5, 1], [2, 1], [3, 1], [3, 2], [2.5, 2.5],
    ])
    anchors = np.vstack([anchors, anchors[:, ::-1]])
    anchors = np.vstack([anchors, anchors * 1.5])
    # anchors = np.vstack([anchors, anchors * 2.0])
    anchors /= (anchors.max() * 3)
    anchors = np.array(sorted(set(map(tuple, anchors.tolist()))))

    cfg = {
        'anchors': anchors,
        'gsize': gsize,
        'n_imgs': n_imgs,
        'categories': catpats,
    }
    cacher = ub.Cacher('toy_dset_v3', dpath=ub.ensuredir(dpath, 'cache'),
                       cfgstr=ub.repr2(cfg), verbose=3, enabled=0)

    img_dpath = ub.ensuredir((dpath, 'imgs_{}_{}'.format(
        cfg['n_imgs'], cacher._condense_cfgstr())))

    n_have = len(list(glob.glob(join(img_dpath, '*.png'))))
    # Only allow cache loading if the data seems to exist
    cacher.enabled = n_have == n_imgs

    bg_intensity = .1
    fg_scale = 0.5
    bg_scale = 0.8

    dataset = cacher.tryload()
    if dataset is None:
        ub.delete(img_dpath)
        ub.ensuredir(img_dpath)
        dataset = {
            'images': [],
            'annotations': [],
            'categories': [],
        }

        dataset['categories'].append({
            'id': 0,
            'name': 'background',
        })

        name_to_cid = {}
        for i, cat in enumerate(catpats, start=1):
            dataset['categories'].append(cat)
            name_to_cid[cat['name']] = cat['id']

        for i in ub.ProgIter(range(n_imgs), label='creating data'):
            img, anns = demodata_toy_img(anchors, gsize=gsize,
                                         categories=catpats,
                                         fg_scale=fg_scale, bg_scale=bg_scale,
                                         bg_intensity=bg_intensity, rng=rng)
            imdata = img.pop('imdata')

            gid = len(dataset['images']) + 1
            fpath = join(img_dpath, 'img_{:05d}.png'.format(gid))
            img.update({
                'id': gid,
                'file_name': fpath
            })

            dataset['images'].append(img)
            for ann in anns:
                cid = name_to_cid[ann.pop('category_name')]
                ann.update({
                    'id': len(dataset['annotations']) + 1,
                    'image_id': gid,
                    'category_id': cid,
                })
                dataset['annotations'].append(ann)

            kwimage.imwrite(fpath, imdata)

        cacher.enabled = True
        cacher.save(dataset)

    return dataset
