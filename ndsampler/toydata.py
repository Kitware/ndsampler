# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import glob
import numpy as np
import ubelt as ub
import kwil
import skimage
import skimage.morphology  # NOQA
import cv2
from ndsampler import abstract_sampler
from ndsampler import category_tree
import networkx as nx
from kwil.misc import fast_rand


class DynamicToySampler(abstract_sampler.AbstractSampler):
    """
    TODO: Generate new positive samples on the fly.

    Ignore:
        >>> from ndsampler.toydata import *
        >>> self = DynamicToySampler()
        >>> window_dims = (96, 96)

        img, anns = self.load_positive(window_dims=window_dims)
        kwil.autompl()
        kwil.imshow(img['imdata'])

        img, anns = self.load_negative(window_dims=window_dims)
        kwil.autompl()
        kwil.imshow(img['imdata'])

    CommandLine:
        xdoctest -m ndsampler.toydata DynamicToySampler --show

    Ignore:
        >>> # Test that this sampler works with the dataset
        >>> from ndsampler.toydata import *
        >>> from ovharn import mc_dataset
        >>> self = DynamicToySampler(1e3)
        >>> #dset = ov_dataset.OV_Dataset(self, window_dims=(96, 96), neg_to_pos_ratio=1, augment=True, training=True)
        >>> dset = mc_dataset.MC_Dataset(self, window_dims=(172, 172),
        >>>                              augment='complex', training=True)
        >>> dset.preselect()
        >>> dset[0]
        >>> dset[self.n_positives + 1]
        >>> # xdoctest: +REQUIRES(--show)
        >>> dset.display_datas(n=9, _raw_sseg=True)
        >>> kwil.show_if_requested()
    """

    def __init__(self, n_positives=1e5, seed=None, gsize=(416, 416)):
        self.categories = [
            # 'box',
            'circle',
            'star',
            'superstar',
            # 'octagon',
            # 'diamond'
        ]
        self.cname_to_cid = {
            cname: cid for cid, cname in enumerate(self.categories, start=1)
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

    def load_sample(self, tr, pad=None, window_dims=None):
        raise NotImplementedError

    def load_image(self, image_id=None, rng=None):
        img, anns = self.load_image_with_annots(image_id=image_id, rng=rng)
        return img['imdata']

    def load_image_with_annots(self, image_id=None, rng=None):
        """ Returns a random image and its annotations """
        if image_id is not None and self.seed is not None:
            rng = kwil.ensure_rng(self.seed * len(self) + image_id)
        rng = kwil.ensure_rng(rng)
        img, anns = demodata_toy_img(gsize=self._full_imgsize,
                                     categories=self.categories,
                                     rng=rng, n_annots=(0, 10))
        _node_to_id = self.catgraph.node_to_id
        img['id'] = int(rng.rand() * self._n_images)
        img['file_name'] = '{}.png'.format(img['id'])
        for ann in anns:
            ann['category_id'] = _node_to_id[ann['category_name']]
        return img, anns

    def load_positive(self, index=None, pad=None, window_dims=None, rng=None):
        """
        Note: window_dims is height / width
        """
        if index is not None and self.seed is not None:
            rng = self.seed * len(self) + index
        rng = kwil.ensure_rng(rng)

        if window_dims is None:
            window_dims = self._full_imgsize[::-1]

        gsize = np.array(window_dims[::-1])
        if pad is not None:
            gsize += 2 * np.array(pad[::-1])

        # guuid = ub.hash_data(rng)
        gid = int(rng.rand() * 500)

        img, anns = demodata_toy_img(gsize=gsize,
                                     rng=rng,
                                     categories=self.categories,
                                     centerobj='pos', n_annots=(1, 100))
        ann = anns[0]
        im = img['imdata']
        cname = ann['category_name']
        cid = self.cname_to_cid[cname]
        tr_ = {
            'aid': 1,
            # 'gid': guuid,
            # 'id': gid,
            'gid': gid,
            'rel_cx': gsize[0] / 2,
            'rel_cy': gsize[1] / 2,
            'cx': gsize[0] / 2,
            'cy': gsize[1] / 2,
            'width': ann['bbox'][2],
            'height': ann['bbox'][3],
            'category_id': cid,
        }
        annots = {
            'aids': np.arange(len(anns)),
            'cids': np.array([self.lookup_class_id(a['category_name']) for a in anns]),
            'rel_cxywh': kwil.Boxes([a['bbox'] for a in anns], 'xywh').to_cxywh().data,
        }

        # TODO: if window_dims was None, then crop the image to the size of the
        # annotation and remove non-visible other annots!
        sample = {'im': im, 'tr': tr_, 'annots': annots}
        return sample

    def load_negative(self, index=None, pad=None, window_dims=None, rng=None):

        if index is not None and self.seed is not None:
            rng = kwil.ensure_rng(self.seed * len(self) + index)
        rng = kwil.ensure_rng(rng)

        if window_dims is None:
            window_dims = self._full_imgsize[::-1]

        # guuid = ub.hash_data(rng)
        gid = int(rng.rand() * 500)

        gsize = np.array(window_dims[::-1])
        if pad is not None:
            gsize += 2 * np.array(pad[::-1])

        img, anns = demodata_toy_img(gsize=gsize,
                                     rng=rng,
                                     categories=self.categories,
                                     centerobj='neg', n_annots=(0, 10))
        im = img['imdata']
        cid = self.BACKGROUND_CLASS_ID
        annots = {
            'aids': np.arange(len(anns)),
            'cids': np.array([self.lookup_class_id(a['category_name']) for a in anns]),
            'rel_cxywh': kwil.Boxes([a['bbox'] for a in anns], 'xywh').to_cxywh().data,
        }
        tr_ = {
            'aid': -1,
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
        # TODO: if window_dims was None, then crop the image to the size of the
        # annotation and remove non-visible other annots!
        sample = {'im': im, 'tr': tr_, 'annots': annots}
        return sample


class Rasters:
    @staticmethod
    def superstar():
        """
        test data patch

        Ignore:
            >>> kwil.autompl()
            >>> data = np.clip(kwil.imscale(Rasters.star(), 2.2)[0], 0, 1)
            >>> kwil.imshow(data)
        """
        (_, i, O) = 0, 1.0, .5
        patch = np.array([
            [_, _, _, _, _, _, _, O, O, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, O, i, i, O, _, _, _, _, _, _],
            [_, _, _, _, _, _, O, i, i, O, _, _, _, _, _, _],
            [_, _, _, _, _, O, i, i, i, i, O, _, _, _, _, _],
            [O, O, O, O, O, O, i, i, i, i, O, O, O, O, O, O],
            [O, i, i, i, i, i, i, i, i, i, i, i, i, i, i, O],
            [_, O, i, i, i, i, O, i, i, O, i, i, i, i, O, _],
            [_, _, O, i, i, i, O, i, i, O, i, i, i, O, _, _],
            [_, _, _, O, i, i, O, i, i, O, i, i, O, _, _, _],
            [_, _, _, O, i, i, i, i, i, i, i, i, O, _, _, _],
            [_, _, O, i, i, i, i, i, i, i, i, i, i, O, _, _],
            [_, _, O, i, i, i, i, i, i, i, i, i, i, O, _, _],
            [_, O, i, i, i, i, i, O, O, i, i, i, i, i, O, _],
            [_, O, i, i, i, O, O, _, _, O, O, i, i, i, O, _],
            [O, i, i, O, O, _, _, _, _, _, _, O, O, i, i, O],
            [O, O, O, _, _, _, _, _, _, _, _, _, _, O, O, O]])
        return patch


class CategoryPatterns:
    def __init__(self, categories=None, fg_scale=0.5, fg_intensity=0.9,
                 rng=None):
        self.rng = kwil.ensure_rng(rng)
        self.fg_scale = fg_scale
        self.fg_intensity = fg_intensity
        self.category_to_elemfunc = {
            'box': skimage.morphology.square,
            'star': skimage.morphology.star,
            'circle': skimage.morphology.disk,
            'superstar': lambda x: Rasters.superstar(),
            'octagon': lambda x: skimage.morphology.octagon(x // 2, int(x / (2 * np.sqrt(2)))),
            'diamond': skimage.morphology.diamond,
        }
        # Make generation of shapes a bit faster?
        # Maybe there are too many input combinations for this?
        # If we only allow certain size generations it should be ok

        # for key in self.category_to_elemfunc.keys():
        #     self.category_to_elemfunc[key] = ub.memoize(self.category_to_elemfunc[key])

        if categories is None:
            self.categories = list(self.category_to_elemfunc.keys())
        else:
            self.categories = categories

        self._catlist = sorted(self.categories)

    def random_category(self, chip):
        name = self.rng.choice(self._catlist)
        elem_func = self.category_to_elemfunc[name]
        return name, self._from_elem(elem_func, chip)

    def _from_elem(self, elem_func, chip):
        x = max(chip.shape[0:2])
        # x = int(2 ** np.floor(np.log2(x)))

        size = tuple(map(int, chip.shape[0:2][::-1]))
        elem = elem_func(x)
        mask = cv2.resize(elem, size).astype(np.float32)
        fg_intensity = np.float32(self.fg_intensity)
        fg_scale = np.float32(self.fg_scale)
        fgdata = fast_rand.standard_normal(chip.shape, std=fg_scale,
                                           mean=fg_intensity, rng=self.rng,
                                           dtype=np.float32)
        fgdata = np.clip(fgdata , 0, 1, out=fgdata)
        fga = kwil.ensure_alpha_channel(fgdata, alpha=mask)
        data = kwil.overlay_alpha_images(fga, chip, keepalpha=False)
        return data


@kwil.profile
def demodata_toy_img(anchors=None, gsize=(104, 104), categories=None,
                     n_annots=(0, 50), fg_scale=0.5, bg_scale=0.8,
                     bg_intensity=0.1,
                     fg_intensity=0.9,
                     centerobj=None, exact=False, rng=None):
    """
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
        >>> img, anns = demodata_toy_img(gsize=(32, 32), anchors=[[.3, .3]], rng=0)
        >>> img['imdata'] = '<ndarray shape={}>'.format(img['imdata'].shape)
        >>> print('img = {}'.format(ub.repr2(img)))
        >>> print('anns = {}'.format(ub.repr2(anns)))
        img = {
            'height': 32,
            'imdata': '<ndarray shape=(32, 32, 3)>',
            'width': 32,
        }
        anns = [
            {'bbox': [15, 10, 9, 8], 'category_name': 'superstar'},
            {'bbox': [11, 20, 7, 7], 'category_name': 'octagon'},
            {'bbox': [4, 4, 8, 6], 'category_name': 'superstar'},
            {'bbox': [3, 20, 6, 7], 'category_name': 'superstar'},
        ]

    Example:
        >>> # xdoctest: +REQUIRES(--show)
        >>> img, anns = demodata_toy_img(gsize=(172, 172), rng=None)
        >>> print('anns = {}'.format(ub.repr2(anns, nl=1)))
        >>> kwil.autompl()
        >>> kwil.imshow(img['imdata'])
        >>> kwil.show_if_requested()

    Ignore:
        from ndsampler.toydata import *
        import xinspect
        globals().update(xinspect.get_kwargs(demodata_toy_img))

    """
    if anchors is None:
        anchors = [[.20, .20]]
    anchors = np.asarray(anchors)

    rng = kwil.ensure_rng(rng)
    catpats = CategoryPatterns(categories, fg_scale=fg_scale,
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
        boxes = kwil.Boxes.random(
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

    tlbr_data = boxes.to_tlbr().data

    try:
        keep = kwil.non_max_supression(
            tlbr_data, scores=box_priority, thresh=0.0, impl='cpu')
    except KeyError:
        import warnings
        warnings.warn('missing cpu impl, trying numpy')
        keep = kwil.non_max_supression(
            tlbr_data, scores=box_priority, thresh=0.0, impl='py')

    boxes = boxes[keep]

    if centerobj == 'neg':
        # The center of the image should be negative so remove the center box
        boxes = boxes[1:]

    boxes = boxes.scale(.8).translate(.1 * min(gsize))
    boxes.data = boxes.data.astype(np.int)

    gw, gh = gsize

    # This is 2x as fast for gsize=(300,300)
    gshape = (gh, gw, 1)
    imdata = fast_rand.standard_normal(gshape, mean=bg_intensity, std=bg_scale,
                                       rng=rng, dtype=np.float32)
    np.clip(imdata, 0, 1, out=imdata)

    catnames = []
    for tl_x, tl_y, br_x, br_y in boxes.to_tlbr().data:
        chip_index = tuple([slice(tl_y, br_y), slice(tl_x, br_x)])
        chip = imdata[chip_index]
        catname, fgdata = catpats.random_category(chip)
        catnames.append(catname)
        imdata[tl_y:br_y, tl_x:br_x, :] = fgdata.mean(axis=2, keepdims=True)

    if 0:
        imdata.mean(axis=2, out=imdata[:, :, 0])
        imdata[:, :, 1] = imdata[:, :, 0]
        imdata[:, :, 2] = imdata[:, :, 0]

    imdata = (imdata * 255).astype(np.uint8)
    imdata = kwil.atleast_3channels(imdata)

    img = {
        'width': gw,
        'height': gh,
        'imdata': imdata,
    }
    anns = []
    for catname, bbox in zip(catnames, boxes.to_xywh().data):
        anns.append({
            'category_name': catname,
            'bbox': bbox.tolist(),
        })
    return img, anns


def demodata_toy_dset(gsize=(600, 600), n_imgs=5):
    """
    Create a toy detection problem

    Returns:
        dict: dataset in mscoco format

    Ignore:
        >>> from ndsampler.toydata import *
        >>> dataset = demodata_toy_dset(gsize=(10, 10))
        >>> dpath = ub.ensure_app_cache_dir('ndsampler', 'toy_dset')
        >>> ub.startfile(dpath)
    """
    dpath = ub.ensure_app_cache_dir('ndsampler', 'toy_dset')

    # gsize = (104, 104)
    # gsize = (300, 300)
    # gsize = (600, 600)

    rng = np.random.RandomState(0)

    categories = [
        # 'box',
        'circle',
        'star',
        'superstar',
        # 'octagon',
        # 'diamond'
    ]

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
        'categories': categories,
    }
    cacher = ub.Cacher('toy_dset', dpath=ub.ensuredir(dpath, 'cache'),
                       cfgstr=ub.repr2(cfg), verbose=3)

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
        for i, name in enumerate(categories, start=1):
            dataset['categories'].append({
                'id': i,
                'name': name,
            })
            name_to_cid[name] = i

        for i in ub.ProgIter(range(n_imgs), label='creating data'):
            img, anns = demodata_toy_img(anchors, gsize=gsize,
                                         categories=categories,
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

            kwil.imwrite(fpath, imdata)

        cacher.enabled = True
        cacher.save(dataset)

    return dataset
