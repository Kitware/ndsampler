import numpy as np
from os.path import normpath

from os.path import join
import ubelt as ub


def parse_mscoco():
    # Test that our implementation can handle the real mscoco data
    root = ub.expandpath('~/data/standard_datasets/mscoco/')

    fpath = join(root, 'annotations/instances_val2014.json')
    img_root = normpath(ub.ensuredir((root, 'images', 'val2014')))

    # fpath = join(root, 'annotations/stuff_val2017.json')
    # img_root = normpath(ub.ensuredir((root, 'images', 'val2017')))

    import ujson
    dataset = ujson.load(open(fpath, 'rb'))

    import ndsampler
    dset = ndsampler.CocoDataset(dataset)
    dset.img_root = img_root

    gid_iter = iter(dset.imgs.keys())

    gid = ub.peek(gid_iter)

    for gid in ub.ProgIter(gid_iter):
        img = dset.imgs[gid]
        ub.grabdata(img['coco_url'], dpath=img_root, verbose=0)
        anns = [dset.anns[aid] for aid in dset.gid_to_aids[gid]]
        dset.show_image(gid=gid)

    ann = anns[0]

    segmentation = ann['segmentation']

    from PIL import Image
    gpath = join(dset.img_root, img['file_name'])
    with Image.open(gpath) as pil_img:
        np_img = np.array(pil_img)
