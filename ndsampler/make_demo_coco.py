import ndsampler
import ubelt as ub
from os.path import join


def dump_demodata():
    """
    Dummy data for tests and demos
    """
    dpath = ub.ensure_app_cache_dir('coco-demo')
    dset = ndsampler.CocoDataset.demo('shapes256')
    dset.fpath = join(dpath, dset.tag + '.mscoco.json')
    dset.dump(dset.fpath, newlines=True)
    print('dset.fpath = {!r}'.format(dset.fpath))
    return dset.fpath

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/ndsampler/ndsampler/make_demo_coco.py
    """
    dump_demodata()
