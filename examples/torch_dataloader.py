"""
This demos how to get a speed improvement with ndsampler
"""
import torch
import kwcoco
import kwarray
import kwimage
import ndsampler
from torch.utils import data as torch_data
import numpy as np
import ubelt as ub


class DemoDataset(torch_data.Dataset):
    def __init__(self, datapath: str, backend=None, workdir=None):
        # Find and save all of the images in the data directory
        image_paths = []
        extensions = ("*.png", "*.jpg")
        for ext in extensions:
            image_paths.extend(ub.Path(datapath).glob("**/" + ext))

        self.image_paths = np.array(sorted(image_paths))

        # Create the sampler
        coco_dataset = {
            'images': [{'id': i, 'file_name': str(path)} for i, path in enumerate(image_paths)],
            'annotations': [],
            'categories': [],
        }

        coco_dset = kwcoco.CocoDataset(coco_dataset)
        print(f'Construct torch dataset for: coco_dset={coco_dset}')
        self.sampler = ndsampler.CocoSampler(coco_dset, backend=backend,
                                             workdir=workdir)
        self.sampler.frames.prepare(workers=8)
        print(f'self.sampler._backend={self.sampler._backend}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_index = np.random.randint(0, len(self.image_paths))

        random_chip_size = np.random.randint(296, 328)

        center_x = np.random.randint(random_chip_size / 2, 1920 - random_chip_size / 2)
        center_y = np.random.randint(random_chip_size / 2, 1080 - random_chip_size / 2)

        target = {'gid': image_index, 'cx': center_x, 'cy': center_y,
                  'width': random_chip_size, 'height': random_chip_size}

        chip = torch.from_numpy(self.sampler.load_sample(target)['im'])
        return chip


def demo():
    dpath = ub.Path.appdir('ndsampler/demo/torch_dataloader').ensuredir()
    image_dpath = dpath / 'images'
    workdir = dpath / 'workdir'

    rng = None
    rng = kwarray.ensure_rng(rng)

    demo_image_params = {
        'num_images': 64,
        'image_dsize': (1920, 1080),
    }
    stamp = ub.CacheStamp('demo_images', dpath=dpath, depends=demo_image_params)
    if stamp.expired():
        image_dpath.ensuredir()
        w, h = demo_image_params['image_dsize']
        for image_idx in ub.ProgIter(range(demo_image_params['num_images']), desc='write demo images'):
            imdata = kwimage.ensure_uint255(rng.rand(h, w, 3))
            image_name = f'demo_image_{image_idx:03d}.png'
            gpath = image_dpath / image_name
            kwimage.imwrite(gpath, imdata, backend='gdal')
        stamp.renew()

    print('')
    print('===============================')
    print('Test ndsampler with COG backend')
    print('Note the first time this is run, there is a conversion to COG '
          'which takes some time. Running a second time will be much faster')
    torch_dset = DemoDataset(image_dpath, backend='cog', workdir=workdir)
    loader = torch_data.DataLoader(torch_dset, batch_size=1)
    with ub.Timer('With COG') as cog_timer:
        for item in ub.ProgIter(loader, desc='loading data'):
            pass

    print('')
    print('===============================')
    print('Test ndsampler without COG backend')
    torch_dset = DemoDataset(image_dpath, backend=None)
    with ub.Timer('Without COG') as default_timer:
        loader = torch_data.DataLoader(torch_dset, batch_size=1)
        for item in ub.ProgIter(loader, desc='loading data'):
            pass

    print('')
    print('===============================')
    print('Final Results')
    print('===============================')
    print(f'default_timer.elapsed = {default_timer.elapsed}')
    print(f'cog_timer.elapsed     = {cog_timer.elapsed}')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/ndsampler/examples/torch_dataloader.py
    """
    demo()
