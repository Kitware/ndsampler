"""
This demos how to get a speed improvement with ndsampler.

On an NVMe drive I can get a 4x speedup on the default benchmark.
"""
import torch
import kwcoco
import kwarray
import kwimage
import ndsampler
from torch.utils import data as torch_data
import numpy as np
import ubelt as ub
import scriptconfig as scfg


@scfg.dataconf
class DemoConfig:
    """
    This is the default configuration for the demo. You can modify it with
    command line parameters to change things like the number of images, image
    sizes, and the path to where the demo happens (HDDs will perform
    differently than SSDs! This filesystem also matters!)
    """
    dpath = scfg.Value(None, help=ub.paragraph(
        '''
        The path to where the demo writes its data.
        Defaults to the ndsmapler application cache directory
        '''))

    image_format = scfg.Value('png', choices=['png', 'jpg', 'cog'], help=ub.paragraph(
        '''
        The format the initial demo images will be written as.
        This will impact multiple parts of the benchmark, particularly
        initial cache generation.
        '''))

    num_images = scfg.Value(32, help='number of demo images in the dataset')

    image_width = scfg.Value(1920, help='The demo image width')
    image_height = scfg.Value(1080, help='The demo image height')

    num_passes = scfg.Value(10, help=ub.paragraph(
        '''
        Number of dataloading passes to perform. The first COG backend
        pass will be slower as ndsampler builds its data structures, but
        the rest will faster.
        '''))


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
    config = DemoConfig.cli()
    if config['dpath'] is None:
        config['dpath'] = dpath = ub.Path.appdir('ndsampler/demo/torch_dataloader')
    else:
        dpath = ub.Path(config['dpath'])

    print('config = {}'.format(ub.repr2(config.asdict(), nl=1)))

    dpath.ensuredir()
    workdir = dpath / 'workdir'

    num_passes = config['num_passes']

    # We could seed the rng if we wanted to.
    rng = None
    rng = kwarray.ensure_rng(rng)
    demo_image_params = {
        'num_images': config['num_images'],
        'image_dsize': (config['image_width'], config['image_height']),
        'image_format': config['image_format'],
    }
    image_dpath = dpath / 'images_{image_format}'.format(**demo_image_params)
    stamp = ub.CacheStamp('demo_images', dpath=dpath, depends=demo_image_params)
    if stamp.expired():
        ext = config['image_format']
        imwrite_backend = 'auto'
        if ext == 'cog':
            ext = 'tif'
            imwrite_backend = 'gdal'
        image_dpath.ensuredir()
        w, h = demo_image_params['image_dsize']
        for image_idx in ub.ProgIter(range(demo_image_params['num_images']), desc='write demo images'):
            imdata = kwimage.ensure_uint255(rng.rand(h, w, 3))
            image_name = f'demo_image_{image_idx:03d}.{ext}'
            gpath = image_dpath / image_name
            kwimage.imwrite(gpath, imdata, backend=imwrite_backend)
        stamp.renew()

    print('')
    print('===============================')
    print('Test ndsampler with COG backend')
    print('Note the first time this is run, there is a conversion to COG '
          'which takes some time. Running a second time will be much faster')
    with ub.Timer('Sampler Construction, backend=COG') as cog_construct:
        torch_dset = DemoDataset(image_dpath, backend='cog', workdir=workdir)
    loader = torch_data.DataLoader(torch_dset, batch_size=1)
    with ub.Timer('Load Data, backend=COG') as cog_timer:
        # Make N passes through the dataloader
        for _ in range(num_passes):
            for item in ub.ProgIter(loader, desc='loading data'):
                pass

    print('')
    print('===============================')
    print('Test ndsampler without COG backend')
    with ub.Timer('Sampler Construction, backend=None') as nocog_construct:
        torch_dset = DemoDataset(image_dpath, backend=None)
    with ub.Timer('Load Data, backend=None') as nocog_timer:
        # Make N passes through the dataloader
        for _ in range(num_passes):
            loader = torch_data.DataLoader(torch_dset, batch_size=1)
            for item in ub.ProgIter(loader, desc='loading data'):
                pass

    print('')
    print('===============================')
    print('Final Results')
    print('===============================')
    print(f'nocog_construct.elapsed = {nocog_construct.elapsed}')
    print(f'cog_construct.elapsed   = {cog_construct.elapsed}')
    print(f'nocog_timer.elapsed = {nocog_timer.elapsed}')
    print(f'cog_timer.elapsed   = {cog_timer.elapsed}')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/ndsampler/examples/torch_dataloader.py
    """
    demo()
