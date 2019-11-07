Built with detection related tasks in mind, but also useful for classification
tasks.

The basic idea is to ensure your data is in MS-coco format, and then the
CocoSampler class will let you sample positive and negative regions.

For classification tasks the MS-COCO data could just be that every image has an
annotation that takes up the entire image.

# Features

* CocoDataset for managing and manipulating annotated image datasets
* Amortized O(1) sampling of N-dimension space-time data (e.g. images and video).
* Hierarchical or mutually exclusive category management.
* Random negative window sampling.
* Coverage-based positive sampling.
* Dynamic toydata generator.


# Example

This example shows how you can efficiently load subregions from images.

```python
>>> # Imagine you have some images
>>> import kwimage
>>> image_paths = [
>>>     kwimage.grab_test_image_fpath('astro'),
>>>     kwimage.grab_test_image_fpath('carl'),
>>>     kwimage.grab_test_image_fpath('airport'),
>>> ]  # xdoc: +IGNORE_WANT
['~/.cache/kwimage/demodata/KXhKM72.png',
 '~/.cache/kwimage/demodata/flTHWFD.png',
 '~/.cache/kwimage/demodata/Airport.jpg']
>>> # And you want to randomly load subregions of them in O(1) time
>>> import ndsampler
>>> # First make a COCO dataset that refers to your images (and possibly annotations)
>>> dataset = {
>>>     'images': [{'id': i, 'file_name': fpath} for i, fpath in enumerate(image_paths)],
>>>     'annotations': [],
>>>     'categories': [],
>>> }
>>> coco_dset = ndsampler.CocoDataset(dataset)
>>> print(coco_dset)
<CocoDataset(tag=None, n_anns=0, n_imgs=3, n_cats=0)>
>>> # Now pass the dataset to a sampler and tell it where it can store temporary files
>>> workdir = ub.ensure_app_cache_dir('ndsampler/demo')
>>> sampler = ndsampler.CocoSampler(coco_dset, workdir=workdir)
>>> # Now you can load arbirary samples by specifing a target dictionary
>>> # with an image_id (gid) center location (cx, cy) and width, height.
>>> target = {'gid': 0, 'cx': 200, 'cy': 200, 'width': 100, 'height': 100}
>>> sample = sampler.load_sample(target)
>>> # The sample contains the image data, any visible annotations, a reference
>>> # to the original target, and params of the transform used to sample this
>>> # patch
>>> print(sorted(sample.keys()))
['annots', 'im', 'params', 'tr']
>>> im = sample['im']
>>> print(im.shape)
(100, 100, 3)
>>> # The load sample function is at the core of what ndsampler does
>>> # There are other helper functions like load_positive / load_negative
>>> # which deal with annotations. See those for more details.
>>> # For random negative sampling see coco_regions.
```


# TODO:

- [ ] Currently only supports image-based detection tasks, but not much work is
  needed to extend to video. The code was originally based on sampling code for
  video, so ndimensions is builtin to most places in the code. However, there are
  currently no test cases that demonstrate that this library does work with video.
  So we should (a) port the video toydata code from irharn to test ndcases and (b)
  fix the code to work for both still images and video where things break. 

- [ ] Currently we are good at loading many small objects in 2d images.
  However, we are bad at loading images with one single large object that needs
  to be downsampled (e.g. loading an entire 1024x1024 image and downsampling it
  to 224x224). We should find a way to mitigate this.


# NOTES:
There is a GDAL backend for FramesSampler

Installing gdal is a pain though.

https://gist.github.com/cspanring/5680334


Using conda is relatively simple
```python
conda install gdal

# Test that this works
python -c "from osgeo import gdal; print(gdal)"
```


Also possible to use system packages
```python
# References:
# https://gis.stackexchange.com/questions/28966/python-gdal-package-missing-header-file-when-installing-via-pip
# https://gist.github.com/cspanring/5680334


# Install GDAL system libs
sudo apt install libgdal-dev

GDAL_VERSION=`gdal-config --version`
echo "GDAL_VERSION = $GDAL_VERSION" 
pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==$GDAL_VERSION


# Test that this works
python -c "from osgeo import gdal; print(gdal)"
```
