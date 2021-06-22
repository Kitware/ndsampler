ndsampler
=========

|GitlabCIPipeline| |GitlabCICoverage| |Pypi| |Downloads| 


+------------------+---------------------------------------------------------+
| Read the docs    | https://ndsampler.readthedocs.io                        |
+------------------+---------------------------------------------------------+
| Gitlab (main)    | https://gitlab.kitware.com/computer-vision/ndsampler    |
+------------------+---------------------------------------------------------+
| Github (mirror)  | https://github.com/Kitware/ndsampler                    |
+------------------+---------------------------------------------------------+
| Pypi             | https://pypi.org/project/ndsampler                      |
+------------------+---------------------------------------------------------+

The main webpage for this project is: https://gitlab.kitware.com/computer-vision/ndsampler

Fast random access to small regions in large images. 

Random access is amortized by converting images into an efficient backend
format (current backends include cloud-optimized geotiffs (cog) or numpy array
files (npy)). If images are already in COG format, then no conversion is
needed.

The ndsampler module was built with detection, segmentation, and classification
tasks in mind, but it is not limited to these use cases.

The basic idea is to ensure your data is in MS-coco format, and then the
CocoSampler class will let you sample positive and negative regions.

For classification tasks the MS-COCO data could just be that every image has an
annotation that takes up the entire image.

Installation
------------

The `ndsampler <https://pypi.org/project/ndsampler/>`_.  package can be installed via pip:

.. code-block:: bash

    pip install ndsampler


Note that ndsampler depends on `kwimage <https://pypi.org/project/kwimage/>`_,
where there is a known compatibility issue between `opencv-python <https://pypi.org/project/opencv-python/>`_
and `opencv-python-headless <https://pypi.org/project/opencv-python-headless/>`_. Please ensure that one
or the other (but not both) are installed as well:

.. code-block:: bash

    pip install opencv-python-headless

    # OR

    pip install opencv-python


Lastly, to fully leverage ndsampler's features GDAL must be installed (although
much of ndsampler can work without it).  Kitware has a pypi index that hosts
GDAL wheels for linux systems, but other systems will need to find some way of
installing gdal (conda is safe choice).

.. code-block:: bash

    pip install --find-links https://girder.github.io/large_image_wheels GDAL


Features
--------

* CocoDataset for managing and manipulating annotated image datasets
* Amortized O(1) sampling of N-dimension space-time data (wrt to constant window size) (e.g. images and video).
* Hierarchical or mutually exclusive category management.
* Random negative window sampling.
* Coverage-based positive sampling.
* Dynamic toydata generator.


Also installs the kwcoco package and CLI tool.


Usage
-----

The main pattern of usage is: 

1. Use kwcoco to load a json-based COCO dataset (or create a ``kwcoco.CocoDataset``
   programatically).

2. Pass that dataset to an ``ndsampler.CocoSampler`` object, and that
   effectively wraps the json structure that holds your images and annotations
   and it allows you to sample patches from those images efficiently. 

3. You can either manually specify image + region or you can specify an
   annotation id, in which case it loads the region corresponding to the
   annotation.   


Example
--------

This example shows how you can efficiently load subregions from images.

.. code-block:: python

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

A Note On COGs
--------------
COGs (cloud optimized geotiffs) are the backbone efficient sampling in the
ndsampler library. 

To preform deep learning efficiently you need to be able to effectively
randomly sample cropped regions from images, so when ``ndsampler.Sampler``
(more acurately the ``FramesSampler`` belonging to the base ``Sampler`` object)
is in "cog" mode, it caches all images larger than 512x512 in cog format. 

I've noticed a significant speedups even for "small" 1024x1024 images.  I
haven't made effective use of the overviews feature yet, but in the future I
plan to, as I want to allow ndsampler to sample in scale as well as in space. 

Its possible to obtain this speedup with the "npy" backend, which supports true
random sampling, but this is an uncompressed format, which can require a large
amount of disk space. Using the "None" backend, means that loading a small
windowed region requires loading the entire image first (which can be ok for
some applications).

Using COGs requires that GDAL is installed.  Installing GDAL is a pain though.

https://gist.github.com/cspanring/5680334

Using conda is relatively simple

.. code-block:: bash

    conda install gdal

    # Test that this works
    python -c "from osgeo import gdal; print(gdal)"


Also possible to use system packages

.. code-block:: bash

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


Kitware also has a pypi index that hosts GDAL wheels for linux systems:

.. code-block:: bash

    pip install --find-links https://girder.github.io/large_image_wheels GDAL


TODO
----

- [ ] Currently only supports image-based detection tasks, but not much work is
  needed to extend to video. The code was originally based on sampling code for
  video, so ndimensions is builtin to most places in the code. However, there are
  currently no test cases that demonstrate that this library does work with video.
  So we should (a) port the video toydata code from irharn to test ndcases and (b)
  fix the code to work for both still images and video where things break. 

- [ ] Currently we are good at loading many small objects in 2d images.
  However, we are bad at loading images with one single large object that needs
  to be downsampled (e.g. loading an entire 1024x1024 image and downsampling it
  to 224x224). We should find a way to mitigate this using pyramid overviews in
  the backend COG files.


.. |Pypi| image:: https://img.shields.io/pypi/v/ndsampler.svg
   :target: https://pypi.python.org/pypi/ndsampler

.. |Downloads| image:: https://img.shields.io/pypi/dm/ndsampler.svg
   :target: https://pypistats.org/packages/ndsampler

.. |ReadTheDocs| image:: https://readthedocs.org/projects/ndsampler/badge/?version=latest
    :target: http://ndsampler.readthedocs.io/en/latest/

.. # See: https://ci.appveyor.com/project/jon.crall/ndsampler/settings/badges
.. .. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/py3s2d6tyfjc8lm3/branch/master?svg=true
.. :target: https://ci.appveyor.com/project/jon.crall/ndsampler/branch/master

.. |GitlabCIPipeline| image:: https://gitlab.kitware.com/computer-vision/ndsampler/badges/master/pipeline.svg
   :target: https://gitlab.kitware.com/computer-vision/ndsampler/-/jobs

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/computer-vision/ndsampler/badges/master/coverage.svg?job=coverage
    :target: https://gitlab.kitware.com/computer-vision/ndsampler/commits/master
