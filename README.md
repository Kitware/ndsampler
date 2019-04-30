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
