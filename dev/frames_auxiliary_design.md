# ND-Sampler Frames Design Document

The purpose of this document is to 

1. Lay out the existing structure of `ndsampler.AbstractFrames` and 

2. Brainstorm design changes to `ndsampler.AbstractFrames` such that auxiliary
   images (e.g. from kwcoco) are better supported


We will discuss this from the perspective of generic `Frames` and `Sampler`
objects even though they may correspond to abstract or coco-specific
implementations. 



### Current State of the Public API

A `Sampler` object has the following methods of interest:


* `load_image(self, image_id, cache=True)`

* `load_sample(self, tr, ...)`


A `Frames` object has the following methods of interest:

* `load_region(self, image_id, region=None, bands=None, scale=0)`

* `load_frame(self, image_id)`

* `load_image(self, image_id, bands=None, scale=0, cache=True)`


We need to maintain this API as best as possible 


### Relevant Frames Private API


* `_lookup_hashid(self, image_id)`


* `_lookup_gpath(self, image_id)` -- 


* `self.image_ids` -- attribute


* `self.image_ids` -- attribute


* `_gnames(self, image_id, mode=None)` -- 


### Target state


* Looking up the hashid needs to change depending on the file / band in
  question.

* If multiple bands are requested, whos job is it to resample and align them?
  Probably the sampler itself and not the frames? The frames should probably
  just serve the data?

* Instead of looking up a single image path, we change `_lookup_gpath` to
  lookup a dictionary containing all of the relevant info in a mapped state?
  Would this include the hashid?


Note that a column-based data structure is going to be much faster and probably
easier to convert to SQL? But might lack expressiveness.



Relevant columns:
```
{
    'primary_fpaths':  [fpath1, fpath2, ...],
    'auxiliary_fpaths':  [fpath1, fpath2, ...],
}
```

Can we write it in a row-based way and then easily switch to column based?


Row based is probably easier to write.

Ultimately we will need to have a database backend strategy if the product ever
needs to scale, which should be a goal.



* The `_lookup_hashid` function isn't exactly relevant anymore. Or it needs an
  overhaul


* Got basic way to query channels in `Frames.load_region`. Now need to allow
`load_sample` to get the appropriate regions from each channel and have those
resampled such that the targets are aligned.




#### On Caching:

 * Caching need to happen at the level of each file path (i.e. id+channel-name)


    """
    I've updated ndsampler such that it's now aware of the multispectral
    information in a kwcoco file. I also have some proof-of-concept logic
    written such that given a bounding box specified in the "base image's"
    pixel coordinates, it can sample the corresponding region from all other
    channels and do the appropriate interpolated upsampling to align the
    channels with subpixel accuracy. What I'm working on now is exposing that
    API.

    What I want to know is

    For a particular image, what is the most natural way for ndsampler to pass to you the multispectral information?

    1. Do you want a concatenated array of aligned - i.e. prefused - data?

    2. Do you want a list containing the (potentially) upscaled data for each channel?

    3. Do you want a list of dictionaries where each one contains the cropped data for each channel at its native resolution (which may include padding on either side) and the transformation that would upscale it to the aligned canvas?

    I think I'll have to implement a combination (1 or 2) and 3


    I've updated ndsampler so it's now aware of the multispectral information in a kwcoco file.

    """



Use case:

In a dataloader we want to load some specific channels for a specific image.
These channels will likely be the same over multiple images. We probably have
some "constant" data loader variable that contains the channels of interest.

Given an image-id, a region, and a list of channels of interest we should we
able to efficiently load the subregions of each channel and align them as
necessary.


```
tr = {
    'gid': 100,
    'region': (slice(100, 200), slice(200, 300)),
    'channels': <List[str] | str | None>,
    'fused': True | False,
}
sample = sampler.load_sample(tr)
sample['im']
```


## DECIDED CHANGELOG

### Removed:

*  `_lookup_hashid` 
*  `_lookup_gpath` 
*  `_gnames` 


### Modified:

* Frame cache names are now using a new hashed directory structure format. Old
  cache names will be invalidated. Please delete these old caches.


### DVC Hash Strategy

* Check if the dataset is in a dvc rep
* Find the DVC cache directory
* For files, check if the file with a .dvc extension exists, if so the hash can
  be found there.

* For files where it doesn't exist, check if a parent directory has a .dvc
  neighbor. In this case the hashid points to a json file in the cache, 
  where we can load the md5 for each relpath.


### Key-Value Database notes:

* https://github.com/RaRe-Technologies/sqlitedict

* https://pypi.org/project/diskcache/

* https://docs.python.org/3/library/dbm.html
