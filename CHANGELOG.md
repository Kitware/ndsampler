# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

## Version 0.7.5 - Unreleased

### Added
* `load_sample` now takes an `annot_ids` argument to skip the spatial lookup
  and use something precomputed.

* Can now specify `optimize=False` in `load_sample` target dictionary to
  prevent delayed image from doing any optimization.


## Version 0.7.4 - Released 2023-03-05

### Added
* Experimental support for `rtree` in `isect_indexer`
* Add `verbose_ndsample` option to the target dictionary for debugging 

### Changed
* Small speedups


## Version 0.7.3 - Released 2023-02-02

### Added
* Add CocoSampler.coerce

### Fixed:
* Work around issue in delayed image where it is not handling interpolation
  correctly.
* Fixed np.int, np.float, np.bool


## Version 0.7.2 - Released 2022-09-28


## Version 0.7.1 - Released 2022-09-16

### Fixed
* The annotations are now returned in the same space as the sample in all cases
  except for the jagged case, which is still not fully developed.


## Version 0.7.0 - Released 2022-08-11

### Changed

* We are renaming the "tr" variable to "target". 
* Removing some old backup sampling methods.
* Removed old "ndsampler.delayed" logic in favor of kwcoco's version


## Version 0.6.10 - Released 2022-07-26

### Added
* Ability to specify `use_native_scale` in the tr dict, which returns jagged data at its original resolution
* Ability to specify `scale` which augments the scale of the data.


## Version 0.6.9 - Released 2022-07-19


## Version 0.6.8 - Released 2022-07-18

### Changed
* Using the new kwcoco delayed operation backend when sampling


## Version 0.6.7 - Released 2022-03-31

### Added
* Environment variable `NDSAMPLER_DISABLE_OPTIONAL_WARNINGS` can disable warnings

* tr can now take parameters interpolation and antialias which are now passed
  to kwcoco delayed finalize.


### Changed
* Setting nodata to "float" or "auto" makes nan the default pad value.
* `new_sample_grid` can now take extra kwargs that are passed to the video or
  image sample grid function. Added a `use_annots` flag that can disable the use 
  of annotations in constructing the grid.
* `new_sample_grid` now returns a list "targets" that contains all targets and
   a list of positive and negative indexes that indicate which of them are
   positive or negative if `use_annots` is specified. The old "positives" and
   "negatives" items in the return dict still exist but are deprecated and will be
   removed.


## Version 0.6.6 - Released 2022-01-11

### Added
* Add `dtype` argument to `load_sample`, which will control the type of data returned.
* Add `nodata` argument to `load_sample`, which will handle nodata values for integer images.


### Changed
* using the experimental loader is now the faster method (given kwcoco >
  0.2.13) and has been set as the default, and the other method has been removed.


## Version 0.6.5 - Released 2021-09-14

### Fixed

* Bug where the video sample grid did not respect "video-space"



## Version 0.6.4 - Released 2021-06-22


## Version 0.6.3 - Released 2021-05-20


## Version 0.6.2 - Released 2021-05-19


## Version 0.6.2 - Released 2021-05-19

### Added
* The target dictionary can now contain `space_slice`, `time_slice` or `gids` 


## Version 0.6.1 - Released 2021-05-13

### Added
* The target dictionary in `CocoSampler.load_sample` can now accept a `vidid`
  and will return a 3D sample.

### Changed 
* New version will kwarray >= 0.5.16
* The CocoSampler now defaults to backend=None instead of backend="auto".
* The `load_sample` signature was modified and deprecated args were removed.


## Version 0.6.0 - Released 2021-04-22


### Added
* The target dictionary in `CocoSampler.load_sample` can now accept a `channels`
  to grab specified channels in multispectral images.


### Removed

* `_hack_old_names` from `AbstractFrames`
* `CocoDataset` which was moved to `kwcoco`
* `toypatterns` which was moved to `kwcoco`
* `stats_dict`, which now exists in `kwarray`
* `ndsampler.util` module, use `ndsampler.utils.util_misc` instead.
* `ndsampler.util_futures`, which now lives in `ndsampler.utils.util_futures`
* `make_demo_coco`, use `kwcoco toydata` instead.

*  `Frames._lookup_hashid` 
*  `Frames._lookup_gpath` 
*  `Frames._gnames` 

### Modified

* Overhaul of `AbstractFrames` for better auxiliary data integration


## Version 0.5.13 - Released 2021-01-26

### Changed
* Removed Python 3.5 support
* No longer using protocol 2 in the Cacher
* Better `LazyGDalFrameFile.demo` classmethod.

### Fixed
* Bug in accessing the LRU

## Version 0.5.11 - Released 2020-08-26

### Fixed
* Minor compatibility fixes for `ndsampler.CategoryTree` and `kwcoco.CategoryTree`


## Version 0.5.10 - Released 2020-06-25

### Added
* Can now call `CocoSampler.load_item` with target dictionary only containing
  an annotation id.

### Changed
* CategoryTree now uses a mixin class to separate class storage from hierarchy
  computations.

### Fixed
* In rare instances a floating point error would cause `load_sample` to raise
  an `AssertionError`. This is now checked for and corrected.


## Version 0.5.9 - Released 2020-05-13

### Fixed
* removed use of deprecated sklearn externals


## Version 0.5.8 - Released 2020-05-01 

### Fixed:
* `CategoryTree.decision` now always refines categories with a single parent
* `run_tests.py` now correctly returns exit code.

### Added
* `batch_convert_to_cog` to `util_gdal`
* `batch_validate_cog` to `util_gdal`


## Version 0.5.6 - Released 2020-08-26

### Added
* `util_lru` containing implementations of a dictionary based LRU cache. 


### Changed
* AbstractFrames now uses the configurable `LRUDict` instead of the `functools` decorator.
* Moved toydata from ndsampler to kwcoco


## Version 0.5.5

### Changed
* Moved `ndsampler.util_futures` and `ndsampler.util` into `ndsampler.utils`.
  Old ways of accessing these files will raise deprecation warnings.


### Fixed:
* Using a better stratified split algorithm
* Fixed issue with prepare and the npy backend when too many files were opened


### Added

* Added `boxsize_stats` to `CocoDataset`.

## Version 0.5.4 - Released 2020-02-19 

### Fixed

* `SerialFuture` was broken in python 3.8
* Fixed incorrect use of the word "logit", what I was calling logits are
  actually log probabilities.

### Added

* `_ensure_imgsize` can now use threads


### Changed

* `CocoDataset.union` now preserves image ids when possible. 


## Version 0.5.3 - Released 2020-02-03

## Version 0.5.2 - Released 2020-01-31

### Added
* `CocoDataset.rebase` for changing the image root.


### Changed
* `CocoDataset.remove_` methods now have a safe kwarg to prevent bad things from happening when there is a duplicate inputs.
* `AbstractFrames` cog backend now supports "auto" compression, which chooses a sensible compression value depending on the type of input data. 


## Version 0.5.1


### Changed
* `CocoDataset` now allows for `category_id` to be `None`.
* `_lookup` now uses the for `lookup` function.
* `_ilookup` is deprecated. 


## Version 0.5.0

### Changed
* First public release


## Version 0.3.0


### Added

* Can now remove keypoint categories (implementation is not optimized)

### Changed
* `CategoryTree` now serializes graph node attributes
* `backend` default is now `auto` instead of `None`. Setting `backend=None` now disables backend caching which might cause the entire image to be loaded in order to load any region.

### Fixed
* Negative regions now have correct aspect ratios.
* Coco union now properly handles basic keypoint categories. 
* `rename_categories` now handles different merge policies.
* The `_next_ids` attribute was incorrectly shallow copied on deep copy.



## Version 0.2.0 - [Finalized 2019 June 9th]

### Added
* Add `CocoDataset.load_annot_sample`, which helps load a chip for a specific annotation. (Note: this is less efficient than using the CocoSampler)
* Add `Regions.get_item`
* GDAL COG-backend for CocoFrames
* Add `coerce_data` helper module for loading train/vali splits
* `CocoDataset.remove_images`
* `CocoDataset._alias_to_cat` looks up categories via the alias foreign key
* Sampler has support for new coco json keypoint structure. 
* Frames now has `prepare` method which safely and quickly prepares the cached representation if needed.
* Add experimental method `CocoDataset._resolve_to_cat` for quick category lookup, may be aliased to lookup category in the future.
* Rename categories now supports merge cases

### Changed
* Annotations no longer require a category id, although it is still recommended.
* Visibility thresh now default to 0.
* Sampler `load` methods now accept `with_annots` flag.
* `id_to_idx` now behaves as a property.
* Removed redundant items from `CategoryTree` serialization in `__json__` and `__getstate__`.
* COG format is now defaulted over memmap frames backend
* Using scalars for `load_sample`'s pad, will now broadcast to appropriate dimensions.
* Can now poss `window_dims='extent'` to `load_sample`.
* The on-disk caches in `CocoRegions` can now be explicitly disabled.
* The on-disk caches in `CocoRegions` now write to `<workdir>/_cache/_coco_regions`
* `CocoDataset.demo` now can accept 'shapes' or 'photos' as a key
* Using new-style coco keypoints by default.

### Fixed
* Fixes for larger images (maybe need to do something else besides memmap)
* Fixes for kwimage 0.4.0 api changes
* Fixed issue in region caches when using python 2 and 3
* CocoDataset union now respects `img_root` that is common to all operands.


## Version 0.1.0

### Added
* Added better support for keypoints and segmentations, and possible extensions

### Removed
* Removed support for "fine-ids"

### Changed
* The CocoDataset "key" now defaults to the full basename instead of only the first part.
* Renamed arguments to `add_image`, `add_annotation`, and `add_category` to be consistent with data internals (BREAKING)


## Version 0.0.4

### Added
* Add a hacky fix for a race condition in AbstractFrames. We should try to fix this more robustly.
* Add `index` to CategoryTree
* Add `ignore_class_idxs` to CategoryTree.decision
* Add `always_refine_idxs` to CategoryTree.decision

### Changed
* Rework construction of CocoDataset.hashid
* Invalidate the CocoDataset.hashid on modification
* CategoryTree now prefers coerce over cast

### Fixed
* Fix issue with modifying CocoDataset subsets
* Fixed CategoryTree.decision when the tree is rooted
* Fix torch incompatibility in CategoryTree.decision


## Version 0.0.2

### Fixed
* Minor test fixes


## Version 0.0.1

### Added
* Initial code for 2D O(1) chip sampling
* Initial code for CategoryTree
* Initial code for CocoDataset
* Initial code for dummy detection toydata
