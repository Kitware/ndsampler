# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

## Version 0.5.13 - Unreleased


## Version 0.5.12 - Released 2021-01-26

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
