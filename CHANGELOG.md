# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.


## Version 0.2.0

### Added
* GDAL COG-backend for CocoFrames
* `CocoDataset.remove_images`

### Changed
* COG format is now defaulted over memmap frames backend

### Fixed
* Fixes for larger images (maybe need to do something else besides memmap)
* Fixes for kwimage 0.4.0 api changes


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
* added `index` to CategoryTree
* Add ignore_class_idxs to CategoryTree.decision
* Add always_refine_idxs to CategoryTree.decision

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
