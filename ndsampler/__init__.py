"""
mkinit ~/code/ndsampler/ndsampler/__init__.py -w
"""
__version__ = '0.5.11'

from ndsampler.utils.util_misc import (HashIdentifiable, stats_dict,)

__explicit__ = [
    'HashIdentifiable',
    'stats_dict',
]

#  ---

__submodules__ = [
        'abstract_frames',
        'abstract_sampler',
        'category_tree',
        'coco_frames',
        'coco_regions',
        'coco_sampler',
        'isect_indexer',

        # DEPRECATED (moved to kwcoco)
        'toydata',
        'toypatterns',
        'coco_dataset',
]
from ndsampler import abstract_frames
from ndsampler import abstract_sampler
from ndsampler import category_tree
from ndsampler import coco_frames
from ndsampler import coco_regions
from ndsampler import coco_sampler
from ndsampler import isect_indexer
from ndsampler import toydata
from ndsampler import toypatterns
from ndsampler import coco_dataset

from ndsampler.abstract_frames import (CorruptCOG, DEBUG_COG_ATOMIC_WRITE,
                                       DEBUG_FILE_LOCK_CACHE_WRITE,
                                       DEBUG_LOAD_COG, Frames,
                                       RUN_COG_CORRUPTION_CHECKS,
                                       SimpleFrames,)
from ndsampler.abstract_sampler import (AbstractSampler,)
from ndsampler.category_tree import (CategoryTree,)
from ndsampler.coco_frames import (CocoFrames,)
from ndsampler.coco_regions import (CocoRegions, MissingNegativePool, Targets,
                                    select_positive_regions,
                                    tabular_coco_targets,)
from ndsampler.coco_sampler import (CocoSampler, padded_slice,)
from ndsampler.isect_indexer import (FrameIntersectionIndex,)
from ndsampler.toydata import (DynamicToySampler,)
from ndsampler.coco_dataset import (CocoDataset,)

__all__ = ['AbstractSampler', 'CategoryTree', 'CocoDataset', 'CocoFrames',
           'CocoRegions', 'CocoSampler', 'CorruptCOG',
           'DEBUG_COG_ATOMIC_WRITE', 'DEBUG_FILE_LOCK_CACHE_WRITE',
           'DEBUG_LOAD_COG', 'DynamicToySampler', 'FrameIntersectionIndex',
           'Frames', 'HashIdentifiable', 'MissingNegativePool',
           'RUN_COG_CORRUPTION_CHECKS', 'SimpleFrames', 'Targets',
           'abstract_frames', 'abstract_sampler', 'category_tree',
           'coco_dataset', 'coco_frames', 'coco_regions', 'coco_sampler',
           'isect_indexer', 'padded_slice', 'select_positive_regions',
           'stats_dict', 'tabular_coco_targets', 'toydata', 'toypatterns']
