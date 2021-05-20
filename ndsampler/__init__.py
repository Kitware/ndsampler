"""
mkinit ~/code/ndsampler/ndsampler/__init__.py --diff
mkinit ~/code/ndsampler/ndsampler/__init__.py -w
"""
__version__ = '0.6.4'

from ndsampler.utils.util_misc import (HashIdentifiable,)

__explicit__ = [
    'HashIdentifiable',
]

#  ---

__submodules__ = {
        'abstract_frames': ['Frames', 'SimpleFrames'],
        'abstract_sampler': None,
        'category_tree': None,
        'coco_frames': None,
        'coco_regions': None,
        'coco_sampler': None,
        'isect_indexer': None,

        # DEPRECATED (moved to kwcoco)
        'toydata': None,
}
from ndsampler import abstract_frames
from ndsampler import abstract_sampler
from ndsampler import category_tree
from ndsampler import coco_frames
from ndsampler import coco_regions
from ndsampler import coco_sampler
from ndsampler import isect_indexer
from ndsampler import toydata

from ndsampler.abstract_frames import (Frames, SimpleFrames,)
from ndsampler.abstract_sampler import (AbstractSampler,)
from ndsampler.category_tree import (CategoryTree,)
from ndsampler.coco_frames import (CocoFrames,)
from ndsampler.coco_regions import (CocoRegions, MissingNegativePool, Targets,
                                    select_positive_regions,
                                    tabular_coco_targets,)
from ndsampler.coco_sampler import (CocoSampler,)
from ndsampler.isect_indexer import (FrameIntersectionIndex,)
from ndsampler.toydata import (DynamicToySampler,)

__all__ = ['AbstractSampler', 'CategoryTree', 'CocoFrames', 'CocoRegions',
           'CocoSampler', 'DynamicToySampler', 'FrameIntersectionIndex',
           'Frames', 'HashIdentifiable', 'MissingNegativePool', 'SimpleFrames',
           'Targets', 'abstract_frames', 'abstract_sampler', 'category_tree',
           'coco_frames', 'coco_regions', 'coco_sampler', 'isect_indexer',
           'select_positive_regions', 'tabular_coco_targets',
           'toydata']
