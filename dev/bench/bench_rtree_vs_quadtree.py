"""
Notes on spatial index structures

https://stackoverflow.com/questions/41946007/efficient-and-well-explained-implementation-of-a-quadtree-for-2d-collision-det

https://github.com/Elan456/pyquadtree

https://informix-spatial-technology.blogspot.com/2012/01/comparison-on-b-tree-r-tree-quad-tree.html

TODO:
    Look into CDAL: https://github.com/CGAL/cgal-swig-bindings/wiki/Package_wrappers_available
"""
import kwimage
import rtree
import pyqtree
import numpy as np
import ubelt as ub
import kwarray

try:
    from line_profiler import profile
except Exception:
    profile = ub.identity


class RTreeWrapper():
    """
    Wrapper around pure python libspatialindex rtree implementation:

    https://pypi.org/project/Rtree/
    https://rtree.readthedocs.io/en/latest/tutorial.html#using-rtree-as-a-cheapo-spatial-database
    https://libspatialindex.org/en/latest/
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._module = pyqtree
        self._tree = rtree.index.Index(bounds=[0, 0, width, height])
        self.aid_to_ltrb = {}

    @profile
    def add_annotation(self, annot_id, ltrb_box):
        _tree = self._tree
        _tree.insert(annot_id, ltrb_box, obj=annot_id)
        self.aid_to_ltrb[annot_id] = ltrb_box

    @profile
    def query(self, ltrb_box):
        isect_aids = sorted(set(self._tree.intersection(ltrb_box)))
        return isect_aids


class PyQTreeWrapper():
    """
    Wrapper around pure python QTree implementation:
    https://pypi.org/project/Pyqtree/
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._module = pyqtree
        self._tree = pyqtree.Index((0, 0, width, height))
        self.aid_to_ltrb = {}

    @profile
    def add_annotation(self, annot_id, ltrb_box):
        self._tree.insert(annot_id, ltrb_box)
        self.aid_to_ltrb[annot_id] = ltrb_box

    @profile
    def query(self, ltrb_box):
        isect_aids = sorted(set(self._tree.intersect(ltrb_box)))
        return isect_aids


class MultiFrameTree:
    def __init__(self, backend):
        if backend == 'rtree':
            self._cls = RTreeWrapper
        elif backend == 'pyqtree':
            self._cls = PyQTreeWrapper
        else:
            raise KeyError(backend)
        self.image_id_to_tree = {}

    def add_image(self, image_id, width, height):
        tree = self._cls(width, height)
        self.image_id_to_tree[image_id] = tree
        return tree

    def __getitem__(self, image_id):
        tree = self.image_id_to_tree[image_id]
        return tree


def generate_demo_data(basis_item):
    """
    Generate a random-ish set of images with boxes and a set of query
    boxes for each of those images.
    """
    num_images = basis_item['num_images'] = basis_item.get('num_images', 100)
    annots_per_image = basis_item['annots_per_image'] = basis_item.get('annots_per_image', 100)
    window_scale = basis_item['window_scale'] = basis_item.get('window_scale', 1 / 8)
    image_dim = basis_item['image_dim'] = basis_item.get('image_dim', 4_000)

    width = image_dim
    height = image_dim

    image_to_info = {}
    max_annot_id = 0
    for image_id in range(num_images):
        boxes = kwimage.Boxes.random(annots_per_image).scale((width, height)).round()
        # Make boxes spatially smaller to simulate lots of tiny objects in a
        # large image
        boxes = boxes.scale(0.1, about='centroid')

        first_annot_id = max_annot_id + 1
        last_annot_id = first_annot_id + annots_per_image - 1
        annot_ids = np.arange(first_annot_id, last_annot_id + 1)

        query_slices = list(kwarray.SlidingWindow(
            shape=(width, height),
            window=(int(width * window_scale),
                    int(height * window_scale)),
            keepbound=True, allow_overshoot=True))
        query_boxes = kwimage.Boxes.concatenate([
            kwimage.Boxes.from_slice(s) for s in query_slices])
        image_to_info[image_id] = {
            'width': width,
            'height': height,
            'boxes': boxes,
            'annot_ids': annot_ids,
            'query_boxes': query_boxes,
            'num_images': num_images,
            'width': width,
            'height': height,
            'annots_per_image': annots_per_image,
        }
        max_annot_id = last_annot_id
    return image_to_info


def run_tests(image_to_info):
    backends = {}
    backends['rtree'] = MultiFrameTree(backend='rtree')
    backends['pyqtree'] = MultiFrameTree(backend='pyqtree')

    measures = []
    real_results = {}

    for backend_name, backend in backends.items():

        with ub.Timer() as build_timer:
            for image_id, info in image_to_info.items():
                tree = backend.add_image(image_id, info['width'], info['height'])
                boxes = info['boxes'].to_ltrb()
                annot_ids = info['annot_ids']
                for annot_id, ltrb_box in zip(annot_ids, boxes.data):
                    tree.add_annotation(annot_id,  ltrb_box)

        with ub.Timer() as query_timer:
            impl_results = []
            for image_id, info in image_to_info.items():
                boxes = info['query_boxes'].to_ltrb()
                tree = backend[image_id]
                for query_box in boxes:
                    found = tree.query(ltrb_box)
                    impl_results.append(found)
            real_results[backend_name] = impl_results

        measure = {}
        measure['build_time'] = build_timer.elapsed
        measure['query_time'] = query_timer.elapsed
        measure['backend_name'] = backend_name
        measures.append(measure)
    assert real_results['rtree'] == real_results['pyqtree']
    return measures


def main():
    basis = list(ub.named_product({
        'num_images': [
            1,
            100,
            500,
            1_000,
            # 5_000, 10_000, 25_000, 50_000
        ],
        'annots_per_image': [
            100,
            1000
        ],
        'window_scale': [
            1 / 8,
            1 / 16
        ],
    }))
    all_measures = []
    for basis_item in ub.ProgIter(basis, verbose=3, desc='Measure Times'):
        print(f'basis_item = {ub.urepr(basis_item, nl=1)}')
        image_to_info = generate_demo_data(basis_item)
        measures = run_tests(image_to_info)
        for m in measures:
            m.update(basis_item)
        print(f'measures = {ub.urepr(measures, nl=1)}')
        all_measures.extend(measures)

    import pandas as pd
    import kwplot
    sns = kwplot.autosns()
    df = pd.DataFrame(all_measures)
    ax = kwplot.figure(fnum=1, doclf=True, pnum=(1, 2, 1)).gca()

    sns.lineplot(data=df,
                 x='num_images', y='build_time',
                 hue='backend_name',
                 style='annots_per_image',
                 ax=ax)
    ax.set_title('Build Time')

    ax = kwplot.figure(fnum=1, docla=True, pnum=(1, 2, 2)).gca()
    sns.lineplot(data=df,
                 x='num_images', y='query_time',
                 hue='backend_name',
                 # style='annots_per_image',
                 style='window_scale',
                 ax=ax)
    ax.set_title('Query Time')

    kwplot.show_if_requested()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/ndsampler/dev/bench/bench_rtree_vs_quadtree.py --show
        LINE_PROFILE=1 python ~/code/ndsampler/dev/bench/bench_rtree_vs_quadtree.py
    """
    main()
