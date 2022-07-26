import ubelt as ub


class HashIdentifiable(object):
    """
    A class is hash-identifiable if its invariants can be tied to a specific
    list of hashable dependencies.

    The inheriting class must either:
        * implement `_depends`
        * implement `_make_hashid`
        * define `_hashid`

    Example:
        class Base:
            def __init__(self):
                # commenting the next line removes cooperative inheritence
                super().__init__()
                self.base = 1

        class Derived(Base, HashIdentifiable):
            def __init__(self):
                super().__init__()
                self.defived = 1

        self = Derived()
        dir(self)
    """
    def __init__(self, **kwargs):
        super(HashIdentifiable, self).__init__(**kwargs)
        self._hashid = None
        self._hashid_parts = None

    def _depends(self):
        raise NotImplementedError

    def _make_hashid(self):
        parts = self._depends()
        hashid = ub.hash_data(parts)
        return hashid, parts

    @property
    def hashid(self):
        if getattr(self, '_hashid', None) is None:
            hashid, parts = self._make_hashid()
            self._hashid = hashid
        return self._hashid


def nestshape(data):
    """
    Examine nested shape of the data

    Example:
        >>> data = [np.arange(10), np.arange(13)]
        >>> nestshape(data)
        [(10,), (13,)]

    Ignore:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> from netharn.data.data_containers import *  # NOQA

        >>> from mmdet.core.mask.structures import *  # NOQA
        >>> masks = [
        >>>     [ np.array([0, 0, 10, 0, 10, 10., 0, 10, 0, 0]) ],
        >>>     [ np.array([0, 0, 10, 0, 10, 10., 0, 10, 5., 5., 0, 0]) ]
        >>> ]
        >>> height, width = 16, 16
        >>> polys = PolygonMasks(masks, height, width)
        >>> nestshape(polys)

        >>> dc = BatchContainer([polys], stack=False)
        >>> print('dc = {}'.format(ub.repr2(dc, nl=1)))

        >>> num_masks, H, W = 3, 32, 32
        >>> rng = np.random.RandomState(0)
        >>> masks = (rng.rand(num_masks, H, W) > 0.1).astype(np.int)
        >>> bitmasks = BitmapMasks(masks, height=H, width=W)
        >>> nestshape(bitmasks)

        >>> dc = BatchContainer([bitmasks], stack=False)
        >>> print('dc = {}'.format(ub.repr2(dc, nl=1)))

    """
    import ubelt as ub

    def _recurse(d):
        import torch
        import numpy as np
        if isinstance(d, dict):
            return ub.odict(sorted([(k, _recurse(v)) for k, v in d.items()]))

        clsname = type(d).__name__
        if 'Container' in clsname:
            meta = ub.odict(sorted([
                ('stack', d.stack),
                # ('padding_value', d.padding_value),
                # ('pad_dims', d.pad_dims),
                # ('datatype', d.datatype),
                ('cpu_only', d.cpu_only),
            ]))
            meta = ub.repr2(meta, nl=0)
            return {type(d).__name__ + meta: _recurse(d.data)}
        elif isinstance(d, list):
            return [_recurse(v) for v in d]
        elif isinstance(d, tuple):
            return tuple([_recurse(v) for v in d])
        elif isinstance(d, torch.Tensor):
            return d.shape
        elif isinstance(d, np.ndarray):
            return d.shape
        elif isinstance(d, (str, bytes)):
            return d
        elif isinstance(d, (int, float)):
            return d
        elif isinstance(d, slice):
            return d
        elif 'PolygonMasks' == clsname:
            # hack for mmdet
            return repr(d)
        elif 'BitmapMasks' == clsname:
            # hack for mmdet
            return repr(d)
        elif hasattr(d, 'shape'):
            return d.shape
        elif hasattr(d, 'items'):
            # hack for dict-like objects
            return ub.odict(sorted([(k, _recurse(v)) for k, v in d.items()]))
        elif d is None:
            return None
        else:
            raise TypeError(type(d))

    # globals()['_recurse'] = _recurse
    d = _recurse(data)
    return d
