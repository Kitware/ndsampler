import numpy as np


def nestshape(data):
    """
    Examine nested shape of the data

    Example:
        >>> data = [np.arange(10), np.arange(13)]
        >>> nestshape(data)
        [(10,), (13,)]
    """
    import ubelt as ub

    def _recurse(d):
        try:
            import torch
        except ImportError:
            torch = None
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
        elif torch is not None and isinstance(d, torch.Tensor):
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
