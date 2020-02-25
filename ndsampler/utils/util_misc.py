# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import numpy as np
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


def stats_dict(list_, axis=None, nan=False, sum=False, extreme=True,
               n_extreme=False, median=False, shape=True, size=False):
    """
    Args:
        list_ (listlike): values to get statistics of
        axis (int): if `list_` is ndarray then this specifies the axis
        nan (bool): report number of nan items
        sum (bool): report sum of values
        extreme (bool): report min and max values
        n_extreme (bool): report extreme value frequencies
        median (bool): report median
        size (bool): report array size
        shape (bool): report array shape

    Returns:
        collections.OrderedDict: stats: dictionary of common numpy statistics
            (min, max, mean, std, nMin, nMax, shape)

    SeeAlso:
        scipy.stats.describe

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> axis = 0
        >>> rng = np.random.RandomState(0)
        >>> list_ = rng.rand(10, 2).astype(np.float32)
        >>> stats = stats_dict(list_, axis=axis, nan=False)
        >>> result = str(ub.repr2(stats, nl=1, precision=4, with_dtype=True))
        >>> print(result)
        {
            'mean': np.array([ 0.5206,  0.6425], dtype=np.float32),
            'std': np.array([ 0.2854,  0.2517], dtype=np.float32),
            'min': np.array([ 0.0202,  0.0871], dtype=np.float32),
            'max': np.array([ 0.9637,  0.9256], dtype=np.float32),
            'shape': (10, 2),
        }

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> axis = 0
        >>> rng = np.random.RandomState(0)
        >>> list_ = rng.randint(0, 42, size=100).astype(np.float32)
        >>> list_[4] = np.nan
        >>> stats = stats_dict(list_, axis=axis, nan=True)
        >>> result = str(ub.repr2(stats, nl=0, precision=1, strkeys=True))
        >>> print(result)
        {mean: 20.0, std: 13.2, min: 0.0, max: 41.0, shape: (100,), num_nan: 1}
    """
    stats = collections.OrderedDict([])

    # Ensure input is in numpy format
    if isinstance(list_, np.ndarray):
        nparr = list_
    elif isinstance(list_, list):
        nparr = np.array(list_)
    else:
        nparr = np.array(list(list_))
    # Check to make sure stats are feasible
    if len(nparr) == 0:
        stats['empty_list'] = True
        if size:
            stats['size'] = 0
    else:
        if nan:
            min_val = np.nanmin(nparr, axis=axis)
            max_val = np.nanmax(nparr, axis=axis)
            mean_ = np.nanmean(nparr, axis=axis)
            std_  = np.nanstd(nparr, axis=axis)
        else:
            min_val = nparr.min(axis=axis)
            max_val = nparr.max(axis=axis)
            mean_ = nparr.mean(axis=axis)
            std_  = nparr.std(axis=axis)
        # number of entries with min/max val
        nMin = np.sum(nparr == min_val, axis=axis)
        nMax = np.sum(nparr == max_val, axis=axis)

        # Notes:
        # the first central moment is 0
        # the first raw moment is the mean
        # the second central moment is the variance
        # the third central moment is the skweness * var ** 3
        # the fourth central moment is the kurtosis * var ** 4
        # the third standardized moment is the skweness
        # the fourth standardized moment is the kurtosis

        if True:
            stats['mean'] = np.float32(mean_)
            stats['std'] = np.float32(std_)
        if extreme:
            stats['min'] = np.float32(min_val)
            stats['max'] = np.float32(max_val)
        if n_extreme:
            stats['nMin'] = np.int32(nMin)
            stats['nMax'] = np.int32(nMax)
        if size:
            stats['size'] = nparr.size
        if shape:
            stats['shape'] = nparr.shape
        if median:
            stats['med'] = np.nanmedian(nparr)
        if nan:
            stats['num_nan'] = np.isnan(nparr).sum()
        if sum:
            sumfunc = np.nansum if nan else np.sum
            stats['sum'] = sumfunc(nparr, axis=axis)
    return stats
