# -*- coding: utf-8 -*-
import collections
import ubelt as ub


class LRUDict(ub.NiceRepr):
    """
    Pure python implementation for lru cache fallback

    References:
        https://github.com/amitdev/lru-dict
        pip install lru-dict
        http://www.kunxi.org/blog/2014/05/lru-cache-in-python/

    Args:
        max_size (int): (default = 5)

    Returns:
        LRUDict: cache_obj

    CommandLine:
        xdoctest -m /home/joncrall/code/ndsampler/ndsampler/utils/util_lru.py LRUDict

    Example:
        >>> from ndsampler.utils.util_lru import *  # NOQA
        >>> max_size = 5
        >>> self = LRUDict(max_size)
        >>> for count in range(0, 5):
        ...     self[count] = count
        >>> print(self)
        >>> self[0]
        >>> for count in range(5, 8):
        ...     self[count] = count
        >>> print(self)
        >>> del self[5]
        >>> assert 4 in self
        >>> # xdoctest: +IGNORE_WANT
        >>> print('self = {}'.format(ub.repr2(self, nl=1)))
        self = <LRUDict({4: 4, 0: 0, 6: 6, 7: 7}) at 0x7f4c78af95e0>
    """

    def __init__(self, max_size):
        self._max_size = max_size
        self._cache = collections.OrderedDict()

    def __contains__(self, item):
        return item in self._cache

    def __delitem__(self, key):
        del self._cache[key]

    def __nice__(self):
        return ub.repr2(self._cache, nl=0)

    def update(self, other):
        for k, v in other.items():
            self[k] = v

    def __iter__(self):
        return iter(self._cache)

    def items(self):
        return self._cache.items()

    def keys(self):
        return self._cache.keys()

    def values(self):
        return self._cache.values()

    def iteritems(self):
        return self._cache.iteritems()

    def iterkeys(self):
        return self._cache.iterkeys()

    def itervalues(self):
        return self._cache.itervalues()

    def clear(self):
        return self._cache.clear()

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, key):
        try:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        except KeyError:
            raise

    def __setitem__(self, key, value):
        try:
            self._cache.pop(key)
        except KeyError:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value

    # -- api for compatibility with lru-dict

    def has_key(self, item):
        return item in self

    def set_size(self, max_size):
        """
        change the capacity of the LRU cache

        Example:
            >>> self = LRUDict(10)
            >>> self.update(dict(zip(range(10), range(10))))
            >>> assert len(self) == 10
            >>> self.set_size(2)
            >>> assert len(self) == 2
            >>> self.set_size(10)
            >>> assert len(self) == 2
        """
        self._max_size = max_size
        excess = max(0, len(self._cache) - self._max_size)
        for _ in range(excess):
            self._cache.popitem(last=False)

    @classmethod
    def new(cls, max_size, impl='auto'):
        """
        Creates an LRU dictionary instance, but uses the efficient c-backend
        if that is available.

        Args:
            max_size (int):
            impl (str, default='auto'): which implementation to use

        Example:

        """
        try:
            import lru
        except Exception as ex:
            lru = None
            import warnings
            warnings.warn(
                'Optional lru-dict c-implementation is unavailable.'
                ' `pip install lru-dict` to supress this warning.'
                ' Fallback to pure python. ex={!r}'.format(ex))

        if impl == 'auto':
            impl = 'py' if lru is None else 'c'

        if impl == 'py':
            self = cls(max_size)
        elif impl == 'c':
            self = lru.LRU(max_size)
        else:
            raise KeyError(impl)
        return self


def _benchmarks():
    """
    Test the speed of LRU implementations and ensure the API is the same.

    CommandLine:
        xdoctest -m ndsampler.utils.util_lru _benchmarks

    Results:
        --- impl=c ---
        Timed c-integer-test for: 10 loops, best of 3
            time per loop: best=198.068 ms, mean=205.546 ± 5.1 ms
        Timed c-miss-case for: 10 loops, best of 3
            time per loop: best=405.937 µs, mean=410.180 ± 4.9 µs
        Timed c-hit-case for: 10 loops, best of 3
            time per loop: best=257.571 µs, mean=266.696 ± 8.3 µs
        Timed c-clear-test for: 10 loops, best of 3
            time per loop: best=341.119 µs, mean=350.805 ± 7.2 µs

        --- impl=py ---
        Timed py-integer-test for: 10 loops, best of 3
            time per loop: best=896.410 ms, mean=898.566 ± 1.8 ms
        Timed py-miss-case for: 10 loops, best of 3
            time per loop: best=2.041 ms, mean=2.116 ± 0.1 ms
        Timed py-hit-case for: 10 loops, best of 3
            time per loop: best=1.095 ms, mean=1.119 ± 0.0 ms
        Timed py-clear-test for: 10 loops, best of 3
            time per loop: best=1.402 ms, mean=1.451 ± 0.0 ms

    Conclusion:
         As expected, the c backend is more performant.
    """
    import timerit
    import numpy as np
    ti = timerit.Timerit(10, bestof=3, verbose=2)

    images = [np.random.rand(512, 512) for _ in range(100)]
    idx_miss_sequence = np.random.randint(0, len(images), size=1000)
    idx_hit_sequence = idx_miss_sequence.copy()
    idx_hit_sequence[100:900] = 3

    impls = ['c', 'py']

    for impl in impls:
        print('\n--- impl={} ---'.format(impl))

        # Integer test
        for timer in ti.reset(impl + '-integer-test'):
            with timer:
                self = LRUDict.new(1000000, impl=impl)
                for idx in range(int(1100000)):
                    self[idx] = idx

        # test with large numbers of cache misses
        for timer in ti.reset(impl + '-miss-case'):
            with timer:
                self = LRUDict.new(2, impl=impl)
                for idx in idx_miss_sequence:
                    if idx not in self:
                        self[idx] = images[idx]
                    self[idx]

        # test with large numbers of cache hits
        for timer in ti.reset(impl + '-hit-case'):
            with timer:
                self = LRUDict.new(2, impl=impl)
                for idx in idx_hit_sequence:
                    if idx not in self:
                        self[idx] = images[idx]
                    self[idx]

        # Clear test
        for timer in ti.reset(impl + '-clear-test'):
            with timer:
                self = LRUDict.new(2, impl=impl)
                for i, idx in enumerate(idx_hit_sequence):
                    if i % 5 == 0:
                        self.clear()
                    if idx not in self:
                        self[idx] = images[idx]
                    self[idx]

    print('\n===')
    print('ti.rankings = {}'.format(ub.repr2(ti.rankings, nl=2)))
