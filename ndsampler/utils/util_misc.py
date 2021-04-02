# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
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
