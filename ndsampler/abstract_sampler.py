# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


class AbstractSampler(object):
    """
    API for Samplers, not all methods need to be implemented depending on the
    use case (for example, load_sample may not be defined if positive /
    negative cases are generated on the fly).
    """

    # Classification categories

    @property
    def class_ids(self):
        raise NotImplementedError

    def lookup_class_name(self, class_id):
        raise NotImplementedError

    def lookup_class_id(self, class_name):
        raise NotImplementedError

    # Arbitrary subregion sampling

    def load_sample(self, tr, pad=None, window_dims=None, visible_thresh=0.1):
        raise NotImplementedError

    # Binary classification properties

    @property
    def n_positives(self):
        raise NotImplementedError

    def load_item(self, index, pad=None, window_dims=None):
        raise NotImplementedError

    def load_positive(self, index=None, pad=None, window_dims=None, rng=None):
        raise NotImplementedError

    def load_negative(self, index=None, pad=None, window_dims=None, rng=None):
        raise NotImplementedError

    # Full image interface

    def load_image(self, image_id):
        raise NotImplementedError

    def image_ids(self):
        raise NotImplementedError

    # Preselect

    def preselect(self, **kwargs):
        """ Setup a pool of training examples before the epoch begins """
        raise NotImplementedError
