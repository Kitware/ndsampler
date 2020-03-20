# -*- coding: utf-8 -*-
"""
This module exists for backwards compatibility. The code moved into the utils
directory.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.warn('Use ndsampler.utils.util_misc instead of ndsampler.util', DeprecationWarning)
from ndsampler.utils.util_misc import *  # NOQA
