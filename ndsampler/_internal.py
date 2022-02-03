import os


def _boolean_environ(key):
    value = os.environ.get(key, '').lower()
    TRUTHY_ENVIRONS = {'true', 'on', 'yes', '1'}
    return value in TRUTHY_ENVIRONS


NDSAMPLER_DISABLE_WARNINGS = _boolean_environ('NDSAMPLER_DISABLE_WARNINGS')
NDSAMPLER_DISABLE_OPTIONAL_WARNINGS = NDSAMPLER_DISABLE_WARNINGS or _boolean_environ('NDSAMPLER_DISABLE_OPTIONAL_WARNINGS')
