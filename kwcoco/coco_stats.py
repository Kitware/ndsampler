#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ubelt as ub
import scriptconfig as scfg


class CocoStatsConfig(scfg.Config):
    default = {
        'src': scfg.Value(None, help='path to dataset'),
        'basic': scfg.Value(True),
        'extended': scfg.Value(True),
        'catfreq': scfg.Value(True),
        'boxes': scfg.Value(False),
    }


class CocoStatsCLI:

    def main(cmdline=True, **kw):
        """
        Example:
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> CocoStatsCLI.main()
        """
        import ndsampler
        config = CocoStatsConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: '.format(config['src']))

        dset = ndsampler.CocoDataset.coerce(config['src'])
        print('dset.fpath = {!r}'.format(dset.fpath))

        if config['basic']:
            basic = dset.basic_stats()
            print('basic = {}'.format(ub.repr2(basic, nl=1)))

        if config['extended']:
            extended = dset.extended_stats()
            print('extended = {}'.format(ub.repr2(extended, nl=1, precision=2)))

        if config['catfreq']:
            print('Category frequency')
            freq = dset.category_annotation_frequency()
            import pandas as pd
            df = pd.DataFrame.from_dict({str(dset.tag): freq})
            pd.set_option('max_colwidth', 256)
            print(df.to_string(float_format=lambda x: '%0.3f' % x))

        if config['boxes']:
            print('Box stats')
            print(ub.repr2(dset.boxsize_stats(), nl=-1, precision=2))


def _main(*a, **kw):
    return CocoStatsCLI.main(*a, **kw)

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.coco_stats --src=special:shapes8
    """
    _main()
