import scriptconfig as scfg


class MainConfig(scfg.Config):
    description = 'The Kitware COCO CLI'
    default = {
        'command': scfg.Value(
            None, help='command to execute',
            position=1, choices=['stats'])
    }


def _main(cmdline=True, **kw):
    """
    kw = dict(command='stats')
    cmdline = False
    """
    config = MainConfig(kw, cmdline=cmdline)
    if config['command'] is None:
        raise Exception('no command')
    if config['command'] == 'stats':
        from kwcoco import coco_stats
        coco_stats._main()
    else:
        raise NotImplementedError(config['command'])


if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco --help
        python -m kwcoco.coco_stats

        python ~/code/ndsampler/coco_cli/__main__.py
    """
    _main()
