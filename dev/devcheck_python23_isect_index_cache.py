"""
Check if regions created in Python 2 can be loaded via Python 3 and vis-versa
"""
import sys
import ubelt as ub
import ndsampler


def access_cache():
    """
    """
    print('Access regions in {}'.format(sys.executable))
    self = ndsampler.CocoSampler.demo(verbose=0).regions

    # Set workdir to a special location
    self.workdir = ub.ensure_app_cache_dir('ndsampler', 'tests', '23_regions')
    print('self.workdir = {!r}'.format(self.workdir))
    print('self.hashid = {!r}'.format(self.hashid))

    self.verbose = 100
    isect_index = self.isect_index

    for gid, qtree in isect_index.qtrees.items():
        print('----')
        print('gid = {!r}'.format(gid))
        print('qtree = {!r}'.format(qtree))
        for node in qtree.nodes:
            print('node.item, node.rect = {!r}, {!r}'.format(node.item, node.rect))


def main():
    try:
        script = __file__
    except NameError:
        raise
        # for Ipython hacking
        script = ub.expandpath('~/code/ndsampler/dev/devcheck_python23_isect_index_cache.py')

    # py2 = ub.find_exe('python2')
    # py3 = ub.find_exe('python3')
    # ub.cmd([py2, script, 'load_regions'], shell=True)
    # ub.cmd([py3, script, 'save_regions'], shell=True)

    # Register scripts for activating python 2/3 virtual envs that have
    # ndsampler installed

    # Hack for Jon's computer
    activate_cmds = {
        'python2': 'we py2.7',
        'python3': 'we venv3.6',
    }

    def run(py):
        bash_cmd = ' && '.join([
            'source $HOME/.bashrc',
            activate_cmds['python3'],
            'python {} access_cache'.format(script),
        ])
        sh_cmd = 'bash -c "{}"'.format(bash_cmd)
        info = ub.cmd(sh_cmd, shell=True, verbose=3)
        return info

    workdir = ub.ensure_app_cache_dir('ndsampler', 'tests', '23_regions')

    # Save in python3, load in python2
    print('\n\n--- SAVE Python3, LOAD Python2 ---')
    ub.delete(workdir, verbose=1)
    py3_info = run('python3')  # NOQA
    py2_info = run('python2')  # NOQA

    print('\n\n--- SAVE Python2, LOAD Python3 ---')
    ub.delete(workdir, verbose=1)  # Clear the cache
    py2_info = run('python2')  # NOQA
    py3_info = run('python3')  # NOQA


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/ndsampler/dev/devcheck_python23_isect_index_cache.py main
    """
    if sys.argv[1] == 'main':
        main()
    elif sys.argv[1] == 'access_cache':
        access_cache()
    else:
        raise AssertionError
