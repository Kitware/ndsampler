def memmap_failure_mwe():
    from os.path import join
    import numpy as np
    import tempfile

    print('Generate dummy data')
    N = 4
    D = 3
    x, y = np.meshgrid(np.arange(0, N), np.arange(0, N))
    data = np.ascontiguousarray((np.dstack([x] * D) % 256).astype(np.uint8))

    print('Make temp directory')
    dpath = tempfile.mkdtemp()
    mem_fpath = join(dpath, 'foo.npy')

    print('Dump memmap')
    np.save(mem_fpath, data)

    file1 = np.memmap(mem_fpath, dtype=data.dtype.name, shape=data.shape,
                      mode='r')
    file2 = np.load(mem_fpath)
    print('file1 =\n{!r}'.format(file1[0]))
    print('file2 =\n{!r}'.format(file2[0]))

    for i in range(0, 1000):
        file1 = np.memmap(mem_fpath, dtype=data.dtype.name, shape=data.shape,
                          offset=i, mode='r')
        if np.all(file1 == data):
            print('i = {!r}'.format(i))
            break

    print('file1 =\n{!r}'.format(file1[0]))
