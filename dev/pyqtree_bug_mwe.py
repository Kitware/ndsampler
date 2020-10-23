

def pyqtree_bug_mwe():
    import pyqtree
    qtree = pyqtree.Index((0, 0, 600, 480))
    oob_tlbr_box = [939, 169, 2085, 1238]
    for idx in range(1, 11):
        qtree.insert(idx, oob_tlbr_box)
    qtree.insert(11, oob_tlbr_box)

    import pyqtree
    qtree = pyqtree.Index((0, 0, 600, 600))
    oob_tlbr_box = [500, 500, 1000, 1000]
    for idx in range(1, 11):
        print('Insert idx = {!r}'.format(idx))
        qtree.insert(idx, oob_tlbr_box)
    idx = 11
    print('Insert idx = {!r}'.format(idx))
    qtree.insert(idx, oob_tlbr_box)


def pyqtree_bug_test_cases():
    """
    """
    import ubelt as ub
    # Test multiple cases
    def basis_product(basis):
        """
        Args:
            basis (Dict[str, List[T]]): list of values for each axes

        Yields:
            Dict[str, T] - points in the grid
        """
        import itertools as it
        keys = list(basis.keys())
        for vals in it.product(*basis.values()):
            kw = ub.dzip(keys, vals)
            yield kw

    height, width = 600, 600
    # offsets = [-100, -50, 0, 50, 100]
    offsets = [-100, -10, 0, 10, 100]
    # offsets = [-100, 0, 100]
    x_edges = [0, width]
    y_edges = [0, height]
    # x_edges = [width]
    # y_edges = [height]
    basis = {
        'tl_x': [e + p for p in offsets for e in x_edges],
        'tl_y': [e + p for p in offsets for e in y_edges],
        'br_x': [e + p for p in offsets for e in x_edges],
        'br_y': [e + p for p in offsets for e in y_edges],
    }

    # Collect and label valid cases
    # M = in bounds (middle)
    # T = out of bounds on the top
    # L = out of bounds on the left
    # B = out of bounds on the bottom
    # R = out of bounds on the right
    cases = []
    for item in basis_product(basis):
        bbox = (item['tl_x'], item['tl_y'], item['br_x'], item['br_y'])
        x1, y1, x2, y2 = bbox
        if x1 < x2 and y1 < y2:
            parts = []

            if x1 < 0:
                parts.append('x1=L')
            elif x1 < width:
                parts.append('x1=M')
            else:
                parts.append('x1=R')

            if x2 <= 0:
                parts.append('x2=L')
            elif x2 <= width:
                parts.append('x2=M')
            else:
                parts.append('x2=R')

            if y1 < 0:
                parts.append('y1=T')
            elif y1 < width:
                parts.append('y1=M')
            else:
                parts.append('y1=B')

            if y2 <= 0:
                parts.append('y2=T')
            elif y2 <= width:
                parts.append('y2=M')
            else:
                parts.append('y2=B')

            assert len(parts) == 4
            label = ','.join(parts)
            cases.append((label, bbox))

    cases = sorted(cases)
    print('total cases: {}'.format(len(cases)))

    failed_cases = []
    passed_cases = []

    # We will execute the MWE in a separate python process via the "-c"
    # argument so we can programatically kill cases that hang
    test_case_lines = [
        'import pyqtree',
        'bbox, width, height = {!r}, {!r}, {!r}',
        'qtree = pyqtree.Index((0, 0, width, height))',
        '[qtree.insert(idx, bbox) for idx in range(1, 11)]',
        'qtree.insert(11, bbox)',
    ]

    import subprocess
    for label, bbox in ub.ProgIter(cases, desc='checking case', verbose=3):
        pycmd = ';'.join(test_case_lines).format(bbox, width, height)
        command = 'python -c "{}"'.format(pycmd)
        info = ub.cmd(command, detatch=True)
        proc = info['proc']
        try:
            if proc.wait(timeout=0.2) != 0:
                raise AssertionError
        except (subprocess.TimeoutExpired, AssertionError):
            # Kill cases that hang
            proc.terminate()
            text = 'Failed case: {}, bbox = {!r}'.format(label, bbox)
            color = 'red'
            failed_cases.append((label, bbox, text))
        else:
            out, err = proc.communicate()
            text = 'Passed case: {}, bbox = {!r}'.format(label, bbox)
            color = 'green'
            passed_cases.append((label, bbox, text))
        print(ub.color_text(text, color))
    print('len(failed_cases) = {}'.format(len(failed_cases)))
    print('len(passed_cases) = {}'.format(len(passed_cases)))

    passed_labels = set([t[0] for t in passed_cases])
    failed_labels = set([t[0] for t in failed_cases])
    print('passed_labels = {}'.format(ub.repr2(sorted(passed_labels))))
    print('failed_labels = {}'.format(ub.repr2(sorted(failed_labels))))
    print('overlap = {}'.format(set(passed_labels) & set(failed_labels)))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/ndsampler/dev/pyqtree_bug_mwe.py
    """
    pyqtree_bug_test_cases()
