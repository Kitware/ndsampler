import ubelt as ub


def test_category_rename_merge_policy():
    """
    Make sure the same category rename produces the same new hashid

    CommandLine:
        xdoctest -m ~/code/ndsampler/tests/tests_cocodataset.py test_category_rename_hashid_consistency
    """

    import ndsampler
    orig_dset = ndsampler.CocoDataset()
    orig_dset.add_category('a', a=1)
    orig_dset.add_category('b', b=2)
    orig_dset.add_category('c', c=3)
    print(ub.repr2(orig_dset.cats))

    # Test multiple cats map on to one existing unchanged cat
    mapping = {
        'a': 'b',
        'c': 'b',
    }

    dset = orig_dset.copy()
    dset.rename_categories(mapping, merge_policy='update')
    print(ub.repr2(dset.cats, nl=1))
    assert dset.name_to_cat['b'] == {'id': 2, 'name': 'b', 'b': 2, 'a': 1, 'c': 3}

    dset = orig_dset.copy()
    dset.rename_categories(mapping, merge_policy='ignore')
    print(ub.repr2(dset.cats, nl=1))
    assert dset.name_to_cat['b'] == {'id': 2, 'name': 'b', 'b': 2}

    # Test multiple cats map on to one new cat
    mapping = {
        'a': 'e',
        'c': 'e',
        'b': 'd',
    }
    orig_dset._next_ids
    dset = orig_dset.copy()
    dset.rename_categories(mapping, merge_policy='ignore')
    print(ub.repr2(dset.cats, nl=1))
    assert dset.cats == {
        1: { 'id': 1, 'name': 'd', 'b': 2, },
        2: { 'id': 2, 'name': 'e', }}

    dset = orig_dset.copy()
    dset.rename_categories(mapping, merge_policy='update')
    print(ub.repr2(dset.cats, nl=1))
    assert dset.cats == {
        1: {'id': 1, 'name': 'd', 'b': 2},
        2: {'id': 2, 'name': 'e', 'a': 1, 'c': 3},
    }


def test_copy_nextids():
    import ndsampler
    orig_dset = ndsampler.CocoDataset()
    orig_dset.add_category('a')

    # Make sure we aren't shallow copying the next id object
    dset = orig_dset.copy()
    dset.add_category('b')
    new_cid = dset.name_to_cat['b']['id']
    assert new_cid == 2

    dset = orig_dset.copy()
    dset.add_category('b')
    new_cid = dset.name_to_cat['b']['id']
    assert new_cid == 2
