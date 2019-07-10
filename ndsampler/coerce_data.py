import ubelt as ub


def coerce_datasets(config, build_hashid=False, verbose=1):
    """
    Coerce train / val / test datasets from standard netharn config keys

    Example:
        >>> import ndsampler.coerce_data
        >>> config = {'datasets': 'special:shapes'}
        >>> print('config = {!r}'.format(config))
        >>> dsets = ndsampler.coerce_data.coerce_datasets(config)
        >>> print('dsets = {!r}'.format(dsets))

        >>> config = {'datasets': 'special:shapes256'}
        >>> ndsampler.coerce_data.coerce_datasets(config)

        >>> config = {
        >>>     'datasets': ndsampler.CocoDataset.demo('shapes'),
        >>> }
        >>> coerce_datasets(config)
        >>> coerce_datasets({
        >>>     'datasets': ndsampler.CocoDataset.demo('shapes'),
        >>>     'test_dataset': ndsampler.CocoDataset.demo('photos'),
        >>> })
        >>> coerce_datasets({
        >>>     'datasets': ndsampler.CocoDataset.demo('shapes'),
        >>>     'test_dataset': ndsampler.CocoDataset.demo('photos'),
        >>> })
    """
    # Ideally the user specifies a standard train/vali/test split
    def _rectify_fpath(key):
        fpath = key
        fpath = fpath.lstrip('path:').lstrip('PATH:')
        fpath = ub.expandpath(fpath)
        return fpath

    def _ensure_coco(coco):
        print('coco = {!r}'.format(coco))
        # Map a file path or an in-memory dataset to a CocoDataset
        import ndsampler
        import six
        from os.path import exists
        if coco is None:
            return None
        elif isinstance(coco, six.string_types):
            fpath = _rectify_fpath(coco)
            if exists(fpath):
                # print('read dataset: fpath = {!r}'.format(fpath))
                coco = ndsampler.CocoDataset(fpath)
            else:
                if not coco.lower().startswith('special:'):
                    import warnings
                    warnings.warn('warning start dataset codes with special:')
                    code = coco
                else:
                    code = coco.lower()[len('special:'):]
                coco = ndsampler.CocoDataset.demo(code)
        else:
            # print('live dataset')
            assert isinstance(coco, ndsampler.CocoDataset)
        return coco

    config = config.copy()

    subsets = {
        'train': config.get('train_dataset', None),
        'vali': config.get('vali_dataset', None),
        'test': config.get('test_dataset', None),
    }

    # specifying any train / vali / test disables datasets
    if any(d is not None for d in subsets.values()):
        config['datasets'] = None

    if verbose:
        print('Checking for explicit subsets')
    subsets = ub.map_vals(_ensure_coco, subsets)

    # However, sometimes they just specify a single dataset, and we need to
    # make a split for it.
    print('config = {!r}'.format(config))
    base = _ensure_coco(config.get('datasets', None))
    print('base = {!r}'.format(base))
    if base is not None:
        if verbose:
            print('Splitting base into train/vali')
        # TODO: the actual split may need to be cached.
        split_gids = _split_train_vali_test(base)
        if config.get('no_test', False):
            split_gids['train'] += split_gids.pop('test')
        for tag in split_gids.keys():
            gids = split_gids[tag]
            subset = base.subset(sorted(gids), copy=True)
            subset.tag = base.tag + '-' + tag
            subsets[tag] = subset

    subsets = {k: v for k, v in subsets.items() if v is not None}
    if build_hashid:
        print('Building subset hashids')
        for tag, subset in subsets.items():
            print('Build index for {}'.format(subset.tag))
            subset._build_index()
            print('Build hashid for {}'.format(subset.tag))
            subset._build_hashid(hash_pixels=False, verbose=10)

    # if verbose:
    #     print(_catfreq_columns_str(subsets))
    return subsets


def _print_catfreq_columns(subsets):
    print('Category Split Frequency:')
    print(_catfreq_columns_str(subsets))


def _catfreq_columns_str(subsets):
    import pandas as pd
    split_freq = {}
    for tag, subset in subsets.items():
        freq = subset.category_annotation_frequency()
        split_freq[tag] = freq

    df_ = pd.DataFrame.from_dict(split_freq)
    df_['sum'] = df_.sum(axis=1)
    df_ = df_.sort_values('sum')

    with pd.option_context('display.max_rows', 1000):
        text = df_.to_string()
    return text


def _split_train_vali_test(coco_dset, factor=3):
    """

    Args:
        factor (int): number of pieces to divide images into

    Ignore:
        import xdev
        xdev.quantum_random()
    """
    import kwarray
    from sklearn import model_selection
    images = coco_dset.images()

    def _stratified_split(gids, cids, n_splits=2, rng=None):
        """ helper to split while trying to maintain class balance within images """
        rng = kwarray.ensure_rng(rng)
        selector = model_selection.StratifiedKFold(n_splits=n_splits, random_state=rng)
        skf_list = list(selector.split(X=gids, y=cids))
        trainx, testx = skf_list[0]
        return trainx, testx

    # Create flat table of image-ids and category-ids
    gids, cids = [], []
    for gid_, cids_ in zip(images, images.annots.cids):
        cids.extend(cids_)
        gids.extend([gid_] * len(cids_))

    # Split into learn/test then split learn into train/vali
    learnx, testx = _stratified_split(gids, cids, rng=2997217409,
                                      n_splits=factor)
    learn_gids = list(ub.take(gids, learnx))
    learn_cids = list(ub.take(cids, learnx))
    _trainx, _valix = _stratified_split(learn_gids, learn_cids, rng=140860164,
                                        n_splits=factor)
    trainx = learnx[_trainx]
    valix = learnx[_valix]

    split_gids = {
        'train': sorted(ub.unique(ub.take(gids, trainx))),
        'vali': sorted(ub.unique(ub.take(gids, valix))),
        'test': sorted(ub.unique(ub.take(gids, testx))),
    }

    if True:
        # Hack to favor training a good model over testing it properly The only
        # real fix to this is to add more data, otherwise its simply a systemic
        # issue.
        split_gids['vali'] = sorted(set(split_gids['vali']) - set(split_gids['train']))
        split_gids['test'] = sorted(set(split_gids['test']) - set(split_gids['train']))
        split_gids['test'] = sorted(set(split_gids['test']) - set(split_gids['vali']))

    if __debug__:
        import itertools as it
        for a, b in it.combinations(split_gids.values(), 2):
            if (set(a) & set(b)):
                print('split_gids = {!r}'.format(split_gids))
                assert False

    return split_gids
