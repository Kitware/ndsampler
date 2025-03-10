def test_load_without_channels():
    """
    Check that we can load data if channel information doesn't exist
    """
    import ndsampler
    import kwcoco
    dset = kwcoco.CocoDataset.demo('vidshapes8')

    # Remove channel information
    for coco_image in dset.images().coco_images:
        for asset in coco_image.iter_asset_objs():
            asset.pop('channels', None)

    sampler = ndsampler.CocoSampler(dset)
    image_id = list(dset.images())[0]

    # Test 3d Load (really 3d and 2d should be using the same code path
    # when there is only 1 image...)
    for image_id in dset.images():
        target = {
            'image_ids': [image_id],
        }
        sampler.load_sample(target)

    # Test 2d Load
    for image_id in dset.images():
        target = {
            'image_id': image_id
        }
        sampler.load_sample(target)


def check_request_channel_multiple_times():
    import ndsampler
    import kwcoco
    dset = kwcoco.CocoDataset.demo('vidshapes8')
    sampler = ndsampler.CocoSampler(dset)
    image_id = list(dset.images())[0]
    target = {
        'image_ids': [image_id],
        'channels': 'r|r|r'
    }
    sample3d = sampler.load_sample(target)
    assert sample3d['im'].shape == (1, 600, 600, 3)


def test_3d_sample_with_no_video_info():
    """
    Using the 3d loader on something that doesn't have a video
    should default to using image info
    """
    import ndsampler
    import kwcoco
    dset = kwcoco.CocoDataset.demo('vidshapes8')

    for img in dset.dataset['images']:
        img.pop('video_id')

    dset.rebuild_index()

    sampler = ndsampler.CocoSampler(dset)
    image_id = list(dset.images())[0]

    target = {
        'image_ids': [image_id],
    }
    sample3d = sampler.load_sample(target)
    assert sample3d['im'].shape == (1, 600, 600, 3)
