def test_video_channel_alignment_parts():
    import ndsampler
    import kwimage
    # import numpy as np
    import ubelt as ub
    sampler = ndsampler.CocoSampler.demo('vidshapes5-multispectral', num_frames=5, backend=None)
    dset = sampler.dset  # NOQA
    frames = sampler.frames
    pathinfo = frames._build_pathinfo(1)
    print('pathinfo = {}'.format(ub.repr2(pathinfo, nl=3)))

    b1_aux = pathinfo['channels']['B1']
    assert kwimage.Affine.coerce(b1_aux['warp_aux_to_img']).matrix[0][0] == 1.0
    b11_aux = pathinfo['channels']['B11']
    assert kwimage.Affine.coerce(b11_aux['warp_aux_to_img']).matrix[0][0] == 3.0

    alignable = frames._load_alignable(1)
    frame_b1 = alignable._load_native_channel('B1')
    frame_b11 = alignable._load_native_channel('B11')

    assert frame_b1.shape[0] == 600
    assert frame_b11.shape[0] == 200

    subregion1 = alignable._load_prefused_region(None, channels=['B1'])
    subregion11 = alignable._load_prefused_region(None, channels=['B11'])
    subregion = alignable._load_prefused_region(None, channels=['B1', 'B11'])  # NOQA

    # Prefused inputs should be in native resolution

    delayed1 = subregion1['subregions'][0]['im']
    delayed11 = subregion11['subregions'][0]['im']
    if hasattr(delayed1, 'sub_shape'):
        assert delayed1.sub_shape[0] == 600
        assert delayed11.sub_shape[0] == 200
        # They should be carried along with their transforms
        assert delayed1.transform[0][0] == 1
        assert delayed11.transform[0][0] == 3
    else:
        # Newer stuff
        assert delayed1.subdata.dsize[0] == 600
        assert delayed11.subdata.subdata.dsize[0] == 200
        # They should be carried along with their transforms
        # assert delayed1.meta['transform'].matrix[0][0] == 1
        assert 'transform' not in delayed1.meta
        assert delayed11.meta['transform'].matrix[0][0] == 3

    img_region = (slice(10, 32), slice(10, 32))
    fused_region = alignable._load_fused_region(img_region, channels=['B1', 'B11'])
    assert fused_region.shape == (22, 22, 2)

    fused_full = alignable._load_fused_region(None, channels=['B1', 'B11'])
    assert fused_full.shape == (600, 600, 2)

    subregion1_crop = alignable._load_prefused_region(img_region, channels=['B1'])
    subregion11_crop = alignable._load_prefused_region(img_region, channels=['B11'])
    if hasattr(delayed1, 'sub_shape'):
        assert subregion1_crop['subregions'][0]['im'].transform[0][2] == 0
        assert subregion11_crop['subregions'][0]['im'].transform[0][2] != 0, (
            'should have a small translation factor because img_region aligns to '
            'subpixels in the auxiliary channel')
    else:
        delayed11 = subregion11_crop['subregions'][0]['im']
        delayed1 = subregion1_crop['subregions'][0]['im']
        # assert delayed1.meta['transform'].matrix[0][2] == 0
        assert delayed11.meta['transform'].matrix[0][2] != 0, (
            'should have a small translation factor because img_region aligns to '
            'subpixels in the auxiliary channel')

    if 0:
        # TEST video component
        vidid = 1
        video = dset.index.videos[vidid]

        vid_dsize = (video['width'], video['height'])

        channels = ['B1', 'B11', 'B8']
        gids = dset.index.vidid_to_gids[vidid]

        from ndsampler.delayed import (
            DelayedWarp, DelayedChannelConcat, DelayedFrameConcat)
        vid_frames_ = []
        for gid in gids:
            img = dset.index.imgs[gid]
            tf_img_to_vid = kwimage.Affine.coerce(img['warp_img_to_vid']).matrix

            img_chans_ = []
            for chan_name in channels:
                img_chan = frames.load_image(gid, channels=chan_name)
                img_chans_.append(img_chan)
            img_frame = DelayedChannelConcat(img_chans_)
            vid_frame = DelayedWarp(img_frame, tf_img_to_vid, dsize=vid_dsize)
            vid_frames_.append(vid_frame)

        # space_region = (slice(10, 40), slice(10, 20))
        # for f in vid_frames_:
        #     # f.virtual_crop(space_region)
        #     for leaf in vid.leafs():
        #         pass
        #     print(list(vid.leafs()))
        #     pass

        vid = DelayedFrameConcat(vid_frames_)
        final = vid.finalize()
        print('final.shape = {!r}'.format(final.shape))
        print(ub.repr2(vid.nesting(), nl=-1))


def test_image_channel_alignment_api():
    import ndsampler
    sampler = ndsampler.CocoSampler.demo('vidshapes5-multispectral', num_frames=5, backend=None)

    sample = sampler.load_sample({'gid': 1, 'channels': '<all>'})
    assert sample['im'].shape == (600, 600, 5)

    sample = sampler.load_sample({'gid': 1, 'slices': (slice(0, 10), slice(0, 10)), 'channels': '<all>'})
    assert sample['im'].shape == (10, 10, 5)

    sample = sampler.load_sample({'gid': 1, 'slices': (slice(-10, 700), slice(0, 10)), 'channels': '<all>'})
    assert sample['im'].shape == (710, 10, 5)


def test_vid_channel_alignment_api():
    import ndsampler
    sampler = ndsampler.CocoSampler.demo('vidshapes5-multispectral', num_frames=5, backend=None)

    sample = sampler.load_sample({'vidid': 1, 'channels': '<all>'})
    assert sample['im'].shape == (5, 600, 600, 5)

    sample = sampler.load_sample({'vidid': 1, 'slices': (slice(None), slice(0, 10), slice(0, 10)), 'channels': '<all>'})
    assert sample['im'].shape == (5, 10, 10, 5)

    sample = sampler.load_sample({'vidid': 1, 'slices': (slice(0, 3), slice(0, 10), slice(0, 10)), 'channels': '<all>'})
    assert sample['im'].shape == (3, 10, 10, 5)

    # Dont try to pad time yet
    # sample = sampler.load_sample({'vidid': 1, 'slices': (slice(0, 8), slice(0, 10), slice(0, 10)), 'channels': '<all>'})
    # assert sample['im'].shape == (8, 10, 10, 5)
