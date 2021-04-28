def test_channel_alignment():
    import ndsampler
    import ubelt as ub
    sampler = ndsampler.CocoSampler.demo('vidshapes5-multispectral', num_frames=5, backend=None)
    dset = sampler.dset  # NOQA
    frames = sampler.frames
    pathinfo = frames._build_pathinfo(1)
    print('pathinfo = {}'.format(ub.repr2(pathinfo, nl=3)))

    b1_aux = pathinfo['channels']['B1']
    assert b1_aux['warp_aux_to_img']['matrix'][0][0] == 1.0
    b11_aux = pathinfo['channels']['B11']
    assert b11_aux['warp_aux_to_img']['matrix'][0][0] == 3.0

    alignable = frames._load_alignable(1)
    frame_b1 = alignable._load_native_channel('B1')
    frame_b11 = alignable._load_native_channel('B11')

    assert frame_b1.shape[0] == 600
    assert frame_b11.shape[0] == 200

    subregion1 = alignable._load_prefused_region(None, channels=['B1'])
    subregion11 = alignable._load_prefused_region(None, channels=['B11'])
    subregion = alignable._load_prefused_region(None, channels=['B1', 'B11'])

    # Prefused inputs should be in native resolution
    assert subregion1['subregions'][0]['im'].sub_shape[0] == 600
    assert subregion11['subregions'][0]['im'].sub_shape[0] == 200
    # They should be carried along with their transforms
    assert subregion1['subregions'][0]['im'].transform[0][0] == 1
    assert subregion11['subregions'][0]['im'].transform[0][0] == 3

    img_region = (slice(10, 32), slice(10, 32))
    fused_region = alignable._load_fused_region(img_region, channels=['B1', 'B11'])
    assert fused_region.shape == (22, 22, 2)

    fused_full = alignable._load_fused_region(None, channels=['B1', 'B11'])
    assert fused_full.shape == (600, 600, 2)

    subregion1_crop = alignable._load_prefused_region(img_region, channels=['B1'])
    assert subregion1_crop['subregions'][0]['im'].transform[0][2] == 0
    subregion11_crop = alignable._load_prefused_region(img_region, channels=['B11'])
    assert subregion11_crop['subregions'][0]['im'].transform[0][2] != 0, (
        'should have a small translation factor because img_region aligns to '
        'subpixels in the auxiliary channel')


    if 0:
        # TEST video component

        vidid = 1
        video = dset.index.videos[vidid]

        vid_dsize = (video['width'], video['height'])

        channels = ['B1', 'B11', 'B8']
        gids = dset.index.vidid_to_gids[vidid]

        from ndsampler.virtual import VirtualTransform, VirtualChannels
        vid_frames = []
        for gid in gids:
            import numpy as np
            img = dset.index.imgs[gid]
            tf_img_to_vid = np.array(img['warp_img_to_vid']['matrix'])

            chans = []
            for chan_name in channels:
                img_chan = frames.load_image(gid, channels=chan_name)
                # vid_chan = VirtualTransform(img_chan, tf_img_to_vid, vid_dsize)
                vid_chan = img_chan.virtual_warp(tf_img_to_vid, vid_dsize)
                chans.append(vid_chan)
            frame = VirtualChannels(chans)

            vid_frames.append(frame)
