if 0:
    # Maybe?
    import smqtk
    import abc
    smqtk.Pluggable = smqtk.utils.plugin.Pluggable
    smqtk.Configurable = smqtk.utils.configuration.Configurable

    """
    {
        "frames": {
            "ndsampler.CogFrames:0": {
                'compression': 'JPEG',
            },
            "ndsampler.NPYFrames": {
            },
            "type": "ndsampler.CogFrames"
        }
    }
    """

    class AbstractFrames(smqtk.Pluggable, smqtk.Configurable):

        def __init__(self, config, foo=1):
            pass

        @abc.abstractmethod
        def load_region(self, spec):
            pass

        @classmethod
        def is_usable(cls):
            return True

    class CogFrames(AbstractFrames):

        # default_config = {
        #     'compression': Value(
        #         default='JPEG', choices=['JPEG', 'DEFLATE']),
        # }
        # def __init__(self, **kwargs):
        #     super().__init__(**kwargs)

        def load_region(self, spec):
            return spec

    class NPYFrames(AbstractFrames):
        def load_region(self, spec):
            return spec


def __notes__():
    """
    Testing alternate implementations
    """

    def _subregion_with_pil(img, region):
        """
        References:
            https://stackoverflow.com/questions/23253205/read-a-subset-of-pixels-with-pil-pillow/50823293#50823293
        """
        # It is much faster to crop out a subregion from a npy memmap file
        # than it is to work with a raw image (apparently). Thus the first
        # time we see an image, we load the entire thing, and dump it to a
        # cache directory in npy format. Then any subsequent time the image
        # is needed, we efficiently load it from numpy.
        import kwimage
        from PIL import Image
        gpath = kwimage.grab_test_image_fpath()
        # Directly load a crop region with PIL (this is slow, why?)
        pil_img = Image.open(gpath)

        width = img['width']
        height = img['height']

        yregion, xregion = region
        left, right, _ = xregion.indices(width)
        upper, lower, _ = yregion.indices(height)
        area = left, upper, right, lower

        # Ok, so loading images with PIL is still slow even with crop.  Not
        # sure why. I though I measured that it was significantly better
        # before. Oh well. Let hack this and the first time we read an image we
        # will dump it to a numpy file and then memmap into it to crop out the
        # appropriate slice.
        sub_img = pil_img.crop(area)
        im = np.asarray(sub_img)
        return im

    def _alternate_memap_creation(mem_gpath, img, region):
        # not sure where the 128 magic number comes from
        # https://stackoverflow.com/questions/51314748/is-the-bytes-offset-in-files-produced-by-np-save-always-128
        width = img['width']
        height = img['height']
        img_shape = (height, width, 3)
        img_dtype = np.dtype('uint8')
        file = np.memmap(mem_gpath, dtype=img_dtype, shape=img_shape,
                         offset=128, mode='r')
        return file


def __devcheck_align_regions():
    import ndsampler
    sampler = ndsampler.CocoSampler.demo('vidshapes1-multispectral')
    frames = sampler.frames

    pathinfo = frames._lookup_pathinfo(1)
    print('frames = {}'.format(ub.repr2(pathinfo, nl=3)))

    self = AlignedImageData(pathinfo, frames)
    base_region = (slice(197, 512), (slice(125, 512)))

    chan1 = self.load_image(1, channels='B1')
    tf1_base_to_chan = np.array(pathinfo['channels']['B1']['base_to_aux']['matrix'])

    channels2 = 'B11'
    channels2 = 'B8'
    chan2 = frames.load_image(1, channels=channels2)
    tf2_base_to_chan = np.array(pathinfo['channels'][channels2]['base_to_aux']['matrix'])


    chan = chan1
    tf_base_to_chan = tf1_base_to_chan

    chan = chan2
    tf_base_to_chan = tf2_base_to_chan

    def getslice_aligned2d(chan, base_region, tf_base_to_chan):
        import cv2
        import kwimage
        tf_chan_to_base = np.linalg.inv(tf_base_to_chan)

        # Find relevant points in the base-coordinate system
        y_sl, x_sl = base_region[0:2]
        base_box = kwimage.Boxes([[
            x_sl.start, y_sl.start, x_sl.stop, y_sl.stop
        ]], 'ltrb')
        base_dsize = tuple(map(int, base_box.to_xywh().data[0, 2:4].tolist()))
        base_corners = base_box.to_polygons()[0]

        # Warp the base coordinates into the target channel space
        chan_corners = base_corners.warp(tf_base_to_chan)
        chan_corners_box = chan_corners.bounding_box().to_ltrb()

        # Quantize to a region that is possible to sample from
        chan_sample_box = chan_corners_box.quantize()
        lt_x, lt_y, rb_x, rb_y = chan_sample_box.data[0, 0:4]

        chan_region = (slice(lt_y, rb_y), slice(lt_x, rb_x))
        chan_crop = chan[chan_region]

        # Because we sampled a larget quantized region, we need to modify the
        # chan-to-base base_to_aux to nudge it a bit to the left, undoing the
        # quantization, which has a bit of extra padding on the left, before
        # applying the final base_to_aux.
        sample_offset = chan_sample_box.data[0, 0:2]
        corner_offset = chan_corners_box.data[0, 0:2]
        offset_xy =  sample_offset - corner_offset
        dequantize_offset = np.array([
            [1, 0, offset_xy[0]],
            [0, 1, offset_xy[1]],
            [0, 0, 1],
        ])

        # Resample the smaller region to align it with the base region
        # Note: The right most base_to_aux is applied first
        tf_crop_to_base =  tf_chan_to_base @ dequantize_offset

        M = tf_crop_to_base[0:2]
        flags = kwimage.im_cv2._coerce_interpolation('linear')
        aligned_data = cv2.warpAffine(chan_crop, M, dsize=base_dsize, flags=flags)
        return aligned_data

    aligned1 = getslice_aligned2d(chan1, base_region, tf1_base_to_chan)
    aligned2 = getslice_aligned2d(chan2, base_region, tf2_base_to_chan)

    import kwimage
    align1_norm = kwimage.normalize(aligned1.astype(np.float32))
    align2_norm = kwimage.normalize(aligned2.astype(np.float32))
    chan1_norm = kwimage.normalize(chan1[:].astype(np.float32))
    chan2_norm = kwimage.normalize(chan2[:].astype(np.float32))

    blend = kwimage.overlay_alpha_layers([
        kwimage.ensure_alpha_channel(align2_norm, 0.5),
        kwimage.ensure_alpha_channel(align1_norm, 0.5),
    ])
    align1_norm = kwimage.Boxes([[47, 242, 1, 1]], 'xywh').draw_on(align1_norm, color='red')
    align2_norm = kwimage.Boxes([[47, 242, 1, 1]], 'xywh').draw_on(align2_norm, color='red')

    import kwplot
    kwplot.autompl()

    kwplot.imshow(align1_norm, pnum=(2, 3, 1), fnum=1)
    kwplot.imshow(align2_norm, pnum=(2, 3, 2), fnum=1)

    kwplot.imshow(blend, pnum=(2, 3, 3), fnum=1)

    kwplot.imshow(align2_norm, pnum=(2, 3, 2), fnum=1)

    kwplot.imshow(chan1_norm, pnum=(2, 2, 3), fnum=1)
    kwplot.imshow(chan2_norm, pnum=(2, 2, 4), fnum=1)

    kwplot.imshow(kwimage.normalize(chan1[base_region].astype(np.float32)), fnum=2)
