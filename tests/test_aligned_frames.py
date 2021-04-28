import ubelt as ub
import numpy as np


def test_channel_alignment():
    import ndsampler
    sampler = ndsampler.CocoSampler.demo('vidshapes5-multispectral', backend=None)
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

    # Prefused inputs should be in native resolution
    assert subregion1['subregions'][0]['im'].shape[0] == 600
    assert subregion11['subregions'][0]['im'].shape[0] == 200
    # They should be carried along with their transforms
    assert subregion1['subregions'][0]['tf_crop_to_img'][0][0] == 1
    assert subregion11['subregions'][0]['tf_crop_to_img'][0][0] == 3

    img_region = (slice(10, 32), slice(10, 32))
    fused_region = alignable._load_fused_region(img_region, channels=['B1', 'B11'])
    assert fused_region.shape == (22, 22, 2)

    fused_full = alignable._load_fused_region(None, channels=['B1', 'B11'])
    assert fused_full.shape == (600, 600, 2)

    subregion1_crop = alignable._load_prefused_region(img_region, channels=['B1'])
    assert subregion1_crop['subregions'][0]['tf_crop_to_img'][0][2] == 0
    subregion11_crop = alignable._load_prefused_region(img_region, channels=['B11'])
    assert subregion11_crop['subregions'][0]['tf_crop_to_img'][0][2] > 0, (
        'should have a small translation factor because img_region aligns to '
        'subpixels in the auxiliary channel')


class VirtualConcat:
    pass


class VirtualTransform(ub.NiceRepr):
    """
    POC for chainable transforms

    Ignore:
        shape = (12, 12)
        tf1 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        tf2 = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]])
        tf3 = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 1]])
        band1 = VirtualTransform(np.random.rand(6, 6)).update_transform(tf1, shape)
        band2 = VirtualTransform(np.random.rand(4, 4)).update_transform(tf2, shape)
        band3 = VirtualTransform(np.random.rand(3, 3)).update_transform(tf3, shape)

        self = band2

        parent_region_slices = (slice(5, 10), slice(0, 12))
        self.crop(parent_region_slices)


        # Chain one virtual transform inside another
        tf4 = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
        chained = VirtualTransform(self, tf4, (18, 18))

        tf4 = np.array([[.5, 0, 0], [0, .5, 0], [0, 0, 1]])
        chained = VirtualTransform(self, tf4, (6, 6))

        parent_region_slices = (slice(1, 5), slice(2, 4))
        crop = chained.crop(parent_region_slices)

        import cv2
        from kwimage import im_cv2
        crop_data = crop['data']
        dsize = crop['dsize']
        M = crop['transform'][0:2]
        flags = im_cv2._coerce_interpolation('linear')
        aligned_chan = cv2.warpAffine(crop_data, M, dsize=dsize, flags=flags)
    """
    def __init__(self, sub_data, tf_sub_to_parent=None, parent_dsize=None):
        import kwimage
        import numpy as np
        self.sub_data = sub_data
        self.sub_shape = self.sub_data.shape
        sub_h, sub_w = self.sub_shape[0:2]
        self.sub_dsize = (sub_w, sub_h)
        self.sub_corners = kwimage.Coords(
            np.array([[0,     0],
                      [sub_w, 0],
                      [    0, sub_h],
                      [sub_w, sub_h]])
        )
        self.parent_dsize = None
        self.tf_sub_to_parent = None
        self.parent_corners = None
        self.update_transform(tf_sub_to_parent, parent_dsize)

    @property
    def shape(self):
        w, h = self.parent_dsize
        return (h, w)

    def update_transform(self, tf_sub_to_parent, parent_dsize=ub.NoParam):
        self.tf_sub_to_parent = tf_sub_to_parent
        self.parent_corners = self.sub_corners.warp(self.tf_sub_to_parent)
        if parent_dsize is ub.NoParam:
            pass
        elif parent_dsize is None:
            self.parent_dsize = parent_dsize
            max_x = max(self.parent_corners.data[:, 0].max(), 0)
            max_y = max(self.parent_corners.data[:, 1].max(), 0)
            self.parent_dsize = (max_x, max_y)
        else:
            self.parent_dsize = parent_dsize
        return self

    def __nice__(self):
        return '{} -> {}'.format(self.sub_dsize, self.parent_dsize)

    def crop(self, parent_region_slices):
        parent_region_box = _slice_to_box(parent_region_slices, *self.parent_dsize)
        parent_region_corners = parent_region_box.to_polygons()[0]
        parent_region_dsize = tuple(map(int, parent_region_box.to_xywh().data[0, 2:4].tolist()))

        tf_sub_to_parent = self.tf_sub_to_parent
        tf_parent_to_sub = np.linalg.inv(tf_sub_to_parent)

        sub_region_corners = parent_region_corners.warp(tf_parent_to_sub)
        sub_region_box = sub_region_corners.bounding_box().to_ltrb()

        # Quantize to a region that is possible to sample from
        sub_crop_box = sub_region_box.quantize()
        lt_x, lt_y, rb_x, rb_y = sub_crop_box.data[0, 0:4]

        sub_crop_slices = (slice(lt_y, rb_y), slice(lt_x, rb_x))

        # Because we sampled a large quantized region, we need to modify the
        # aux-to-img transform to nudge it a bit to the left, undoing the
        # quantization, which has a bit of extra padding on the left, before
        # applying the final transform.
        crop_offset = sub_crop_box.data[0, 0:2]
        corner_offset = sub_region_box.data[0, 0:2]
        offset_xy =  crop_offset - corner_offset

        subpixel_offset = np.array([
            [1, 0, offset_xy[0]],
            [0, 1, offset_xy[1]],
            [0, 0, 1],
        ], dtype=np.float32)
        # Resample the smaller region to align it with the base region
        # Note: The right most transform is applied first
        tf_crop_to_parent = tf_sub_to_parent @ subpixel_offset

        if hasattr(self.sub_data, 'crop'):
            sub_crop = self.sub_data.crop(sub_crop_slices)
            crop = {
                'data': sub_crop['data'],
                'transform_chain': [tf_crop_to_parent] + sub_crop['transform_chain'],
                'transform': tf_crop_to_parent @ sub_crop['transform'],
                'dsize': parent_region_dsize,
            }
        else:
            crop_data = self.sub_data[sub_crop_slices]
            crop = {
                'data': crop_data,
                'transform_chain': [tf_crop_to_parent],
                'transform': tf_crop_to_parent,
                'dsize': parent_region_dsize,
            }
        return crop


def _slice_to_box(slices, width=None, height=None):
    import kwimage
    y_sl, x_sl = slices[0:2]
    tl_x = x_sl.start
    tl_y = y_sl.start
    rb_x = x_sl.stop
    rb_y = y_sl.stop

    if tl_x is None or tl_x < 0:
        tl_x = 0

    if tl_y is None or tl_y < 0:
        tl_y = 0

    if rb_x is None:
        rb_x = width
        if width is None:
            raise Exception
    elif rb_x < 0:
        if width is None:
            raise Exception
        rb_x = width + rb_x

    if rb_y is None:
        rb_y = height
        if height is None:
            raise Exception
    elif rb_y < 0:
        if height is None:
            raise Exception
        rb_y = height + rb_y

    ltrb = np.array([[tl_x, tl_y, rb_x, rb_y]])
    box = kwimage.Boxes(ltrb, 'ltrb')
    return box
