"""
Proof of concept for virtual chainable transforms in Python.

This is similar to GDAL's virtual raster table.
"""
import ubelt as ub
import numpy as np


class VirtualConcat:
    pass


class VirtualImage(ub.NiceRepr):
    def __init__(self, data):
        self.data = data

    @classmethod
    def coerce(cls, data):
        self = VirtualImage(data)
        return self

    @property
    def shape(self):
        return self.data.shape

    @property
    def dsize(self):
        h, w = self.data.shape[0:2]
        return (w, h)

    def virtual_crop(self, region_slices):
        # region_box = _slice_to_box(region_slices, *self.dsize)
        # region_dsize = tuple(map(int, region_box.to_xywh().data[0, 2:4].tolist()))
        crop_sub_data = self.data[region_slices]
        new = crop_sub_data
        # cropkw = {
        #     'sub_data': crop_sub_data,
        #     'transform': np.eye(3),
        #     'dsize': region_dsize,
        # }
        # new = VirtualTransform(**cropkw)
        return new


class VirtualTransform(ub.NiceRepr):
    """
    POC for chainable transforms

    Notes:
        "sub" is used to refer to the underlying data in its native coordinates
        and resolution.

        "self" is used to refer to the data in the transformed coordinates that
        are exposed by this class.

    Attributes:

        sub_data (VirtualImage):
            data at a naitive resolution

        sub_corners (Coords):
            coordinates of corners in "sub"-image-space.

        sub_dsize (Tuple):
            width and height of the

        transform (Transform):
            transforms data from native "sub"-image-space to
            "self"-image-space.

        dsize (Tuple[int, int]):
            width and height of the data in "self"-space.

        corners (Coords):
            coordinates of corners in "self"-image-space.

    Example:
        >>> from ndsampler.virtual import *  # NOQA
        >>> shape = (12, 12)
        >>> tf1 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        >>> tf2 = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]])
        >>> tf3 = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 1]])
        >>> band1 = VirtualTransform(np.random.rand(6, 6)).set_transform(tf1, shape)
        >>> band2 = VirtualTransform(np.random.rand(4, 4)).set_transform(tf2, shape)
        >>> band3 = VirtualTransform(np.random.rand(3, 3)).set_transform(tf3, shape)
        >>> #
        >>> # Execute a crop in a one-level transformed space
        >>> region_slices = (slice(5, 10), slice(0, 12))
        >>> virtual_crop = band2.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.execute()
        >>> #
        >>> # Execute a crop in a nested transformed space
        >>> tf4 = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
        >>> chained = VirtualTransform(band2, tf4, (18, 18))
        >>> virtual_crop = chained.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.execute()
        >>> #
        >>> tf4 = np.array([[.5, 0, 0], [0, .5, 0], [0, 0, 1]])
        >>> chained = VirtualTransform(band2, tf4, (6, 6))
        >>> virtual_crop = chained.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.execute()
        >>> #
        >>> region_slices = (slice(1, 5), slice(2, 4))
        >>> virtual_crop = chained.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.execute()
    """
    def __init__(self, sub_data, transform=None, dsize=None):
        import kwimage
        import numpy as np
        self.sub_data = sub_data

        self.sub_shape = self.sub_data.shape
        sub_h, sub_w = self.sub_shape[0:2]
        self.sub_dsize = (sub_w, sub_h)
        self.sub_corners = kwimage.Coords(
            np.array([[0,     0], [sub_w, 0], [    0, sub_h], [sub_w, sub_h]])
        )
        self.dsize = None
        self.transform = None
        self.corners = None

        self.set_transform(transform, dsize)

    @property
    def shape(self):
        trailing_shape = self.sub_data.shape[2:]
        w, h = self.dsize
        return (h, w) + trailing_shape

    def set_transform(self, transform, dsize=ub.NoParam):
        self.transform = transform
        self.corners = self.sub_corners.warp(self.transform)
        if dsize is ub.NoParam:
            pass
        elif dsize is None:
            self.dsize = dsize
            max_x = max(self.corners.data[:, 0].max(), 0)
            max_y = max(self.corners.data[:, 1].max(), 0)
            self.dsize = (max_x, max_y)
        else:
            self.dsize = dsize
        return self

    def virtual_warp(self, transform, dsize):
        """
        Virtually transform the underlying data.

        Note:
            this deviates from kwimage warp functions because instead of
            "output_dims" (specified in c-style shape) we specify dsize (w, h).

        Returns:
            VirtualTransform : new virtual transform a chained transform
        """
        sub_data = self.sub_data
        new_transform = transform @ self.transform
        warped = VirtualTransform(sub_data, new_transform, dsize)
        return warped

    def __nice__(self):
        return '{} -> {}'.format(self.sub_dsize, self.dsize)

    def execute(self):
        """
        Execute the final transform
        """
        import cv2
        from kwimage import im_cv2
        sub_data = self.sub_data
        dsize = self.dsize
        M = self.transform[0:2]
        flags = im_cv2._coerce_interpolation('linear')
        final = cv2.warpAffine(sub_data, M, dsize=dsize, flags=flags)
        return final

    def __getitem__(self, region_slices):
        return self.virtual_crop(region_slices)

    def virtual_crop(self, region_slices):
        """
        Create a new virtual image that performs a crop in the transformed
        "self" space.

        Args:
            region_slices (Tuple[slice, slice]): y-slice and x-slice.
        """
        region_box = _slice_to_box(region_slices, *self.dsize)
        region_dsize = tuple(map(int, region_box.to_xywh().data[0, 2:4].tolist()))
        region_corners = region_box.to_polygons()[0]

        tf_sub_to_self = self.transform
        tf_self_to_sub = np.linalg.inv(tf_sub_to_self)

        sub_region_corners = region_corners.warp(tf_self_to_sub)
        sub_region_box = sub_region_corners.bounding_box().to_ltrb()

        # Quantize to a region that is possible to sample from
        sub_crop_box = sub_region_box.quantize()
        lt_x, lt_y, rb_x, rb_y = sub_crop_box.data[0, 0:4]

        sub_crop_slices = (slice(lt_y, rb_y), slice(lt_x, rb_x))

        # Because we sampled a large quantized region, we need to modify the
        # transform to nudge it a bit to the left, undoing the quantization,
        # which has a bit of extra padding on the left, before applying the
        # final transform.
        crop_offset = sub_crop_box.data[0, 0:2]
        corner_offset = sub_region_box.data[0, 0:2]
        offset_xy = crop_offset - corner_offset
        offset_x, offset_y = offset_xy
        subpixel_offset = np.array([
            [1., 0., offset_x],
            [0., 1., offset_y],
            [0., 0., 1.],
        ], dtype=np.float32)
        # Resample the smaller region to align it with the self region
        # Note: The right most transform is applied first
        tf_crop_to_self = tf_sub_to_self @ subpixel_offset

        if hasattr(self.sub_data, 'virtual_crop'):
            sub_crop = self.sub_data.virtual_crop(sub_crop_slices)
            new = sub_crop.virtual_warp(
                tf_crop_to_self, dsize=region_dsize)
        else:
            crop_sub_data = self.sub_data[sub_crop_slices]
            new = VirtualTransform(
                crop_sub_data, tf_crop_to_self, dsize=region_dsize)
        return new


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
