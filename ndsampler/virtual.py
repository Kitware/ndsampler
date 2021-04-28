"""
Proof of concept for virtual chainable transforms in Python.

There are several optimizations that could be applied.

This is similar to GDAL's virtual raster table.


Concepts:

    Each class should be a layer that adds a new transformation on top of
    underlying nested layers. Adding new layers should be quick, and there
    should always be the option to "finalize" a stack of layers, chaining the
    transforms / operations and then applying one final efficient transform at
    the end. Currently it does not work exactly like this, but that is the idea.

"""
import ubelt as ub
import numpy as np


class VirtualChannels(ub.NiceRepr):
    """
    Represents multiple channels in an image that could be concatenated

    Attributes:
        channels (List[ArrayLike]): a list of stackable channels
    """
    def __init__(self, channels, dsize=None):
        self.channels = channels
        if dsize is None:
            candidates = [c.dsize for c in self.channels]
            if not ub.allsame(candidates):
                raise ValueError(
                    'channels must all have the same virtual size')
            dsize = candidates[0]
        self.dsize = dsize
        self.num_bands = sum(c.num_bands for c in self.channels)

    @property
    def shape(self):
        w, h = self.dsize
        return (h, w, self.num_bands)

    def __nice__(self):
        return '{}'.format(self.shape)


class VirtualImage(ub.NiceRepr):
    """
    Reproduces the supported functionality of an array like image.

    Example:
        data = np.random.rand(8,
    """
    def __init__(self, data, dsize=None, num_bands=None):
        # Infer attribute from array-like
        if dsize is None:
            h, w = data.shape[0:2]
            dsize = (w, h)
        if num_bands is None:
            if len(data.shape) == 2:
                num_bands = 1
            elif len(data.shape) == 3:
                num_bands = data.shape[2]
            else:
                raise ValueError('Only 2D images for now')
        self.data = data
        self.num_bands = num_bands
        self.dsize = dsize

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

    def __getitem__(self, index):
        return self.data[index]

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

        sub_data (VirtualImage | ArrayLike):
            array-like image data at a naitive resolution

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
        >>> dsize = (12, 12)
        >>> tf1 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        >>> tf2 = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]])
        >>> tf3 = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 1]])
        >>> band1 = VirtualTransform(np.random.rand(6, 6)).set_transform(tf1, dsize)
        >>> band2 = VirtualTransform(np.random.rand(4, 4)).set_transform(tf2, dsize)
        >>> band3 = VirtualTransform(np.random.rand(3, 3)).set_transform(tf3, dsize)
        >>> #
        >>> # Execute a crop in a one-level transformed space
        >>> region_slices = (slice(5, 10), slice(0, 12))
        >>> virtual_crop = band2.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.finalize()
        >>> #
        >>> # Execute a crop in a nested transformed space
        >>> tf4 = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
        >>> chained = VirtualTransform(band2, tf4, (18, 18))
        >>> virtual_crop = chained.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.finalize()
        >>> #
        >>> tf4 = np.array([[.5, 0, 0], [0, .5, 0], [0, 0, 1]])
        >>> chained = VirtualTransform(band2, tf4, (6, 6))
        >>> virtual_crop = chained.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.finalize()
        >>> #
        >>> region_slices = (slice(1, 5), slice(2, 4))
        >>> virtual_crop = chained.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.finalize()

    Example:
        >>> dsize = (17, 12)
        >>> tf = np.array([[5.2, 0, 1.1], [0, 3.1, 2.2], [0, 0, 1]])
        >>> self = VirtualTransform(np.random.rand(3, 5, 13), tf, dsize=dsize)
        >>> self.finalize().shape
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

        if len(self.sub_data.shape) == 2:
            num_bands = 1
        elif len(self.sub_data.shape) == 3:
            num_bands = self.sub_data.shape[2]
        else:
            raise ValueError(
                'Data may only have 2 space dimensions and 1 channel '
                'dimension')
        self.num_bands = num_bands

    @property
    def shape(self):
        # trailing_shape = self.sub_data.shape[2:]
        # trailing shape should only be allowed to have 0 or 1 dimension
        w, h = self.dsize
        return (h, w, self.num_bands)

    def set_transform(self, transform, dsize=ub.NoParam):
        self.transform = transform
        self.corners = self.sub_corners.warp(self.transform)
        if dsize is ub.NoParam:
            pass
        elif dsize is None:
            self.dsize = dsize
            max_xy = np.ceil(self.corners.data.max(axis=0))
            max_x = int(max_xy[0])
            max_y = int(max_xy[1])
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
        # note, we are modifying the transform, but we could
        # just add a new virtual transform layer on top of this.
        sub_data = self.sub_data
        new_transform = transform @ self.transform
        warped = VirtualTransform(sub_data, new_transform, dsize)
        return warped

    def __nice__(self):
        return '{} -> {}'.format(self.sub_shape, self.shape)

    def finalize(self):
        """
        Execute the final transform
        """
        # todo: needs to be extended for the case where the sub_data is a
        # nested chain of transforms.
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

        Returns:
            VirtualTransform: lazy executed virtual transform
        """
        # todo: this could be done as a virtual layer
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
