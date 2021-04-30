"""
Proof of concept for virtual chainable transforms in Python.

There are several optimizations that could be applied.

This is similar to GDAL's virtual raster table.


Concepts:

    Each class should be a layer that adds a new transformation on top of
    underlying nested layers. Adding new layers should be quick, and there
    should always be the option to "finalize" a stack of layers, chaining the
    transforms / operations and then applying one final efficient transform at
    the end.


Conventions:

    * dsize = (always in width / height), no channels are present

    * shape for images is always (height, width, channels)

    * channels are always the last dimension of each image, if no channel
      dim is specified, finalize will add it.

    * Videos must be the last process in the stack, and add a leading
        time dimension to the shape. dsize is still width, height, but shape
        is now: (time, height, width, chan)

"""
import ubelt as ub
import numpy as np


class VirtualVideo(ub.NiceRepr):
    """
    Represents multiple frames in a video
    """
    def __init__(self, frames, dsize=None):
        self.frames = frames
        if dsize is None:
            dsize_cands = [frame.dsize for frame in self.frames]
            if not ub.allsame(dsize_cands):
                raise ValueError(
                    'components must all have the same virtual size')
            dsize = dsize_cands[0]
        self.dsize = dsize
        nband_cands = [frame.num_bands for frame in self.frames]
        if not ub.allsame(nband_cands):
            raise ValueError(
                'components must all have the same virtual size')
        self.num_bands = nband_cands[0]
        self.num_frames = len(self.frames)

    @property
    def shape(self):
        w, h = self.dsize
        return (self.num_frames, h, w, self.num_bands)

    def __nice__(self):
        return '{}'.format(self.shape)

    def finalize(self, transform=None, dsize=None):
        """
        Execute the final transform
        """
        # Add in the video axis
        stack = [frame.finalize(transform=transform, dsize=dsize)[None, :]
                 for frame in self.frames]
        final = np.concatenate(stack, axis=0)
        return final

    def nesting(self):
        return {
            'shape': self.shape,
            'frames': [frame.nesting() for frame in self.frames],
        }


class VirtualChannels(ub.NiceRepr):
    """
    Represents multiple channels in an image that could be concatenated

    Attributes:
        components (List[VirtualImage]): a list of stackable channels. Each
            component may be comprised of multiple channels.

    Example:
        >>> comp1 = VirtualImage(np.random.rand(11, 7))
        >>> comp2 = VirtualImage(np.random.rand(11, 7, 3))
        >>> comp3 = VirtualImage(
        >>>     np.random.rand(3, 5, 2),
        >>>     transform=Affine.affine(scale=(7/5, 11/3)).matrix,
        >>>     dsize=(7, 11)
        >>> )
        >>> components = [comp1, comp2, comp3]
        >>> chans = VirtualChannels(components)
        >>> final = chans.finalize()
        >>> assert final.shape == chans.shape
        >>> assert final.shape == (11, 7, 6)

        >>> # We should be able to nest VirtualChannels inside virutal images
        >>> frame1 = VirtualImage(
        >>>     chans, transform=Affine.affine(scale=2.2).matrix,
        >>>     dsize=(20, 26))
        >>> frame2 = VirtualImage(
        >>>     np.random.rand(3, 3, 6), dsize=(20, 26))
        >>> frame3 = VirtualImage(
        >>>     np.random.rand(3, 3, 6), dsize=(20, 26))

        >>> print(ub.repr2(frame1.nesting(), nl=-1, sort=False))
        >>> frame1.finalize()
        >>> vid = VirtualVideo([frame1, frame2, frame3])
        >>> print(ub.repr2(vid.nesting(), nl=-1, sort=False))
    """
    def __init__(self, components, dsize=None):
        self.components = components
        if dsize is None:
            dsize_cands = [comp.dsize for comp in self.components]
            if not ub.allsame(dsize_cands):
                raise ValueError(
                    'components must all have the same virtual size')
            dsize = dsize_cands[0]
        self.dsize = dsize
        self.num_bands = sum(comp.num_bands for comp in self.components)

    @property
    def shape(self):
        w, h = self.dsize
        return (h, w, self.num_bands)

    def __nice__(self):
        return '{}'.format(self.shape)

    def finalize(self, transform=None, dsize=None):
        """
        Execute the final transform
        """
        stack = [comp.finalize(transform=transform, dsize=dsize)
                 for comp in self.components]
        final = np.concatenate(stack, axis=2)
        return final

    def nesting(self):
        return {
            'shape': self.shape,
            'components': [comp.nesting() for comp in self.components],
        }


class VirtualImage(ub.NiceRepr):
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
            width and height of the subdata canvas

        transform (Transform):
            transforms data from native "sub"-image-space to
            "self"-image-space.

        dsize (Tuple[int, int]):
            width and height of the canvas in "self"-space.

        corners (Coords):
            coordinates of corners of the sub-data in "self"-image-space.

    Example:
        >>> from ndsampler.virtual import *  # NOQA
        >>> dsize = (12, 12)
        >>> tf1 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        >>> tf2 = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]])
        >>> tf3 = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 1]])
        >>> band1 = VirtualImage(np.random.rand(6, 6)).set_transform(tf1, dsize)
        >>> band2 = VirtualImage(np.random.rand(4, 4)).set_transform(tf2, dsize)
        >>> band3 = VirtualImage(np.random.rand(3, 3)).set_transform(tf3, dsize)
        >>> #
        >>> # Execute a crop in a one-level transformed space
        >>> region_slices = (slice(5, 10), slice(0, 12))
        >>> virtual_crop = band2.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.finalize()
        >>> #
        >>> # Execute a crop in a nested transformed space
        >>> tf4 = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
        >>> chained = VirtualImage(band2, tf4, (18, 18))
        >>> virtual_crop = chained.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.finalize()
        >>> #
        >>> tf4 = np.array([[.5, 0, 0], [0, .5, 0], [0, 0, 1]])
        >>> chained = VirtualImage(band2, tf4, (6, 6))
        >>> virtual_crop = chained.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.finalize()
        >>> #
        >>> region_slices = (slice(1, 5), slice(2, 4))
        >>> virtual_crop = chained.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.finalize()

    Example:
        >>> dsize = (17, 12)
        >>> tf = np.array([[5.2, 0, 1.1], [0, 3.1, 2.2], [0, 0, 1]])
        >>> self = VirtualImage(np.random.rand(3, 5, 13), tf, dsize=dsize)
        >>> self.finalize().shape
    """
    def __init__(self, sub_data, transform=None, dsize=None):
        import kwimage
        self.sub_data = sub_data

        if hasattr(self.sub_data, 'corners'):
            self.sub_shape = self.sub_data.shape
            self.sub_dsize = self.sub_data.sub_dsize
            self.sub_corners = self.sub_data.corners
        else:
            self.sub_shape = self.sub_data.shape
            sub_h, sub_w = self.sub_shape[0:2]
            self.sub_corners = kwimage.Coords(
                np.array([[0,     0], [sub_w, 0], [    0, sub_h], [sub_w, sub_h]])
            )
            self.sub_dsize = (sub_w, sub_h)
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
        if transform is None:
            transform = np.eye(3)
        self.transform = transform
        self.corners = self.sub_corners.warp(self.transform)
        if dsize is ub.NoParam:
            pass
        elif dsize == 'auto':
            self.dsize = dsize
            max_xy = np.ceil(self.corners.data.max(axis=0))
            max_x = int(max_xy[0])
            max_y = int(max_xy[1])
            self.dsize = (max_x, max_y)
        elif dsize is None:
            (h, w) = self.sub_shape[0:2]
            self.dsize = (w, h)
        else:
            self.dsize = dsize
        return self

    def nesting(self):
        if isinstance(self.sub_data, np.ndarray):
            return {
                'shape': self.shape,
                'sub_data': {'shape': self.sub_data.shape},
            }
        else:
            return {
                'shape': self.shape,
                'sub_data': self.sub_data.nesting(),
            }

    def virtual_warp(self, transform, dsize=None):
        """
        Virtually transform the underlying data.

        Note:
            this deviates from kwimage warp functions because instead of
            "output_dims" (specified in c-style shape) we specify dsize (w, h).

        Returns:
            VirtualImage : new virtual transform a chained transform
        """
        # note, we are modifying the transform, but we could
        # just add a new virtual transform layer on top of this.
        sub_data = self.sub_data
        new_transform = transform @ self.transform
        warped = VirtualImage(sub_data, new_transform, dsize)
        return warped

    def __nice__(self):
        return '{} -> {}'.format(self.sub_shape, self.shape)

    def finalize(self, transform=None, dsize=None):
        """
        Execute the final transform

        Can pass a parent transform to augment this underlying transform.

        Args:
            transform (Transform):
            dsize (Tuple[int, int]):

        Example:
            >>> import kwimage
            >>> tf = np.array([[0.99, 0, 3.9], [0, 1.01, -.5], [0, 0, 1]])
            >>> raw = kwimage.grab_test_image(dsize=(54, 65))
            >>> raw = kwimage.ensure_float01(raw)
            >>> # Test nested finalize
            >>> layer1 = raw
            >>> num = 10
            >>> for _ in range(num):
            ...     layer1  = VirtualImage(layer1, tf)
            >>> final1 = layer1.finalize()
            >>> # Test non-nested finalize
            >>> layer2 = VirtualImage(raw, np.eye(3))
            >>> for _ in range(num):
            ...     layer2 = layer2.virtual_warp(tf)
            >>> final2 = layer2.finalize()
            >>> #
            >>> print('final1 = {!r}'.format(final1))
            >>> print('final2 = {!r}'.format(final2))
            >>> print('final1.shape = {!r}'.format(final1.shape))
            >>> print('final2.shape = {!r}'.format(final2.shape))
            >>> assert np.allclose(final1, final2)
            >>> #
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(raw, pnum=(1, 3, 1), fnum=1)
            >>> kwplot.imshow(final1, pnum=(1, 3, 2), fnum=1)
            >>> kwplot.imshow(final2, pnum=(1, 3, 3), fnum=1)
            >>> kwplot.show_if_requested()
        """
        # todo: needs to be extended for the case where the sub_data is a
        # nested chain of transforms.
        import cv2
        import kwarray
        from kwimage import im_cv2
        if dsize is None:
            dsize = self.dsize
        if transform is None:
            transform = self.transform
        else:
            transform = transform @ self.transform
        sub_data = self.sub_data
        if hasattr(sub_data, 'finalize'):
            # Branch finalize
            final = sub_data.finalize(transform=transform, dsize=dsize)
        elif transform is None:
            final = sub_data
        else:
            # Leaf finalize
            flags = im_cv2._coerce_interpolation('linear')
            M = transform[0:2]  # todo allow projective
            final = cv2.warpAffine(sub_data, M, dsize=dsize, flags=flags)

        # Ensure that the last dimension is channels
        final = kwarray.atleast_nd(final, 3, front=False)
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
            VirtualImage: lazy executed virtual transform
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
            new = VirtualImage(
                crop_sub_data, tf_crop_to_self, dsize=region_dsize)
        return new


class Affine(ub.NiceRepr):
    """
    Helper for making affine transform matrices
    """
    def __init__(self, matrix):
        self.matrix = matrix

    def __nice__(self):
        return repr(self.matrix)

    @classmethod
    def coerce(cls, data):
        if isinstance(data, np.ndarray):
            self = cls(matrix=data)
        else:
            raise NotImplementedError
        return self

    def decompose(self):
        raise NotImplementedError

    def __matmul__(self, other):
        return Affine(self.matrix @ other.matrix)

    def inv(self):
        return Affine(np.linalg.inv(self.matrix))

    @classmethod
    def affine(cls, scale=None, offset=None, theta=None, shear=None,
               about=None):
        """
        Create an affine matrix from high-level parameters

        Args:
            scale (float | Tuple[float, float]): x, y scale factor
            offset (float | Tuple[float, float]): x, y translation factor
            theta (float): counter-clockwise rotation angle in radians
            shear (float): counter-clockwise shear angle in radians
            about (float | Tuple[float, float]): x, y location of the origin

        Example:
            >>> from ndsampler.virtual import *  # NOQA
            >>> import kwarray
            >>> rng = kwarray.ensure_rng(None)
            >>> scale = rng.randn(2) * 10
            >>> offset = rng.randn(2) * 10
            >>> about = rng.randn(2) * 10
            >>> theta = rng.randn() * 10
            >>> shear = rng.randn() * 10
            >>> # Create combined matrix from all params
            >>> F = Affine.affine(
            >>>     scale=scale, offset=offset, theta=theta, shear=shear,
            >>>     about=about)
            >>> # Test that combining components matches
            >>> S = Affine.affine(scale=scale)
            >>> T = Affine.affine(offset=offset)
            >>> R = Affine.affine(theta=theta)
            >>> H = Affine.affine(shear=shear)
            >>> O = Affine.affine(offset=about)
            >>> # combine (note shear must be on the RHS of rotation)
            >>> alt  = O @ T @ R @ H @ S @ O.inv()
            >>> print('F    = {}'.format(ub.repr2(F.matrix.tolist(), nl=1)))
            >>> print('alt  = {}'.format(ub.repr2(alt.matrix.tolist(), nl=1)))
            >>> assert np.all(np.isclose(alt.matrix, F.matrix))
            >>> pt = np.vstack([np.random.rand(2, 1), [[1]]])
            >>> print(F.matrix @ pt)
            >>> print(alt.matrix @ pt)

        Sympy:
            >>> # xdoctest: +SKIP
            >>> import sympy
            >>> # Shows the symbolic construction of the code
            >>> # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
            >>> from sympy.abc import theta
            >>> x0, y0, sx, sy, theta, shear, tx, ty = sympy.symbols(
            >>>     'x0, y0, sx, sy, theta, shear, tx, ty')
            >>> # move the center to 0, 0
            >>> tr1_ = np.array([[1, 0,  -x0],
            >>>                  [0, 1,  -y0],
            >>>                  [0, 0,    1]])
            >>> # Affine 3x3 about the origin
            >>> sin1_ = sympy.sin(theta)
            >>> cos1_ = sympy.cos(theta)
            >>> sin2_ = sympy.sin(theta + shear)
            >>> cos2_ = sympy.cos(theta + shear)
            >>> aff0 = np.array([
            >>>     [sx * cos1_, -sy * sin2_, tx],
            >>>     [sx * sin1_,  sy * cos2_, ty],
            >>>     [        0,            0,  1]
            >>> ])
            >>> # move 0, 0 back to the specified origin
            >>> tr2_ = np.array([[1, 0,  x0],
            >>>                  [0, 1,  y0],
            >>>                  [0, 0,   1]])
            >>> # combine transformations
            >>> aff = tr2_ @ aff0 @ tr1_
            >>> print('aff = {}'.format(ub.repr2(aff.tolist(), nl=1)))
        """
        scale = 1 if scale is None else scale
        offset = 0 if offset is None else offset
        shear = 0 if shear is None else shear
        theta = 0 if theta is None else theta
        about = 0 if about is None else about
        sx, sy = _ensure_iterablen(scale, 2)
        tx, ty = _ensure_iterablen(offset, 2)
        x0, y0 = _ensure_iterablen(about, 2)

        # Make auxially varables to reduce the number of sin/cos calls
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_shear_p_theta = np.cos(shear + theta)
        sin_shear_p_theta = np.sin(shear + theta)
        tx_ = -sx * x0 * cos_theta + sy * y0 * sin_shear_p_theta + tx + x0
        ty_ = -sx * x0 * sin_theta - sy * y0 * cos_shear_p_theta + ty + y0
        # Sympy compiled expression
        mat = np.array([[sx * cos_theta, -sy * sin_shear_p_theta, tx_],
                        [sx * sin_theta,  sy * cos_shear_p_theta, ty_],
                        [             0,                       0,  1]])
        self = cls(mat)
        return self


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


def _ensure_iterablen(scalar, n):
    try:
        iter(scalar)
    except TypeError:
        return [scalar] * n
    return scalar
