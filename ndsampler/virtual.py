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

    Notes:


        Video[0]:
            Frame[0]:
                Chan[0]: (32) +--------------------------------+
                Chan[1]: (16) +----------------+
                Chan[2]: ( 8) +--------+
            Frame[1]:
                Chan[0]: (30) +------------------------------+
                Chan[1]: (14) +--------------+
                Chan[2]: ( 6) +------+


        Crop:
                         +              +

            (32) +--------------------------------+


    TODO:
        - [ ] Support computing the transforms when none of the data is loaded

    Example:
        >>> # Example demonstrating the modivating use case
        >>> # We have multiple aligned frames for a video, but each of
        >>> # those frames is in a different resolution. Furthermore,
        >>> # each of the frames consists of channels in different resolutions.
        >>> from ndsampler.virtual import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(None)
        >>> def demo_chan(key, dsize, chan):
        ...     return kwimage.grab_test_image(key, dsize=dsize)[..., chan]
        >>> # Create raw channels in some "native" resolution for frame 1
        >>> f1_chan1 = VirtualImage(demo_chan('astro', (300, 300), 0))
        >>> f1_chan2 = VirtualImage(demo_chan('astro', (200, 200), 1))
        >>> f1_chan3 = VirtualImage(demo_chan('astro', (10, 10), 2))
        >>> # Create raw channels in some "native" resolution for frame 2
        >>> f2_chan1 = VirtualImage(demo_chan('carl', (64, 64), 0))
        >>> f2_chan2 = VirtualImage(demo_chan('carl', (260, 260), 1))
        >>> f2_chan3 = VirtualImage(demo_chan('carl', (10, 10), 2))
        >>> #
        >>> # Virtual warp each channel into its "image" space
        >>> # Note: the images never actually enter this space we transform through it
        >>> f1_dsize = np.array((3, 3))
        >>> f2_dsize = np.array((2, 2))
        >>> f1_img = VirtualChannels([
        >>>     f1_chan1.virtual_warp(Affine.scale(f1_dsize / f1_chan1.dsize), dsize=f1_dsize),
        >>>     f1_chan2.virtual_warp(Affine.scale(f1_dsize / f1_chan2.dsize), dsize=f1_dsize),
        >>>     f1_chan3.virtual_warp(Affine.scale(f1_dsize / f1_chan3.dsize), dsize=f1_dsize),
        >>> ])
        >>> f2_img = VirtualChannels([
        >>>     f2_chan1.virtual_warp(Affine.scale(f2_dsize / f2_chan1.dsize), dsize=f2_dsize),
        >>>     f2_chan2.virtual_warp(Affine.scale(f2_dsize / f2_chan2.dsize), dsize=f2_dsize),
        >>>     f2_chan3.virtual_warp(Affine.scale(f2_dsize / f2_chan3.dsize), dsize=f2_dsize),
        >>> ])
        >>> # Combine frames into a video
        >>> vid_dsize = np.array((280, 280))
        >>> vid = VirtualVideo([
        >>>     f1_img.virtual_warp(Affine.scale(vid_dsize / f1_img.dsize), dsize=vid_dsize),
        >>>     f2_img.virtual_warp(Affine.scale(vid_dsize / f2_img.dsize), dsize=vid_dsize),
        >>> ])
        >>> final = vid.finalize(interpolation='nearest')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(final[0], pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(final[1], pnum=(1, 2, 2), fnum=1)

    Example:
        >>> # Simpler case with fewer nesting levels
        >>> from ndsampler.virtual import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(None)
        >>> def demo_img(key, dsize):
        ...     return kwimage.grab_test_image(key, dsize=dsize)
        >>> # Virtual warp each channel into its "image" space
        >>> # Note: the images never actually enter this space we transform through it
        >>> f1_img = VirtualImage(demo_img('astro', (300, 300)))
        >>> f2_img = VirtualImage(demo_img('carl', (256, 256)))
        >>> # Combine frames into a video
        >>> vid_dsize = np.array((100, 100))
        >>> self = vid = VirtualVideo([
        >>>     f1_img.virtual_warp(Affine.scale(vid_dsize / f1_img.dsize), dsize=vid_dsize),
        >>>     f2_img.virtual_warp(Affine.scale(vid_dsize / f2_img.dsize), dsize=vid_dsize),
        >>> ])
        >>> final = vid.finalize(interpolation='nearest')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(final[0], pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(final[1], pnum=(1, 2, 2), fnum=1)

        >>> region_slices = (slice(0, 90), slice(30, 60))
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

    def finalize(self, transform=None, dsize=None, interpolation='linear'):
        """
        Execute the final transform
        """
        # Add in the video axis
        stack = [frame.finalize(transform=transform, dsize=dsize,
                                interpolation=interpolation)[None, :]
                 for frame in self.frames]
        final = np.concatenate(stack, axis=0)
        return final

    def leafs(self, transform=None, dsize=None, interpolation='linear'):
        """
        Iterate through the leaf nodes
        """
        for frame_x, frame in enumerate(self.frames):
            yield from frame.leafs(transform=transform, dsize=dsize,
                                   interpolation=interpolation)

    def nesting(self):
        return {
            'shape': self.shape,
            'frames': [frame.nesting() for frame in self.frames],
        }

    def virtual_crop(self, region_slices):
        """
        Create a new virtual image that performs a crop in the transformed
        "self" space.

        Args:
            region_slices (Tuple[slice, slice]): y-slice and x-slice.

        Returns:
            VirtualImage: lazy executed virtual transform

               The slice is currently not lazy
        """
        # todo: this could be done as a virtual layer
        region_box = _slice_to_box(region_slices, *self.dsize)
        raise NotImplementedError


class VirtualChannels(ub.NiceRepr):
    """
    Represents multiple channels in an image that could be concatenated

    Attributes:
        components (List[VirtualImage]): a list of stackable channels. Each
            component may be comprised of multiple channels.

    TODO:
        - [ ] can this be generalized into a virtual concat?
        - [ ] can all concats be delayed until the very end?

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

    @classmethod
    def random(cls, num_parts=3, rng=None):
        """
        CommandLine:
            xdoctest -m ndsampler.virtual VirtualImage.random

        Example:
            >>> self = VirtualChannels.random()
            >>> print('self = {!r}'.format(self))
            >>> print(self.nesting())
        """
        import kwarray
        rng = kwarray.ensure_rng(rng)
        self_w = rng.randint(8, 64)
        self_h = rng.randint(8, 64)
        components = []
        for _ in range(num_parts):
            subcomp = VirtualImage.random(rng=rng)
            tf = Affine.random(rng=rng).matrix
            comp = subcomp.virtual_warp(tf, dsize=(self_w, self_h))
            components.append(comp)
        self = VirtualChannels(components)
        return self

    @property
    def shape(self):
        w, h = self.dsize
        return (h, w, self.num_bands)

    def __nice__(self):
        return '{}'.format(self.shape)

    def finalize(self, transform=None, dsize=None, interpolation='linear'):
        """
        Execute the final transform
        """
        stack = [comp.finalize(transform=transform, dsize=dsize,
                               interpolation=interpolation)
                 for comp in self.components]
        final = np.concatenate(stack, axis=2)
        return final

    def leafs(self, transform=None, dsize=None, interpolation='linear'):
        """
        Iterate through the leaf nodes

        Example:
            >>> from ndsampler.virtual import *  # NOQA
            >>> import kwimage
            >>> tf = np.array([[0.9, 0, 3.9], [0, 1.1, -.5], [0, 0, 1]])
            >>> raw = kwimage.grab_test_image(dsize=(54, 65))
            >>> raw = kwimage.ensure_float01(raw)
            >>> # Test nested finalize
            >>> layer1 = raw
            >>> num = 10
            >>> for _ in range(num):
            ...     layer1  = VirtualImage(layer1, tf, dsize='auto')
            >>> self = layer1
            >>> leafs = list(self.leafs())
        """
        for comp in self.components:
            yield from comp.leafs(transform=transform, dsize=dsize,
                                  interpolation=interpolation)

    def nesting(self):
        return {
            'shape': self.shape,
            'components': [comp.nesting() for comp in self.components],
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
        warped = VirtualImage(self, transform=transform, dsize=dsize)
        return warped


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

        transform (Transform):
            transforms data from native "sub"-image-space to
            "self"-image-space.

    Example:
        >>> from ndsampler.virtual import *  # NOQA
        >>> dsize = (12, 12)
        >>> tf1 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        >>> tf2 = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]])
        >>> tf3 = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 1]])
        >>> band1 = VirtualImage(np.random.rand(6, 6), tf1, dsize)
        >>> band2 = VirtualImage(np.random.rand(4, 4), tf2, dsize)
        >>> band3 = VirtualImage(np.random.rand(3, 3), tf3, dsize)
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

        # TODO: We probably don't need to track sub-bounds, size, shape
        # or any of that anywhere except at the root and leaf.

        if hasattr(self.sub_data, 'bounds'):
            self.sub_shape = self.sub_data.shape
            self.sub_bounds = self.sub_data.bounds
        else:
            self.sub_shape = self.sub_data.shape
            sub_h, sub_w = self.sub_shape[0:2]
            self.sub_bounds = kwimage.Coords(
                np.array([[0,     0], [sub_w, 0],
                          [0, sub_h], [sub_w, sub_h]])
            )
        self.dsize = None
        self.transform = None
        self.bounds = None

        self._set_transform(transform, dsize)

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

    def _set_transform(self, transform, dsize=ub.NoParam):
        if transform is None:
            transform = np.eye(3)
        if isinstance(transform, Affine):
            transform = transform.matrix
        self.transform = transform
        self.bounds = self.sub_bounds.warp(self.transform)
        if dsize is ub.NoParam:
            pass
        elif dsize is None:
            (h, w) = self.sub_shape[0:2]
            self.dsize = (w, h)
        elif isinstance(dsize, str):
            if dsize == 'auto':
                self.dsize = dsize
                max_xy = np.ceil(self.bounds.data.max(axis=0))
                max_x = int(max_xy[0])
                max_y = int(max_xy[1])
                self.dsize = (max_x, max_y)
            else:
                raise KeyError(dsize)
        else:
            if isinstance(dsize, np.ndarray):
                dsize = tuple(map(int, dsize))
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
        return '{}'.format(self.shape)

    def leafs(self, transform=None, dsize=None, interpolation='linear'):
        """
        Iterate through the leaf nodes

        Example:
            >>> from ndsampler.virtual import *  # NOQA
            >>> import kwimage
            >>> self = VirtualImage.random()
            >>> leafs = list(self.leafs())
            >>> print('leafs = {!r}'.format(leafs))
        """
        if dsize is None:
            dsize = self.dsize
        if transform is None:
            transform = self.transform
        else:
            transform = transform @ self.transform
        sub_data = self.sub_data
        if hasattr(sub_data, 'leafs'):
            # Branch finalize
            yield from sub_data.leafs(
                transform=transform, dsize=dsize, interpolation=interpolation)
        else:
            leaf = {
                'transform': transform,
                'sub_data_shape': sub_data.shape,
                'dsize': dsize,
            }
            yield leaf

    @classmethod
    def random(cls, nesting=(2, 5), rng=None):
        """
        CommandLine:
            xdoctest -m ndsampler.virtual VirtualImage.random

        Example:
            >>> from ndsampler.virtual import *  # NOQA
            >>> self = VirtualImage.random()
            >>> print('self = {!r}'.format(self))
            >>> print(self.nesting())
        """
        import kwarray
        rng = kwarray.ensure_rng(rng)
        num_chan = rng.randint(1, 5)
        leaf_w = rng.randint(8, 64)
        leaf_h = rng.randint(8, 64)
        raw = rng.rand(leaf_h, leaf_w, num_chan)
        layer = raw
        num = rng.randint(*nesting)
        for _ in range(num):
            tf = Affine.random(rng=rng).matrix
            layer  = VirtualImage(layer, tf, dsize='auto')
        self = layer
        return self

    def finalize(self, transform=None, dsize=None, interpolation='linear'):
        """
        Execute the final transform

        Can pass a parent transform to augment this underlying transform.

        Args:
            transform (Transform): an additional transform to perform
            dsize (Tuple[int, int]): overrides destination canvas size

        Example:
            >>> from ndsampler.virtual import *  # NOQA
            >>> import kwimage
            >>> tf = np.array([[0.9, 0, 3.9], [0, 1.1, -.5], [0, 0, 1]])
            >>> raw = kwimage.grab_test_image(dsize=(54, 65))
            >>> raw = kwimage.ensure_float01(raw)
            >>> # Test nested finalize
            >>> layer1 = raw
            >>> num = 10
            >>> for _ in range(num):
            ...     layer1  = VirtualImage(layer1, tf, dsize='auto')
            >>> final1 = layer1.finalize()
            >>> # Test non-nested finalize
            >>> layer2 = VirtualImage(raw, np.eye(3))
            >>> for _ in range(num):
            ...     layer2 = layer2.virtual_warp(tf, dsize='auto')
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
        if transform is None:
            final = sub_data
        elif hasattr(sub_data, 'finalize'):
            # Branch finalize
            final = sub_data.finalize(transform=transform, dsize=dsize,
                                      interpolation=interpolation)
        else:
            # Leaf finalize
            flags = im_cv2._coerce_interpolation(interpolation)
            M = transform[0:2]  # todo allow projective
            final = cv2.warpAffine(sub_data, M, dsize=dsize, flags=flags)

        # Ensure that the last dimension is channels
        final = kwarray.atleast_nd(final, 3, front=False)
        return final

    # def __getitem__(self, region_slices):
    #     return self.virtual_crop(region_slices)

    def virtual_corners(self):
        """

        self = VirtualImage.random(rng=0)
        print(self.nesting())
        region_slices = (slice(40, 90), slice(20, 62))
        region_box = _slice_to_box(region_slices, *self.dsize)
        region_bounds = region_box.to_polygons()[0]

        for leaf in self.leafs():
            pass

        tf_leaf_to_root = leaf['transform']
        tf_root_to_leaf = np.linalg.inv(tf_leaf_to_root)

        leaf_region_bounds = region_bounds.warp(tf_root_to_leaf)
        leaf_region_box = leaf_region_bounds.bounding_box().to_ltrb()
        leaf_crop_box = leaf_region_box.quantize()
        lt_x, lt_y, rb_x, rb_y = leaf_crop_box.data[0, 0:4]

        root_crop_corners = leaf_crop_box.to_polygons()[0].warp(tf_leaf_to_root)

        leaf_crop_slices = (slice(lt_y, rb_y), slice(lt_x, rb_x))

        crop_offset = leaf_crop_box.data[0, 0:2]
        corner_offset = leaf_region_box.data[0, 0:2]
        offset_xy = crop_offset - corner_offset

        tf_root_to_leaf


        # NOTE:

        # Cropping applies a translation in whatever space we do it in
        # We need to save the bounds of the crop.
        # But now we need to adjust the transform so it points to the
        # cropped-leaf-space not just the leaf-space, so we invert the implicit
        # crop

        tf_crop_to_leaf = Affine.affine(offset=crop_offset)

        tf_newroot_to_root = Affine.affine(offset=region_box.data[0, 0:2])
        tf_root_to_newroot = Affine.affine(offset=region_box.data[0, 0:2]).inv()

        tf_crop_to_leaf = Affine.affine(offset=crop_offset)
        tf_crop_to_newroot = tf_root_to_newroot @ tf_leaf_to_root @ tf_crop_to_leaf
        tf_newroot_to_crop = tf_crop_to_newroot.inv()

        tf_leaf_to_crop

        tf_corner_offset = Affine.affine(offset=offset_xy)

        subpixel_offset = Affine.affine(offset=offset_xy).matrix
        tf_crop_to_leaf = subpixel_offset
        tf_crop_to_root = tf_leaf_to_root @ tf_crop_to_leaf
        tf_root_to_crop = np.linalg.inv(tf_crop_to_root)

        if 1:
            import kwplot
            kwplot.autoplt()

            import kwimage
            lw, lh = leaf['sub_data_shape'][0:2]
            leaf_box = kwimage.Boxes([[0, 0, lw, lh]], 'xywh')
            root_box = kwimage.Boxes([[0, 0, self.dsize[0], self.dsize[1]]], 'xywh')

            ax1 = kwplot.figure(fnum=1, pnum=(2, 2, 1), doclf=1).gca()
            ax2 = kwplot.figure(fnum=1, pnum=(2, 2, 2)).gca()
            ax3 = kwplot.figure(fnum=1, pnum=(2, 2, 3)).gca()
            ax4 = kwplot.figure(fnum=1, pnum=(2, 2, 4)).gca()
            root_box.draw(setlim=True, ax=ax1)
            leaf_box.draw(setlim=True, ax=ax2)

            region_bounds.draw(ax=ax1, color='green', alpha=.4)
            leaf_region_bounds.draw(ax=ax2, color='green', alpha=.4)
            leaf_crop_box.draw(ax=ax2, color='purple')
            root_crop_corners.draw(ax=ax1, color='purple', alpha=.4)

            new_w = region_box.to_xywh().data[0, 2]
            new_h = region_box.to_xywh().data[0, 3]
            ax3.set_xlim(0, new_w)
            ax3.set_ylim(0, new_h)

            crop_w = leaf_crop_box.to_xywh().data[0, 2]
            crop_h = leaf_crop_box.to_xywh().data[0, 3]
            ax4.set_xlim(0, crop_w)
            ax4.set_ylim(0, crop_h)

            pts3_ = kwimage.Points.random(3).scale((new_w, new_h))
            pts3 = kwimage.Points(xy=np.vstack([[[0, 0], [5, 5], [0, 49], [40, 45]], pts3_.xy]))
            pts4 = pts3.warp(tf_newroot_to_crop.matrix)
            pts3.draw(ax=ax3)
            pts4.draw(ax=ax4)



        virtual_crop = band2.virtual_crop(region_slices)
        final_crop = virtual_crop.finalize()
        """

    def virtual_crop(self, region_slices):
        """
        Create a new virtual image that performs a crop in the transformed
        "self" space.

        Args:
            region_slices (Tuple[slice, slice]): y-slice and x-slice.

        Returns:
            VirtualImage: lazy executed virtual transform

               The slice is currently not lazy

        Example:
            >>> from ndsampler.virtual import *  # NOQA
            >>> dsize = (12, 12)
            >>> tf2 = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]])
            >>> self = VirtualImage(np.random.rand(4, 4), tf2, dsize)
            >>> region_slices = (slice(5, 10), slice(1, 12))
            >>> virtual_crop = self.virtual_crop(region_slices)
        """
        # todo: this could be done as a virtual layer
        region_box = _slice_to_box(region_slices, *self.dsize)
        region_dsize = tuple(map(int, region_box.to_xywh().data[0, 2:4].tolist()))
        region_bounds = region_box.to_polygons()[0]

        # Transform the region bounds into the sub-image space
        tf_sub_to_self = self.transform
        tf_self_to_sub = np.linalg.inv(tf_sub_to_self)
        sub_region_bounds = region_bounds.warp(tf_self_to_sub)
        sub_region_box = sub_region_bounds.bounding_box().to_ltrb()

        # Quantize to a region that is possible to sample from
        sub_crop_box = sub_region_box.quantize()
        lt_x, lt_y, rb_x, rb_y = sub_crop_box.data[0, 0:4]

        sub_crop_slices = (slice(lt_y, rb_y), slice(lt_x, rb_x))

        # Because we sampled a large quantized region, we need to modify the
        # transform to nudge it a bit to the left, undoing the quantization,
        # which has a bit of extra padding on the left, before applying the
        # final transform.
        crop_offset = sub_crop_box.data[0, 0:2]

        tf_self_to_new = Affine.affine(offset=region_box.data[0, 0:2]).inv().matrix
        tf_sub_to_crop = Affine.affine(offset=crop_offset).inv().matrix

        # corner_offset = sub_region_box.data[0, 0:2]
        # offset_xy = crop_offset - corner_offset
        # offset_x, offset_y = offset_xy
        # subpixel_offset = np.array([
        #     [1., 0., offset_x],
        #     [0., 1., offset_y],
        #     [0., 0., 1.],
        # ], dtype=np.float32)

        # Resample the smaller region to align it with the self region
        # Note: The right most transform is applied first
        # tf_crop_to_self = tf_sub_to_self @ subpixel_offset

        tf_crop_to_new = tf_self_to_new @ tf_sub_to_self @ tf_sub_to_crop
        # np.linalg.inv(tf_crop_to_new)
        # tf_new_to_crop

        if hasattr(self.sub_data, 'virtual_crop'):
            sub_crop = self.sub_data.virtual_crop(sub_crop_slices)
            new = sub_crop.virtual_warp(
                tf_crop_to_new, dsize=region_dsize)
        else:
            crop_sub_data = self.sub_data[sub_crop_slices]
            new = VirtualImage(
                crop_sub_data, tf_crop_to_new, dsize=region_dsize)
        return new


class Affine(ub.NiceRepr):
    """
    Helper for making affine transform matrices.

    Notes:
        Might make sense to move to kwimage

    Example:
        >>> self = Affine(np.eye(3))
        >>> m1 = np.eye(3) @ self
        >>> m2 = self @ np.eye(3)
    """
    def __init__(self, matrix):
        self.matrix = matrix

    def __nice__(self):
        return repr(self.matrix)

    def __array__(self):
        """
        Allow this object to be passed to np.asarray

        References:
            https://numpy.org/doc/stable/user/basics.dispatch.html
        """
        return self.matrix

    @classmethod
    def coerce(cls, data=None, **kwargs):
        if data is None:
            data = kwargs
        if isinstance(data, np.ndarray):
            self = cls(matrix=data)
        elif isinstance(data, dict):
            keys = set(data.keys())
            if 'matrix' in keys:
                self = cls(matrix=np.array(data['matrix']))
            elif len({'scale', 'shear', 'offset', 'theta'} & keys):
                self = cls.affine(**data)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return self

    def decompose(self):
        raise NotImplementedError

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            return Affine(self.matrix @ other)
        else:
            return Affine(self.matrix @ other.matrix)

    def inv(self):
        return Affine(np.linalg.inv(self.matrix))

    @classmethod
    def scale(cls, scale):
        return cls.affine(scale=scale)

    @classmethod
    def random(cls, rng=None, **kw):
        params = cls.random_params(rng=rng, **kw)
        self = cls.affine(**params)
        return self

    @classmethod
    def random_params(cls, rng=None, **kw):
        from kwarray import distributions
        import kwarray
        TN = distributions.TruncNormal
        rng = kwarray.ensure_rng(rng)

        # scale_kw = dict(mean=1, std=1, low=0, high=2)
        # offset_kw = dict(mean=0, std=1, low=-1, high=1)
        # theta_kw = dict(mean=0, std=1, low=-6.28, high=6.28)
        scale_kw = dict(mean=1, std=1, low=1, high=2)
        offset_kw = dict(mean=0, std=1, low=-1, high=1)
        theta_kw = dict(mean=0, std=1, low=-np.pi / 8, high=np.pi / 8)

        scale_dist = TN(**scale_kw, rng=rng)
        offset_dist = TN(**offset_kw, rng=rng)
        theta_dist = TN(**theta_kw, rng=rng)

        # offset_dist = distributions.Constant(0)
        # theta_dist = distributions.Constant(0)

        # todo better parametarization
        params = dict(
            scale=scale_dist.sample(2),
            offset=offset_dist.sample(2),
            theta=theta_dist.sample(),
            shear=0,
            about=0,
        )
        return params

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
            >>> warp_pt1 = (F.matrix @ pt)
            >>> warp_pt2 = (alt.matrix @ pt)
            >>> assert np.allclose(warp_pt2, warp_pt1)

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
            >>> # Define core components of the affine transform
            >>> # scale
            >>> S = np.array([
            >>>     [sx,  0, 0],
            >>>     [ 0, sy, 0],
            >>>     [ 0,  0, 1],
            >>> ])
            >>> # shear
            >>> H = np.array([
            >>>     [1, -sympy.sin(shear), 0],
            >>>     [0,  sympy.cos(shear), 0],
            >>>     [0,                 0, 1],
            >>> ])
            >>> # rotation
            >>> R = np.array([
            >>>     [sympy.cos(theta), -sympy.sin(theta), 0],
            >>>     [sympy.sin(theta),  sympy.cos(theta), 0],
            >>>     [               0,                 0, 1],
            >>> ])
            >>> # translation
            >>> T = np.array([
            >>>     [ 1,  0, tx],
            >>>     [ 0,  1, ty],
            >>>     [ 0,  0,  1],
            >>> ])
            >>> # Contruct the affine 3x3 about the origin
            >>> aff0 = np.array(sympy.simplify(T @ R @ H @ S))
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
        sx_cos_theta = sx * cos_theta
        sx_sin_theta = sx * sin_theta
        sy_sin_shear_p_theta = sy * sin_shear_p_theta
        sy_cos_shear_p_theta = sy * cos_shear_p_theta
        tx_ = tx + x0 - (x0 * sx_cos_theta) + (y0 * sy_sin_shear_p_theta)
        ty_ = ty + y0 - (x0 * sx_sin_theta) - (y0 * sy_cos_shear_p_theta)
        # Sympy simplified expression
        mat = np.array([[sx_cos_theta, -sy_sin_shear_p_theta, tx_],
                        [sx_sin_theta,  sy_cos_shear_p_theta, ty_],
                        [           0,                     0,  1]])
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
