"""
The classes in this file represent a tree of virtual operations.

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

Example:
    >>> # Example demonstrating the modivating use case
    >>> # We have multiple aligned frames for a video, but each of
    >>> # those frames is in a different resolution. Furthermore,
    >>> # each of the frames consists of channels in different resolutions.
    >>> from ndsampler.virtual import *  # NOQA
    >>> rng = kwarray.ensure_rng(None)
    >>> def demo_chan(key, dsize, chan):
    ...     return kwimage.grab_test_image(key, dsize=dsize)[..., chan]
    >>> # Create raw channels in some "native" resolution for frame 1
    >>> f1_chan1 = VirtualWarp(demo_chan('astro', (300, 300), 0))
    >>> f1_chan2 = VirtualWarp(demo_chan('astro', (200, 200), 1))
    >>> f1_chan3 = VirtualWarp(demo_chan('astro', (10, 10), 2))
    >>> # Create raw channels in some "native" resolution for frame 2
    >>> f2_chan1 = VirtualWarp(demo_chan('carl', (64, 64), 0))
    >>> f2_chan2 = VirtualWarp(demo_chan('carl', (260, 260), 1))
    >>> f2_chan3 = VirtualWarp(demo_chan('carl', (10, 10), 2))
    >>> #
    >>> # Virtual warp each channel into its "image" space
    >>> # Note: the images never actually enter this space we transform through it
    >>> f1_dsize = np.array((3, 3))
    >>> f2_dsize = np.array((2, 2))
    >>> f1_img = VirtualChannelConcat([
    >>>     f1_chan1.virtual_warp(Affine.scale(f1_dsize / f1_chan1.dsize), dsize=f1_dsize),
    >>>     f1_chan2.virtual_warp(Affine.scale(f1_dsize / f1_chan2.dsize), dsize=f1_dsize),
    >>>     f1_chan3.virtual_warp(Affine.scale(f1_dsize / f1_chan3.dsize), dsize=f1_dsize),
    >>> ])
    >>> f2_img = VirtualChannelConcat([
    >>>     f2_chan1.virtual_warp(Affine.scale(f2_dsize / f2_chan1.dsize), dsize=f2_dsize),
    >>>     f2_chan2.virtual_warp(Affine.scale(f2_dsize / f2_chan2.dsize), dsize=f2_dsize),
    >>>     f2_chan3.virtual_warp(Affine.scale(f2_dsize / f2_chan3.dsize), dsize=f2_dsize),
    >>> ])
    >>> # Combine frames into a video
    >>> vid_dsize = np.array((280, 280))
    >>> vid = VirtualFrameConcat([
    >>>     f1_img.virtual_warp(Affine.scale(vid_dsize / f1_img.dsize), dsize=vid_dsize),
    >>>     f2_img.virtual_warp(Affine.scale(vid_dsize / f2_img.dsize), dsize=vid_dsize),
    >>> ])
    >>> vid.nesting
    >>> print('vid.nesting = {}'.format(ub.repr2(vid.nesting(), nl=-1)))
    >>> final = vid.finalize(interpolation='nearest')
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> kwplot.imshow(final[0], pnum=(1, 2, 1), fnum=1)
    >>> kwplot.imshow(final[1], pnum=(1, 2, 2), fnum=1)
"""
import ubelt as ub
import numpy as np
import kwimage
import kwarray
from ndsampler._transform import Affine


class VirtualOperation(ub.NiceRepr):
    """
    Base class for nodes in a tree of virtual operations
    """
    def __nice__(self):
        return '{}'.format(self.shape)

    def children(self):
        """
        Abstract method, which should generate all of the direct children of a
        node in the operation tree.
        """
        raise NotImplementedError

    def virtual_leafs(self, **kwargs):
        """
        Iterate through the leaf nodes, which are virtually transformed into
        the root space.
        """
        for child in self.children:
            yield from child.virtual_leafs(**kwargs)

    def nesting(self):
        # # for child in self.children()
        # if isinstance(self.sub_data, np.ndarray):
        #     return {
        #         'shape': self.shape,
        #         'sub_data': {'shape': self.sub_data.shape},
        #     }
        # else:
        #     return {
        #         'shape': self.shape,
        #         'sub_data': self.sub_data.nesting(),
        #     }
        return {
            'type': self.__class__.__name__,
            'shape': self.shape,
            'children': [child.nesting() for child in self.children()],
        }


class VirtualVideoOperation(VirtualOperation):
    pass


class VirtualImageOperation(VirtualOperation):
    """
    Operations that pertain only to images
    """

    def virtual_crop(self, region_slices):
        """
        Create a new virtual image that performs a crop in the transformed
        "self" space.

        Args:
            region_slices (Tuple[slice, slice]): y-slice and x-slice.

        Returns:
            VirtualWarp: lazy executed virtual transform

               The slice is currently not lazy

        Example:
            >>> from ndsampler.virtual import *  # NOQA
            >>> dsize = (100, 100)
            >>> tf2 = Affine.affine(scale=3).matrix
            >>> self = VirtualWarp(np.random.rand(33, 33), tf2, dsize)
            >>> region_slices = (slice(5, 10), slice(1, 12))
            >>> virtual_crop = self.virtual_crop(region_slices)
            >>> virtual_crop.finalize()
        """
        components = []
        for virtual_leaf in self.virtual_leafs():
            # Compute, sub_crop_slices, and new tf_newleaf_to_newroot
            tf_leaf_to_root = virtual_leaf.transform

            root_region_box = kwimage.Boxes.from_slice(
                region_slices, shape=virtual_leaf.shape)
            root_region_bounds = root_region_box.to_polygons()[0]

            leaf_crop_slices, tf_newleaf_to_newroot = _compute_leaf_subcrop(
                root_region_bounds, tf_leaf_to_root)

            crop = VirtualCrop(virtual_leaf.sub_data, leaf_crop_slices)
            warp = VirtualWarp(crop, tf_newleaf_to_newroot)
            components.append(warp)

        if len(components) == 1:
            return components[0]
        else:
            return VirtualChannelConcat(components)

    def virtual_warp(self, transform, dsize=None):
        """
        Virtually transform the underlying data.

        Note:
            this deviates from kwimage warp functions because instead of
            "output_dims" (specified in c-style shape) we specify dsize (w, h).

        Returns:
            VirtualWarp : new virtual transform a chained transform
        """
        # if 0:
        #     # note, we are modifying the transform, but we could
        #     # just add a new virtual transform layer on top of this.
        #     sub_data = self.sub_data
        #     new_transform = transform @ self.transform
        #     warped = VirtualWarp(sub_data, new_transform, dsize)
        warped = VirtualWarp(self, transform=transform, dsize=dsize)
        return warped


class VirtualFrameConcat(VirtualVideoOperation):
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

    TODO:
        - [ ] Support computing the transforms when none of the data is loaded

    Example:
        >>> # Simpler case with fewer nesting levels
        >>> from ndsampler.virtual import *  # NOQA
        >>> rng = kwarray.ensure_rng(None)
        >>> def demo_img(key, dsize):
        ...     return kwimage.grab_test_image(key, dsize=dsize)
        >>> # Virtual warp each channel into its "image" space
        >>> # Note: the images never actually enter this space we transform through it
        >>> f1_img = VirtualWarp(demo_img('astro', (300, 300)))
        >>> f2_img = VirtualWarp(demo_img('carl', (256, 256)))
        >>> # Combine frames into a video
        >>> vid_dsize = np.array((100, 100))
        >>> self = vid = VirtualFrameConcat([
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

    def children(self):
        yield from self.frames

    @property
    def shape(self):
        w, h = self.dsize
        return (self.num_frames, h, w, self.num_bands)

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

    # def nesting(self):
    #     return {
    #         'shape': self.shape,
    #         'frames': [frame.nesting() for frame in self.frames],
    #     }

    # def virtual_crop(self, region_slices):
    #     """
    #     Create a new virtual image that performs a crop in the transformed
    #     "self" space.

    #     Args:
    #         region_slices (Tuple[slice, slice]): y-slice and x-slice.

    #     Returns:
    #         VirtualWarp: lazy executed virtual transform
    #            The slice is currently not lazy
    #     """
    #     # todo: this could be done as a virtual layer
    #     region_box = kwimage.Boxes.from_slice(region_slices, shape=self.shape)
    #     raise NotImplementedError


class VirtualChannelConcat(VirtualImageOperation):
    """
    Represents multiple channels in an image that could be concatenated

    Attributes:
        components (List[VirtualWarp]): a list of stackable channels. Each
            component may be comprised of multiple channels.

    TODO:
        - [ ] can this be generalized into a virtual concat?
        - [ ] can all concats be delayed until the very end?

    Example:
        >>> comp1 = VirtualWarp(np.random.rand(11, 7))
        >>> comp2 = VirtualWarp(np.random.rand(11, 7, 3))
        >>> comp3 = VirtualWarp(
        >>>     np.random.rand(3, 5, 2),
        >>>     transform=Affine.affine(scale=(7/5, 11/3)).matrix,
        >>>     dsize=(7, 11)
        >>> )
        >>> components = [comp1, comp2, comp3]
        >>> chans = VirtualChannelConcat(components)
        >>> final = chans.finalize()
        >>> assert final.shape == chans.shape
        >>> assert final.shape == (11, 7, 6)

        >>> # We should be able to nest VirtualChannelConcat inside virutal images
        >>> frame1 = VirtualWarp(
        >>>     chans, transform=Affine.affine(scale=2.2).matrix,
        >>>     dsize=(20, 26))
        >>> frame2 = VirtualWarp(
        >>>     np.random.rand(3, 3, 6), dsize=(20, 26))
        >>> frame3 = VirtualWarp(
        >>>     np.random.rand(3, 3, 6), dsize=(20, 26))

        >>> print(ub.repr2(frame1.nesting(), nl=-1, sort=False))
        >>> frame1.finalize()
        >>> vid = VirtualFrameConcat([frame1, frame2, frame3])
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

    def children(self):
        yield from self.components

    @classmethod
    def random(cls, num_parts=3, rng=None):
        """
        CommandLine:
            xdoctest -m ndsampler.virtual VirtualWarp.random

        Example:
            >>> self = VirtualChannelConcat.random()
            >>> print('self = {!r}'.format(self))
            >>> print(self.nesting())
        """
        rng = kwarray.ensure_rng(rng)
        self_w = rng.randint(8, 64)
        self_h = rng.randint(8, 64)
        components = []
        for _ in range(num_parts):
            subcomp = VirtualWarp.random(rng=rng)
            tf = Affine.random(rng=rng).matrix
            comp = subcomp.virtual_warp(tf, dsize=(self_w, self_h))
            components.append(comp)
        self = VirtualChannelConcat(components)
        return self

    @property
    def shape(self):
        w, h = self.dsize
        return (h, w, self.num_bands)

    def finalize(self, transform=None, dsize=None, interpolation='linear'):
        """
        Execute the final transform
        """
        stack = [comp.finalize(transform=transform, dsize=dsize,
                               interpolation=interpolation)
                 for comp in self.components]
        final = np.concatenate(stack, axis=2)
        return final

    # def virtual_leafs(self, **kwargs):
    #     """
    #     Iterate through the leaf nodes

    #     Flattens the operation tree into a separate node for each leaf.

    #     Concatenation information information is lost because operations
    #     (but this is ok when operations are communative)

    #     Example:
    #         >>> from ndsampler.virtual import *  # NOQA
    #         >>> self  = VirtualChannelConcat.random()
    #         >>> leafs = list(self.virtual_leafs())
    #         >>> print('leafs = {!r}'.format(leafs))
    #     """
    #     for comp in self.components:
    #         yield from comp.virtual_leafs(transform=transform, dsize=dsize,
    #                                       interpolation=interpolation)

    # def nesting(self):
    #     return {
    #         'shape': self.shape,
    #         'components': [comp.nesting() for comp in self.components],
    #     }


class VirtualWarp(VirtualImageOperation):
    """
    POC for chainable transforms

    Notes:
        "sub" is used to refer to the underlying data in its native coordinates
        and resolution.

        "self" is used to refer to the data in the transformed coordinates that
        are exposed by this class.

    Attributes:

        sub_data (VirtualWarp | ArrayLike):
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
        >>> band1 = VirtualWarp(np.random.rand(6, 6), tf1, dsize)
        >>> band2 = VirtualWarp(np.random.rand(4, 4), tf2, dsize)
        >>> band3 = VirtualWarp(np.random.rand(3, 3), tf3, dsize)
        >>> #
        >>> # Execute a crop in a one-level transformed space
        >>> region_slices = (slice(5, 10), slice(0, 12))
        >>> virtual_crop = band2.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.finalize()
        >>> #
        >>> # Execute a crop in a nested transformed space
        >>> tf4 = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
        >>> chained = VirtualWarp(band2, tf4, (18, 18))
        >>> virtual_crop = chained.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.finalize()
        >>> #
        >>> tf4 = np.array([[.5, 0, 0], [0, .5, 0], [0, 0, 1]])
        >>> chained = VirtualWarp(band2, tf4, (6, 6))
        >>> virtual_crop = chained.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.finalize()
        >>> #
        >>> region_slices = (slice(1, 5), slice(2, 4))
        >>> virtual_crop = chained.virtual_crop(region_slices)
        >>> final_crop = virtual_crop.finalize()

    Example:
        >>> dsize = (17, 12)
        >>> tf = np.array([[5.2, 0, 1.1], [0, 3.1, 2.2], [0, 0, 1]])
        >>> self = VirtualWarp(np.random.rand(3, 5, 13), tf, dsize=dsize)
        >>> self.finalize().shape
    """
    def __init__(self, sub_data, transform=None, dsize=None):
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

        if len(self.sub_data.shape) == 2:
            num_bands = 1
        elif len(self.sub_data.shape) == 3:
            num_bands = self.sub_data.shape[2]
        else:
            raise ValueError(
                'Data may only have 2 space dimensions and 1 channel '
                'dimension')
        self.num_bands = num_bands

    def children(self):
        yield self.sub_data

    @property
    def shape(self):
        # trailing_shape = self.sub_data.shape[2:]
        # trailing shape should only be allowed to have 0 or 1 dimension
        w, h = self.dsize
        return (h, w, self.num_bands)

    def nesting(self):
        if isinstance(self.sub_data, np.ndarray):
            return {
                'shape': self.shape,
                'sub_data': {'shape': self.sub_data.shape, 'type': 'ndarray'},
            }
        else:
            return {
                'shape': self.shape,
                'sub_data': self.sub_data.nesting(),
            }

    def virtual_leafs(self, **kwargs):
        """

        Example:
            >>> from ndsampler.virtual import *  # NOQA
            >>> self = VirtualWarp.random()
            >>> leafs = list(self.virtual_leafs())
            >>> print('leafs = {!r}'.format(leafs))
        """
        dsize = kwargs.get('dsize', None)
        transform = kwargs.get('transform', None)
        if dsize is None:
            dsize = self.dsize
        if transform is None:
            transform = self.transform
        else:
            transform = kwargs.get('transform', None) @ self.transform
        kwargs['dsize'] = dsize
        kwargs['transform'] = transform
        sub_data = self.sub_data
        if hasattr(sub_data, 'virtual_leafs'):
            # Branch finalize
            yield from sub_data.virtual_leafs(
                transform=transform, dsize=dsize)
        else:
            leaf = VirtualWarp(sub_data, transform, dsize=dsize)
            yield leaf

    @classmethod
    def random(cls, nesting=(2, 5), rng=None):
        """
        CommandLine:
            xdoctest -m ndsampler.virtual VirtualWarp.random

        Example:
            >>> from ndsampler.virtual import *  # NOQA
            >>> self = VirtualWarp.random()
            >>> print('self = {!r}'.format(self))
            >>> print(self.nesting())
        """
        rng = kwarray.ensure_rng(rng)
        num_chan = rng.randint(1, 5)
        leaf_w = rng.randint(8, 64)
        leaf_h = rng.randint(8, 64)
        raw = rng.rand(leaf_h, leaf_w, num_chan)
        layer = raw
        num = rng.randint(*nesting)
        for _ in range(num):
            tf = Affine.random(rng=rng).matrix
            layer  = VirtualWarp(layer, tf, dsize='auto')
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
            >>> tf = np.array([[0.9, 0, 3.9], [0, 1.1, -.5], [0, 0, 1]])
            >>> raw = kwimage.grab_test_image(dsize=(54, 65))
            >>> raw = kwimage.ensure_float01(raw)
            >>> # Test nested finalize
            >>> layer1 = raw
            >>> num = 10
            >>> for _ in range(num):
            ...     layer1  = VirtualWarp(layer1, tf, dsize='auto')
            >>> final1 = layer1.finalize()
            >>> # Test non-nested finalize
            >>> layer2 = VirtualWarp(raw, np.eye(3))
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


class VirtualCrop(VirtualImageOperation):
    """
    Represent a delayed crop operation
    """
    def __init__(self, sub_data, sub_slices):
        self.sub_data = sub_data
        self.sub_slices = sub_slices

        sl_x, sl_y = sub_slices[0:2]
        width = sl_x.stop - sl_x.start
        height = sl_y.stop - sl_y.start
        num_bands = kwimage.num_channels(self.sub_data)
        self.num_bands = num_bands
        self.shape = (height, width, num_bands)

    def children(self):
        yield self.sub_data

    def nesting(self):
        if isinstance(self.sub_data, np.ndarray):
            return {
                'shape': self.shape,
                'sub_slices': self.sub_slices,
                'sub_data': {'shape': self.sub_data.shape},
            }
        else:
            return {
                'shape': self.shape,
                'sub_slices': self.sub_slices,
                'sub_data': self.sub_data.nesting(),
            }

    def finalize(self, **kwargs):
        if hasattr(self.sub_data, 'finalize'):
            return self.sub_data.finalize(**kwargs)[self.sub_slices]
        else:
            return self.sub_data[self.sub_slices]


# def _compute_subcrop(root_region_slices, root_shape):
#     tf_leaf_to_root = self.transform
#     root_region_box = kwimage.Boxes.from_slice(root_region_slices, shape=self.shape)
#     region_dsize = tuple(map(int, region_box.to_xywh().data[0, 2:4].tolist()))
#     root_region_bounds = root_region_box.to_polygons()[0]


def _compute_leaf_subcrop(root_region_bounds, tf_leaf_to_root):
    r"""
    Given a region in a "root" image and a trasnform between that "root" and
    some "leaf" image, compute the appropriate quantized region in the "leaf"
    image and the adjusted transformation between that root and leaf.

    Example:
        >>> region_slices = (slice(33, 100), slice(22, 62))
        >>> region_shape = (100, 100, 1)
        >>> root_region_box = kwimage.Boxes.from_slice(region_slices, shape=region_shape)
        >>> root_region_bounds = root_region_box.to_polygons()[0]
        >>> tf_leaf_to_root = Affine.affine(scale=7).matrix
        >>> slices, tf_new = _compute_leaf_subcrop(root_region_bounds, tf_leaf_to_root)
        >>> print('tf_new =\n{!r}'.format(tf_new))
        >>> print('slices = {!r}'.format(slices))

    """
    # Transform the region bounds into the sub-image space
    tf_root_to_leaf = np.linalg.inv(tf_leaf_to_root)
    leaf_region_bounds = root_region_bounds.warp(tf_root_to_leaf)
    leaf_region_box = leaf_region_bounds.bounding_box().to_ltrb()

    # Quantize to a region that is possible to sample from
    leaf_crop_box = leaf_region_box.quantize()

    # Because we sampled a large quantized region, we need to modify the
    # transform to nudge it a bit to the left, undoing the quantization,
    # which has a bit of extra padding on the left, before applying the
    # final transform.
    # subpixel_offset = leaf_region_box.data[0, 0:2]
    crop_offset = leaf_crop_box.data[0, 0:2]
    root_offset = root_region_bounds.exterior.data.min(axis=0)

    tf_root_to_newroot = Affine.affine(offset=root_offset).inv().matrix
    tf_newleaf_to_leaf = Affine.affine(offset=crop_offset).matrix

    # Resample the smaller region to align it with the root region
    # Note: The right most transform is applied first
    tf_newleaf_to_newroot = (
        tf_root_to_newroot @
        tf_leaf_to_root @
        tf_newleaf_to_leaf
    )

    lt_x, lt_y, rb_x, rb_y = leaf_crop_box.data[0, 0:4]
    leaf_crop_slices = (slice(lt_y, rb_y), slice(lt_x, rb_x))

    return leaf_crop_slices, tf_newleaf_to_newroot


def _devcheck_corner():
    self = VirtualWarp.random(rng=0)
    print(self.nesting())
    region_slices = (slice(40, 90), slice(20, 62))
    region_box = kwimage.Boxes.from_slice(region_slices, shape=self.shape)
    region_bounds = region_box.to_polygons()[0]

    for leaf in self.virtual_leafs():
        pass

    tf_leaf_to_root = leaf['transform']
    tf_root_to_leaf = np.linalg.inv(tf_leaf_to_root)

    leaf_region_bounds = region_bounds.warp(tf_root_to_leaf)
    leaf_region_box = leaf_region_bounds.bounding_box().to_ltrb()
    leaf_crop_box = leaf_region_box.quantize()
    lt_x, lt_y, rb_x, rb_y = leaf_crop_box.data[0, 0:4]

    root_crop_corners = leaf_crop_box.to_polygons()[0].warp(tf_leaf_to_root)

    # leaf_crop_slices = (slice(lt_y, rb_y), slice(lt_x, rb_x))

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

    # tf_newroot_to_root = Affine.affine(offset=region_box.data[0, 0:2])
    tf_root_to_newroot = Affine.affine(offset=region_box.data[0, 0:2]).inv()

    tf_crop_to_leaf = Affine.affine(offset=crop_offset)
    tf_crop_to_newroot = tf_root_to_newroot @ tf_leaf_to_root @ tf_crop_to_leaf
    tf_newroot_to_crop = tf_crop_to_newroot.inv()

    # tf_leaf_to_crop
    # tf_corner_offset = Affine.affine(offset=offset_xy)

    subpixel_offset = Affine.affine(offset=offset_xy).matrix
    tf_crop_to_leaf = subpixel_offset
    # tf_crop_to_root = tf_leaf_to_root @ tf_crop_to_leaf
    # tf_root_to_crop = np.linalg.inv(tf_crop_to_root)

    if 1:
        import kwplot
        kwplot.autoplt()

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

    # virtual_crop = band2.virtual_crop(region_slices)
    # final_crop = virtual_crop.finalize()
