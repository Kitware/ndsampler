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
