

def warp_image_test(image, transform, dsize=None):
    """

    from kwimage.transform import Affine
    import kwimage
    image = kwimage.grab_test_image('checkerboard', dsize=(2048, 2048)).astype(np.float32)
    image = kwimage.grab_test_image('astro', dsize=(2048, 2048))
    transform = Affine.random() @ Affine.scale(0.01)

    """
    from kwimage.transform import Affine
    import kwimage
    import numpy as np
    import ubelt as ub

    # Choose a random affine transform that probably has a small scale
    # transform = Affine.random() @ Affine.scale((0.3, 2))
    # transform = Affine.scale((0.1, 1.2))
    # transform = Affine.scale(0.05)
    transform = Affine.random() @ Affine.scale(0.01)
    # transform = Affine.random()

    image = kwimage.grab_test_image('astro')
    image = kwimage.grab_test_image('checkerboard')

    image = kwimage.ensure_float01(image)

    from kwimage import im_cv2
    import kwarray
    import cv2
    transform = Affine.coerce(transform)

    if 1 or dsize is None:
        h, w = image.shape[0:2]

        boxes = kwimage.Boxes(np.array([[0, 0, w, h]]), 'xywh')
        poly = boxes.to_polygons()[0]
        warped_poly = poly.warp(transform.matrix)
        warped_box = warped_poly.to_boxes().to_ltrb().quantize()
        dsize = tuple(map(int, warped_box.data[0, 2:4]))

    import timerit
    ti = timerit.Timerit(10, bestof=3, verbose=2)

    def _full_gauss_kernel(k0, sigma0, scale):
        num_downscales = np.log2(1 / scale)
        if num_downscales < 0:
            return 1, 0

        # Define b0 = kernel size for one downsample operation
        b0 = 5
        # Define s0 = sigma for one downsample operation
        s0 = 1

        # The kernel size and sigma doubles for each 2x downsample
        k = int(np.ceil(b0 * (2 ** (num_downscales - 1))))
        sigma = s0 * (2 ** (num_downscales - 1))

        if k % 2 == 0:
            k += 1

        return k, sigma

    def pyrDownK(a, k=1):
        assert k >= 0
        for _ in range(k):
            a = cv2.pyrDown(a)
        return a

    for timer in ti.reset('naive'):
        with timer:
            interpolation = 'nearest'
            flags = im_cv2._coerce_interpolation(interpolation)
            final_v5 = cv2.warpAffine(image, transform.matrix[0:2], dsize=dsize, flags=flags)

    # --------------------
    # METHOD 1
    #
    for timer in ti.reset('resize+warp'):
        with timer:
            params = transform.decompose()

            sx, sy = params['scale']
            noscale_params = ub.dict_diff(params, {'scale'})
            noscale_warp = Affine.affine(**noscale_params)

            h, w = image.shape[0:2]
            resize_dsize = (int(np.ceil(sx * w)), int(np.ceil(sy * h)))

            downsampled = cv2.resize(image, dsize=resize_dsize, fx=sx, fy=sy,
                                     interpolation=cv2.INTER_AREA)

            interpolation = 'linear'
            flags = im_cv2._coerce_interpolation(interpolation)
            final_v1 = cv2.warpAffine(downsampled, noscale_warp.matrix[0:2], dsize=dsize, flags=flags)

    # --------------------
    # METHOD 2
    for timer in ti.reset('fullblur+warp'):
        with timer:
            k_x, sigma_x = _full_gauss_kernel(k0=5, sigma0=1, scale=sx)
            k_y, sigma_y = _full_gauss_kernel(k0=5, sigma0=1, scale=sy)
            image_ = image.copy()
            image_ = cv2.GaussianBlur(image_, (k_x, k_y), sigma_x, sigma_y)
            image_ = kwarray.atleast_nd(image_, 3)
            # image_ = image_.clip(0, 1)

            interpolation = 'linear'
            flags = im_cv2._coerce_interpolation(interpolation)
            final_v2 = cv2.warpAffine(image_, transform.matrix[0:2], dsize=dsize, flags=flags)

    # --------------------
    # METHOD 3

    for timer in ti.reset('pyrDown+blur+warp'):
        with timer:
            temp = image.copy()
            params = transform.decompose()
            sx, sy = params['scale']

            biggest_scale = max(sx, sy)
            # The -2 allows the gaussian to be a little bigger. This
            # seems to help with border effects at only a small runtime cost
            num_downscales = max(int(np.log2(1 / biggest_scale)) - 2, 0)
            pyr_scale = 1 / (2 ** num_downscales)

            # Does the gaussian downsampling
            temp = pyrDownK(image, num_downscales)

            rest_sx = sx / pyr_scale
            rest_sy = sy / pyr_scale

            partial_scale = Affine.scale((rest_sx, rest_sy))
            rest_warp = noscale_warp @ partial_scale

            k_x, sigma_x = _full_gauss_kernel(k0=5, sigma0=1, scale=rest_sx)
            k_y, sigma_y = _full_gauss_kernel(k0=5, sigma0=1, scale=rest_sy)
            temp = cv2.GaussianBlur(temp, (k_x, k_y), sigma_x, sigma_y)
            temp = kwarray.atleast_nd(temp, 3)

            interpolation = 'cubic'
            flags = im_cv2._coerce_interpolation(interpolation)
            final_v3 = cv2.warpAffine(temp, rest_warp.matrix[0:2], dsize=dsize,
                                      flags=flags)

    # --------------------
    # METHOD 4 - dont do the final blur

    for timer in ti.reset('pyrDown+warp'):
        with timer:
            temp = image.copy()
            params = transform.decompose()
            sx, sy = params['scale']

            biggest_scale = max(sx, sy)
            num_downscales = max(int(np.log2(1 / biggest_scale)), 0)
            pyr_scale = 1 / (2 ** num_downscales)

            # Does the gaussian downsampling
            temp = pyrDownK(image, num_downscales)

            rest_sx = sx / pyr_scale
            rest_sy = sy / pyr_scale

            partial_scale = Affine.scale((rest_sx, rest_sy))
            rest_warp = noscale_warp @ partial_scale

            interpolation = 'linear'
            flags = im_cv2._coerce_interpolation(interpolation)
            final_v4 = cv2.warpAffine(temp, rest_warp.matrix[0:2], dsize=dsize, flags=flags)

    if 1:

        def get_title(key):
            from ubelt.timerit import _choose_unit
            value = ti.measures['mean'][key]
            suffix, mag = _choose_unit(value)
            unit_val = value / mag

            return key + ' ' + ub.repr2(unit_val, precision=2) + ' ' + suffix

        final_v2 = final_v2.clip(0, 1)
        final_v1 = final_v1.clip(0, 1)
        final_v3 = final_v3.clip(0, 1)
        final_v4 = final_v4.clip(0, 1)
        final_v5 = final_v5.clip(0, 1)
        import kwplot
        kwplot.autompl()
        kwplot.imshow(final_v5, pnum=(1, 5, 1), title=get_title('naive'))
        kwplot.imshow(final_v2, pnum=(1, 5, 2), title=get_title('fullblur+warp'))
        kwplot.imshow(final_v1, pnum=(1, 5, 3), title=get_title('resize+warp'))
        kwplot.imshow(final_v3, pnum=(1, 5, 4), title=get_title('pyrDown+blur+warp'))
        kwplot.imshow(final_v4, pnum=(1, 5, 5), title=get_title('pyrDown+warp'))
        # kwplot.imshow(np.abs(final_v2 - final_v1), pnum=(1, 4, 4))
