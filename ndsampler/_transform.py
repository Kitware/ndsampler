"""
This file will likely move to kwimage
"""
import ubelt as ub
import numpy as np
import kwarray


class Transform(ub.NiceRepr):
    pass


class Projective(Transform):
    pass


class Affine(Projective):
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
        """
        TODO:
        """
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

    def __imatmul__(self, other):
        if isinstance(other, np.ndarray):
            other_matrix = other
        else:
            other_matrix = other.matrix
        if self.matrix is None:
            self.matrix = other_matrix
        else:
            self.matrix @= other_matrix

    def __matmul__(self, other):
        """
        Example:
            >>> m = {}
            >>> # Works, and returns an Affine
            >>> m[len(m)] = Affine.random() @ np.eye(3)
            >>> m[len(m)] = Affine.random() @ None
            >>> # Works, and returns an ndarray
            >>> m[len(m)] = np.eye(3) @ Affine.random()
            >>> # These do not work
            >>> # m[len(m)] = None @ Affine.random()
            >>> # m[len(m)] = np.eye(3) @ None
            >>> print('m = {}'.format(ub.repr2(m)))
        """
        if other is None:
            return self
        if self.matrix is None:
            return Affine.coerce(other)
        if isinstance(other, np.ndarray):
            return Affine(self.matrix @ other)
        else:
            return Affine(self.matrix @ other.matrix)

    def inv(self):
        return Affine(np.linalg.inv(self.matrix))

    @classmethod
    def eye(cls):
        return cls.affine()

    @classmethod
    def scale(cls, scale):
        return cls.affine(scale=scale)

    @classmethod
    def translate(cls, offset):
        return cls.affine(offset=offset)

    @classmethod
    def rotate(cls, theta):
        return cls.affine(theta=theta)

    @classmethod
    def random(cls, rng=None, **kw):
        params = cls.random_params(rng=rng, **kw)
        self = cls.affine(**params)
        return self

    @classmethod
    def random_params(cls, rng=None, **kw):
        from kwarray import distributions
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
            >>> S = np.array([  # scale
            >>>     [sx,  0, 0],
            >>>     [ 0, sy, 0],
            >>>     [ 0,  0, 1]])
            >>> H = np.array([  # shear
            >>>     [1, -sympy.sin(shear), 0],
            >>>     [0,  sympy.cos(shear), 0],
            >>>     [0,                 0, 1]])
            >>> R = np.array([  # rotation
            >>>     [sympy.cos(theta), -sympy.sin(theta), 0],
            >>>     [sympy.sin(theta),  sympy.cos(theta), 0],
            >>>     [               0,                 0, 1]])
            >>> T = np.array([  # translation
            >>>     [ 1,  0, tx],
            >>>     [ 0,  1, ty],
            >>>     [ 0,  0,  1]])
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
        scale_ = 1 if scale is None else scale
        offset_ = 0 if offset is None else offset
        shear_ = 0 if shear is None else shear
        theta_ = 0 if theta is None else theta
        about_ = 0 if about is None else about
        sx, sy = _ensure_iterablen(scale_, 2)
        tx, ty = _ensure_iterablen(offset_, 2)
        x0, y0 = _ensure_iterablen(about_, 2)

        # Make auxially varables to reduce the number of sin/cos calls
        cos_theta = np.cos(theta_)
        sin_theta = np.sin(theta_)
        cos_shear_p_theta = np.cos(shear_ + theta_)
        sin_shear_p_theta = np.sin(shear_ + theta_)
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


def _ensure_iterablen(scalar, n):
    try:
        iter(scalar)
    except TypeError:
        return [scalar] * n
    return scalar

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/ndsampler/ndsampler/_transform.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
