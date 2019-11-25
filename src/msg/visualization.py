import numpy as np

_white = np.array([255, 255, 255], dtype=np.uint8)


def render_disparity(disparity, calibration=None, only_shading=False, light_color=_white, light_pos=None, light_dir=None):
    """Render the surface corresponding to the disparity map using simple occlusion-less Lambertian rendering.

    Parameters
    ----------
    disparity : array_like
        Disparity map of the shape (h, w).
    calibration : dict, optional
        Intrinsic parameters of the camera:
            cx, cy -- coordinates of the principal point in the image coordinate system in pixels,
            f -- focal length in pixels,
            baseline -- camera baseline
            doffs -- optional disparity offset
    only_shading : bool, optional
        Whether to return shading value in the range 0-1 or 8-bit colors.
    light_color : array_like, optional
        8-bit color of the light source of the shape (3,). White by default.
    light_pos : array_like, optional
        Position of the light source in the camera coordinate system,
         where X is pointing rightward, Y is pointing downward, Z is pointing forward.
        Mutually exclusive with `to_light`.
    light_dir : array_like, optional
        Direction of the lighting in the camera coordinate system.
        Mutually exclusive with `light_pos`.

    Returns
    -------
    rendering : np.ndarray
        Rendered image of the surface
         in the form of (h, w) array of shading values in the range 0-1 if `only_shading` is `True`
         or in the form of (h, w, 3) array of 8-bit colors otherwise.
    """
    assert (light_pos is not None) != (light_dir is not None), \
        'One and only one of `light_pos` and `to_light` should be specified.'
    light_color = np.asarray(light_color)

    surface_coords = disparity_to_absolute_coordinates(disparity, calibration=calibration)
    surface_normals = coords_to_normals(surface_coords)

    if light_dir is not None:
        to_light = -np.asarray(light_dir)
        to_light = to_light / np.linalg.norm(to_light)
        shading = np.clip(np.einsum('ijk,k->ij', surface_normals, to_light), 0, 1)
    else:
        light_pos = np.asarray(light_pos)
        to_light = light_pos[None, None] - surface_coords
        to_light = to_light / np.linalg.norm(to_light, axis=-1, keepdims=True)
        shading = np.clip(np.einsum('ijk,ijk->ij', surface_normals, to_light), 0, 1)

    if only_shading:
        return shading
    else:
        return np.uint8(shading[..., None] * light_color)


def disparity_to_absolute_coordinates(disparity, calibration):
    if 'doffs' in calibration:
        disparity_offset = calibration['doffs']
    else:
        disparity_offset = 0

    f = calibration['f']

    z = calibration['baseline'] * f / (disparity + disparity_offset)

    cx, cy = calibration['cx'], calibration['cy']
    h, w = z.shape
    j, i = np.meshgrid(np.arange(w), np.arange(h), sparse=True)
    valid = np.isfinite(z)
    z = np.where(valid, z, np.nan)
    x = z * (j - cx) / f
    y = z * (i - cy) / f
    return np.dstack((x, y, z))


def coords_to_normals(coords):
    """Calculate surface normals using first order finite-differences.

    Parameters
    ----------
    coords : array_like
        Coordinates of the points (h, w, 3).

    Returns
    -------
    normals : np.ndarray
        Surface normals (h, w, 3).
    """
    dxdu = coords[:, 1:, 0] - coords[:, :-1, 0]
    dydu = coords[:, 1:, 1] - coords[:, :-1, 1]
    dzdu = coords[:, 1:, 2] - coords[:, :-1, 2]
    dxdv = coords[1:, :, 0] - coords[:-1, :, 0]
    dydv = coords[1:, :, 1] - coords[:-1, :, 1]
    dzdv = coords[1:, :, 2] - coords[:-1, :, 2]

    dxdu = np.pad(dxdu, ((0, 0), (0, 1)), mode='edge')
    dydu = np.pad(dydu, ((0, 0), (0, 1)), mode='edge')
    dzdu = np.pad(dzdu, ((0, 0), (0, 1)), mode='edge')
    dxdv = np.pad(dxdv, ((0, 1), (0, 0)), mode='edge')
    dydv = np.pad(dydv, ((0, 1), (0, 0)), mode='edge')
    dzdv = np.pad(dzdv, ((0, 1), (0, 0)), mode='edge')

    n_x = dydv * dzdu - dydu * dzdv
    n_y = dzdv * dxdu - dzdu * dxdv
    n_z = dxdv * dydu - dxdu * dydv

    n = np.dstack((n_x, n_y, n_z))
    norms = np.linalg.norm(n, axis=-1, keepdims=True)
    norms[norms == 0] = 1
    n = n / norms
    return n
