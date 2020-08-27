import torch


def depth_to_absolute_coordinates(depth, depth_type, K=None, calibration=None):
    """Convert depth map to absolute coordinates.

    Parameters
    ----------
    depth : array_like
        Depth map of shape (h, w) or tensor of depth maps of shape (**, 1, h, w).
    depth_type : str
        Type of the depth map, one of 'perspective' -- meaning the distance from point to camera,
         'orthogonal' -- meaning the distance from point to image plane, or 'disparity'.
    K : array_like, optional
        Camera projection matrix.
    calibration : dict, optional if `type` is not 'disparity'
        Intrinsic parameters of the camera:
            cx, cy -- coordinates of the principal point in the image coordinate system (i.e in pixels),
            f -- focal length in pixels,
            baseline, required for disparity -- baseline of the camera in metric units.
        Either `K` or `calibration` is required.
    Returns
    -------
    coordinates : torch.Tensor
        Coordinates of the points (**, 3, h, w) in the camera coordinate system.
        X is pointing rightward, Y is pointing downward, Z is pointing forward.
    """
    depth = torch.as_tensor(depth)
    dtype = depth.dtype
    h, w = depth.shape[-2:]

    if K is not None:
        K = torch.as_tensor(K, dtype=dtype)
    else:
        K = torch.zeros(3, 3, dtype=dtype)
        K[0, 0] = K[1, 1] = float(calibration['f'])
        K[2, 2] = 1
        K[0, 2] = float(calibration['cx'])
        K[1, 2] = float(calibration['cy'])

    v, u = torch.meshgrid(torch.arange(h, dtype=dtype) + .5, torch.arange(w, dtype=dtype) + .5)
    if depth.ndim < 3:  # ensure depth has channel dimension
        depth = depth[None]
    ones = torch.ones_like(v)
    points = torch.einsum('lk,kij->lij', K.inverse(), torch.stack([u, v, ones]))
    if depth_type == 'perspective':
        points = torch.nn.functional.normalize(points, dim=-3)
        points = points.to(depth) * depth
    elif depth_type == 'orthogonal':
        points = points / points[2:3]
        points = points.to(depth) * depth
    elif depth_type == 'disparity':
        points = points / points[2:3]
        z = calibration['baseline'] * K[0, 0] / depth
        points = points.to(depth) * z
    else:
        raise ValueError(f'Unknown type {depth_type}')
    return points
