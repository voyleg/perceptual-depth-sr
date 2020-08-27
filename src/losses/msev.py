from torch.nn import MSELoss

from depth_utils.reprojections import depth_to_absolute_coordinates
from depth_utils.normals import coords_to_normals


class MSEv(MSELoss):
    def __init__(self, depth_type, reduction='mean'):
        r"""

        Parameters
        ----------
        depth_type : str
            Type of the depth map, one of 'perspective' -- meaning the distance from point to camera,
             'orthogonal' -- meaning the distance from point to image plane, or 'disparity'.
        reduction : str, optional
            See `torch.MSELoss`
        """
        super().__init__(reduction=reduction)
        self.depth_type = depth_type

    def forward(self, output_depth, target_depth, K=None, calibration=None):
        """Calculate MSEv loss.

        Parameters
        ----------
        K : array_like, optional
            Camera projection matrix.
        calibration : dict, optional
            Intrinsic parameters of the camera:
                cx, cy -- coordinates of the principal point in the image coordinate system in pixels,
                f -- focal length in pixels,
                baseline, required for disparity -- baseline of the camera in metric units.
            Either `K` or `calibration` is required.
        """
        output_normals = depth_to_absolute_coordinates(output_depth, depth_type=self.depth_type, K=K,
                                                       calibration=calibration)
        output_normals = coords_to_normals(output_normals)
        target_normals = depth_to_absolute_coordinates(target_depth, depth_type=self.depth_type, K=K,
                                                       calibration=calibration)
        target_normals = coords_to_normals(target_normals)
        loss = super().forward(output_normals, target_normals)
        if self.reduction == 'none':
            loss = loss.mean(-3, keepdim=True)
        return loss
