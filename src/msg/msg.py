from scipy.ndimage import convolve
import torch.nn as nn
import numpy as np
import torch

from .utils import modcrop, normalize, resample, rgb2ycbcr


class MSGNet(nn.Module):
    def __init__(self, upsampling_factor, dtype=None):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        # initialize indexes for layers
        self.upsampling_factor = upsampling_factor
        m = int(np.log2(upsampling_factor))
        M = 3 * (m + 1)
        j = np.arange(2, 2 * m - 1, 2)
        j_ = np.arange(3, 2 * m, 2)
        k = np.arange(1, 3 * m, 3)  # deconv indexes
        k_1 = k + 1  # fusion indexes
        k_2 = k + 2  # post-fusion indexes
        k_3 = np.arange(3 * m + 1, M - 1, 3)  # post-fusion indexes

        self.feature_extraction_Y = nn.ModuleList()
        self.upsampling_X = nn.ModuleList()
        # Y-branch
        in_, out = 1, 49
        self.feature_extraction_Y.append(ConvPReLu(in_, out, 7, 1, 3))
        in_, out = 49, 32
        self.feature_extraction_Y.append(ConvPReLu(in_, out))
        in_, out = 32, 32
        for i in range(2, 2 * m):
            if i in j:
                self.feature_extraction_Y.append(ConvPReLu(in_, out))
            if i in j_:
                self.feature_extraction_Y.append(nn.MaxPool2d(3, 2, padding=1))

        # h(D)-branch
        in_, out = 1, 64
        self.feature_extraction_X = ConvPReLu(in_, out, 5, 1, 2)
        j = 0
        in_, out = 64, 32
        for i in range(1, M):
            if i in k:
                self.upsampling_X.append(DeconvPReLu(in_, out, 5, stride=2, padding=2))  # deconvolution
            if i in k_1:
                self.upsampling_X.append(
                    ConvPReLu(in_ * 2, out, 5, stride=1, padding=2))  # convolution for concatenation aka fusion
            if (i in k_2) or (i in k_3):
                self.upsampling_X.append(ConvPReLu(in_, out, 5, stride=1, padding=2))  # post-fusion
            in_, out = 32, 32
        in_, out = 32, 1
        self.upsampling_X.append(ConvPReLu(in_, out, 5, 1, 2))  # reconstruction
        self.to(dtype)

    def forward(self, lr_disparity_normalized, luma_normalized):
        self.weights = next(self.parameters())
        # early spectral decomposition
        h = np.ones((3, 3)) / 9
        im_Dl_LF = np.zeros(lr_disparity_normalized.shape)
        # convolving depth map to get low-frequency features
        # normalizing depth map and saving min_depth max_depth for future test
        for i in range(lr_disparity_normalized.shape[0]):
            im_Dl_LF[i] = convolve(lr_disparity_normalized[i][0], h, mode='reflect')

        in_D = torch.from_numpy(lr_disparity_normalized - im_Dl_LF).to(self.weights)
        # saving low frequency depth map for future test
        self.lf = np.zeros((lr_disparity_normalized.shape[0], lr_disparity_normalized.shape[1],
                            lr_disparity_normalized.shape[2] * self.upsampling_factor,
                            lr_disparity_normalized.shape[3] * self.upsampling_factor))
        for i in range(lr_disparity_normalized.shape[0]):
            self.lf[i] = resample(im_Dl_LF[i][0], self.upsampling_factor)

        # Y-channel
        im_Y_LF = np.zeros(luma_normalized.shape)
        # preprocessing
        # convolving to get low frequency image
        for i in range(luma_normalized.shape[0]):
            im_Y_LF[i] = convolve(luma_normalized[i][0], h, mode='reflect')
        h_Yh = torch.from_numpy(luma_normalized - im_Y_LF).to(self.weights)
        # normalizing image
        for i in range(h_Yh.shape[0]):
            h_Yh[i][0] = (h_Yh[i][0] - torch.min(h_Yh[i][0])) / (torch.max(h_Yh[i][0]) - torch.min(h_Yh[i][0]))

        # forward model
        m = int(np.log2(self.upsampling_factor))
        k = np.arange(0, 3 * m - 1, 3)
        # Y-branch
        self.outputs_Y = [h_Yh]
        for layer in self.feature_extraction_Y:
            self.outputs_Y.append(layer(self.outputs_Y[-1]))
        # h(D)-branch
        self.outputs_X = []
        self.outputs_X.append(self.feature_extraction_X(in_D))
        for i, layer in enumerate(self.upsampling_X):
            self.outputs_X.append(layer(self.outputs_X[-1]))
            if i in k:
                y_ind = 2 * (m - i // 3)
                self.outputs_X.append(torch.cat((self.outputs_Y[y_ind].float(), self.outputs_X[-1]), 1))
        output = self.outputs_X[-1]

        return output

    def test(self, lr_disparity, rgb):
        self.weights = next(self.parameters())
        assert (rgb.ndim == 3) and (rgb.shape[2] == 3)
        rgb = rgb.astype('float32')
        rgb = modcrop(rgb, self.upsampling_factor)
        luma = rgb2ycbcr(rgb)[:, :, 1]
        del rgb
        luma, _, _ = normalize(luma)

        assert lr_disparity.ndim == 2
        lr_disparity, min_d, max_d = normalize(lr_disparity)
        lr_disparity[~np.isfinite(lr_disparity)] = 0.
        h_low, w_low = lr_disparity.shape[:2]

        self.eval()
        with torch.no_grad():
            lr_disparity = lr_disparity[None, None]
            luma = luma[None, None]

            sr_disparity_hf = self(lr_disparity, luma)
            sr_disparity_lf = self.lf

            # compensate for drift
            lr_disparity[..., :h_low // 4, :w_low // 4] = .5
            drift = self(lr_disparity, luma)[..., h_low // 16: h_low // 4 - h_low // 16,
                                             w_low // 16: w_low // 4 - w_low // 16].mean()
            sr_disparity_hf -= drift
        sr_disparity = (sr_disparity_hf.cpu().detach().numpy() + sr_disparity_lf)[0, 0] * (max_d - min_d) + min_d
        sr_disparity[sr_disparity <= 0] = np.nan
        return sr_disparity


class ConvPReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DeconvPReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding,
                                       output_padding=stride - 1)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
