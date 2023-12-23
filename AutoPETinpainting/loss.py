import torch
import torch.nn as nn
import torchvision.models as models
from kornia.geometry.transform import build_laplacian_pyramid


def valid_loss_fn(i_out, i_gt, mask):
    valid_loss = torch.mean(torch.abs(mask * (i_out - i_gt)))

    return valid_loss


def hole_loss_fn(i_out, i_gt, mask):
    hole_loss = torch.mean(torch.abs((1 - mask) * (i_out - i_gt)))

    return hole_loss


def tv_loss_fn(i_comp):
    batch_size, channels, height, width = i_comp.shape

    horizontal_diff = i_comp[:, :, :, 1:] - i_comp[:, :, :, :-1]
    vertical_diff = i_comp[:, :, 1:, :] - i_comp[:, :, :-1, :]

    tv_loss = torch.sum(torch.abs(horizontal_diff)) + torch.sum(torch.abs(vertical_diff))
    tv_loss /= (batch_size * channels * height * width)

    return tv_loss


class PerceptualLoss(nn.Module):
    """
    Input: i_out (B, 3, H, W) (H = W)
           i_comp: (B, 3, H, W)
           i_gt: (B, 3, H, W)
    """

    def __init__(self, device='cuda'):
        super(PerceptualLoss, self).__init__()
        self.device = device
        self.vgg16 = models.vgg16(pretrained=True).to(device)
        self.pool_1 = nn.Sequential(
            *list(self.vgg16.features.children())[:5])  # 1st pooling layers
        # output shape(B, 64, H/2, W/2)
        self.pool_2 = nn.Sequential(
            *list(self.vgg16.features.children())[5:10])  # 2nd pooling layers
        # output shape(B, 128, H/4, W/4)
        self.pool_3 = nn.Sequential(
            *list(self.vgg16.features.children())[10:17])  # 3rd pooling layers
        # output shape(B, 256, H/8, W/8)

    def forward(self, i_out, i_gt, i_comp):
        # festures extracted with pool_1
        features_out_1 = self.pool_1(i_out)
        features_gt_1 = self.pool_1(i_gt)
        features_comp_1 = self.pool_1(i_comp)

        # festures extracted with pool_2
        features_out_2 = self.pool_2(features_out_1)
        features_gt_2 = self.pool_2(features_gt_1)
        features_comp_2 = self.pool_2(features_gt_1)

        # festures extracted with pool_3
        features_out_3 = self.pool_3(features_out_2)
        features_gt_3 = self.pool_3(features_gt_2)
        features_comp_3 = self.pool_3(features_gt_2)

        # calculate the perceptual loss w.r.t. pool_1,2,3
        perceptual_loss_out = (torch.mean(torch.abs(features_out_1 - features_gt_1)) + torch.mean(
            torch.abs(features_out_2 - features_gt_2)) + torch.mean(torch.abs(features_out_3 - features_gt_3))) / 3.
        perceptual_loss_comp = (torch.mean(torch.abs(features_comp_1 - features_gt_1)) + torch.mean(
            torch.abs(features_comp_2 - features_gt_2)) + torch.mean(torch.abs(features_comp_3 - features_gt_3))) / 3.

        perceptual_loss = perceptual_loss_out + perceptual_loss_comp

        return perceptual_loss


class StyleLoss(nn.Module):
    """
    Input: i_out (B, 3, H, W) (H = W)
           i_comp: (B, 3, H, W)
           i_gt: (B, 3, H, W)
    """

    def __init__(self, device='cuda'):
        super(StyleLoss, self).__init__()
        self.device = device
        self.vgg16 = models.vgg16(pretrained=True).to(device)
        self.pool_1 = nn.Sequential(
            *list(self.vgg16.features.children())[:5])  # 1st pooling layers
        # output shape(B, 64, H/2, W/2)
        self.pool_2 = nn.Sequential(
            *list(self.vgg16.features.children())[5:10])  # 2nd pooling layers
        # output shape(B, 128, H/4, W/4)
        self.pool_3 = nn.Sequential(
            *list(self.vgg16.features.children())[10:17])  # 3rd pooling layers
        # output shape(B, 256, H/8, W/8)

    def forward(self, i_out, i_gt, i_comp):
        # features extracted with pool_1
        features_out_1 = self.pool_1(i_out)
        features_gt_1 = self.pool_1(i_gt)
        features_comp_1 = self.pool_1(i_comp)

        # features extracted with pool_2
        features_out_2 = self.pool_2(features_out_1)
        features_gt_2 = self.pool_2(features_gt_1)
        features_comp_2 = self.pool_2(features_gt_1)

        # features extracted with pool_3
        features_out_3 = self.pool_3(features_out_2)
        features_gt_3 = self.pool_3(features_gt_2)
        features_comp_3 = self.pool_3(features_gt_2)

        # calculate the style loss w.r.t. pool_1,2,3
        style_loss_out = (torch.abs(
            torch.mean(features_out_1 ** 2 - features_gt_1 ** 2) / features_out_1.shape[1] ** 2) + torch.abs(
            torch.mean(features_out_2 ** 2 - features_gt_2 ** 2) / features_out_2.shape[1] ** 2) + torch.abs(
            torch.mean(features_out_3 ** 2 - features_gt_3 ** 2) / features_out_3.shape[1] ** 2)) / 3.
        style_loss_comp = (torch.abs(
            torch.mean(features_comp_1 ** 2 - features_comp_1 ** 2) / features_comp_1.shape[1] ** 2) + torch.abs(
            torch.mean(features_comp_2 ** 2 - features_comp_2 ** 2) / features_comp_2.shape[1] ** 2) + torch.abs(
            torch.mean(features_comp_3 ** 2 - features_comp_3 ** 2) / features_comp_3.shape[1] ** 2)) / 3.

        style_loss = style_loss_out + style_loss_comp
        return style_loss


class LPLoss(nn.Module):
    """
    Laplacian pyramid
    Input: I_out: tensor (B, C, H, W)
           I_gt: tensor (B, C, H, W)
    """

    def __init__(self, max_level=3):
        super(LPLoss, self).__init__()
        self.max_level = max_level
        self.build_pyramid = build_laplacian_pyramid

    def forward(self, i_out, i_gt):
        i_out_lap_list = self.build_pyramid(input=i_out, max_level=self.max_level)
        i_gt_lap_list = self.build_pyramid(input=i_gt, max_level=self.max_level)

        l1_norm_list = []
        for i in range(len(i_out_lap_list)):
            scale_factor = 4 ** i  # Use 4 ** i as the scale factor
            l1_norm = scale_factor * torch.mean(torch.abs(i_out_lap_list[i] - i_gt_lap_list[i]))
            l1_norm_list.append(l1_norm)

        ave_l1_norm = sum(l1_norm_list) / len(l1_norm_list)

        return ave_l1_norm

