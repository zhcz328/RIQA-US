import torch
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
from models.CFA import CrossAttentionBlock
from models.clipreg import clipforReg

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

N_PARAMS = {'affine': 6,
            'translation': 2,
            'rotation': 1,
            'scale': 2,
            'shear': 2,
            'rotation_scale': 3,
            'translation_scale': 4,
            'rotation_translation': 3,
            'rotation_translation_scale': 5}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class R_STNModule(nn.Module):
    def __init__(self, in_num, block_index, args):
        super(R_STNModule, self).__init__()

        self.feat_size = 56 // (4 * block_index)
        self.r_stn_mode = args["mode"]
        self.r_stn_n_params = N_PARAMS[self.r_stn_mode]

        self.conv = nn.Sequential(
            conv3x3(in_planes=in_num, out_planes=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv3x3(in_planes=64, out_planes=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=16 * self.feat_size * self.feat_size, out_features=1024),
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=self.r_stn_n_params),
        )
        self.fc[2].weight.data.fill_(0)
        self.fc[2].weight.data.zero_()

        if self.r_stn_mode == 'affine':
            self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif self.r_stn_mode in ['translation', 'shear']:
            self.fc[2].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        elif self.r_stn_mode == 'scale':
            self.fc[2].bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
        elif self.r_stn_mode == 'rotation':
            self.fc[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        elif self.r_stn_mode == 'rotation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
        elif self.r_stn_mode == 'translation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
        elif self.r_stn_mode == 'rotation_translation':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
        elif self.r_stn_mode == 'rotation_translation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))

    def forward(self, x):
        mode = self.r_stn_mode
        batch_size = x.size(0)
        conv_x = self.conv(x)
        theta = self.fc(conv_x.view(batch_size, -1))

        if mode == 'affine':
            theta1 = theta.view(batch_size, 2, 3)
        else:
            theta1 = Variable(torch.zeros([batch_size, 2, 3], dtype=torch.float32, device=x.get_device()),
                              requires_grad=True)
            theta1 = theta1 + 0
            theta1[:, 0, 0] = 1.0
            theta1[:, 1, 1] = 1.0
            if mode == 'translation':
                theta1[:, 0, 2] = theta[:, 0]
                theta1[:, 1, 2] = theta[:, 1]
            elif mode == 'rotation':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle)
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle)
            elif mode == 'scale':
                theta1[:, 0, 0] = theta[:, 0]
                theta1[:, 1, 1] = theta[:, 1]
            elif mode == 'shear':
                theta1[:, 0, 1] = theta[:, 0]
                theta1[:, 1, 0] = theta[:, 1]
            elif mode == 'rotation_scale':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle) * theta[:, 1]
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle) * theta[:, 2]
            elif mode == 'translation_scale':
                theta1[:, 0, 2] = theta[:, 0]
                theta1[:, 1, 2] = theta[:, 1]
                theta1[:, 0, 0] = theta[:, 2]
                theta1[:, 1, 1] = theta[:, 3]
            elif mode == 'rotation_translation':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle)
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle)
                theta1[:, 0, 2] = theta[:, 1]
                theta1[:, 1, 2] = theta[:, 2]
            elif mode == 'rotation_translation_scale':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle) * theta[:, 3]
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle) * theta[:, 4]
                theta1[:, 0, 2] = theta[:, 1]
                theta1[:, 1, 2] = theta[:, 2]

        grid = F.affine_grid(theta1, torch.Size(x.shape))
        img_transform = F.grid_sample(x, grid, padding_mode="reflection")

        return img_transform, theta1, grid


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, args, block, layers, text_dim=768, img_dim=448, embed_dim=512):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.r_stn1 = R_STNModule(64, 1, args)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.r_stn2 = R_STNModule(128, 2, args)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.r_stn3 = R_STNModule(256, 3, args)
        self.clipforReg = clipforReg()

        # Define the deconvolution layer to reconstruct the image
        self.res_conv0 = nn.Sequential(
            nn.ConvTranspose2d(448, 64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            # Upsample to H/2, W/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            # Upsample to H, W
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

            # nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False),
            # # Keep output size as (4, 1, 224, 224)
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True),

            # nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)  # upsamle to H, W
        )
        self.cfa = CrossAttentionBlock(text_dim, img_dim, embed_dim)  # CrossAttentionBlock
        # Define the deconvolution layer to reconstruct the image
        self.res_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            # Upsample to H/8, W/8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            # Upsample to H/4, W/4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            # Upsample to H/2, W/2
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)  # 上采样到 H, W
        )


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _fix_r_stn(self, x, theta):
        grid = F.affine_grid(theta, torch.Size(x.shape))
        img_transform = F.grid_sample(x, grid, padding_mode="reflection")

        return img_transform

    def embedding_concat(self, x, y, use_cuda) -> Tensor:
        device = torch.device('cuda' if use_cuda else 'cpu')
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).to(device)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
        return z


    def forward(self, x: Tensor, Rec=True) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # torch.Size([32, 64, 56, 56])

        x = self.layer1(x)
        x, theta1, grid1 = self.r_stn1(x)
        out1 = x
        tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32)
        fixthea1 = torch.from_numpy(np.linalg.inv(np.concatenate((theta1.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cuda()

        self.r_stn1_output = self._fix_r_stn(x.detach(), fixthea1)
        # after layer1 shape:  torch.Size([32, 64, 56, 56])

        x = self.layer2(x)
        x, theta2, grid2 = self.r_stn2(x)
        out2 = x
        tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32)
        fixthea2 = torch.from_numpy(np.linalg.inv(np.concatenate((theta2.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cuda()
        self.r_stn2_output = self._fix_r_stn(self._fix_r_stn(x.detach(), fixthea2), fixthea1)
        # after layer2 shape:  torch.Size([32, 128, 28, 28])

        x = self.layer3(x)
        out, theta3, grid3 = self.r_stn3(x)
        tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32)
        fixthea3 = torch.from_numpy(np.linalg.inv(np.concatenate((theta3.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cuda()
        self.r_stn3_output = self._fix_r_stn(self._fix_r_stn(self._fix_r_stn(out.detach(), fixthea3), fixthea2), fixthea1)
        # after layer3 shape:  torch.Size([32, 256, 14, 14])

        if Rec:
            # re-construction
            self.train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
            self.train_outputs['layer1'].append(self.r_stn1_output)
            self.train_outputs['layer2'].append(self.r_stn2_output)
            self.train_outputs['layer3'].append(self.r_stn3_output)

            # Embedding concat
            embedding_vectors = self.train_outputs['layer1'][0]
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = self.embedding_concat(embedding_vectors, self.train_outputs[layer_name][0], True)

            text_feature = self.clipforReg(1)
            text_feature = torch.stack([text_feature] * x.size(0), dim=0).squeeze(1)
            embedding_vectors = self.cfa(text_feature, embedding_vectors)
            x_reconstructed = self.res_conv0(embedding_vectors)

            return out1, out2, out, x_reconstructed, grid1, grid2, grid3
        else:
            return out1, out2, out, None, grid1, grid2, grid3

def r_stn_net(args, pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(args, BasicBlock, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model