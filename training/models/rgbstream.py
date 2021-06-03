import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

# from models import resnet3D

__all__ = ["rgbstream"]


def rgbstream(**kwargs):
    num_classes = kwargs["num_classes"]  # 60
    resnet_shortcut = "B"  # kwargs['resnet_shortcut']
    sample_size = kwargs["inp_res"]  # kwargs['sample_size']
    sample_duration = kwargs["num_in_frames"]  # 16, 32...
    pretrain_path = kwargs["pretrain_path"]
    num_in_channels = kwargs["num_in_channels"]
    with_dropout = kwargs["with_dropout"]
    print("resnet_shortcut\t: %s" % resnet_shortcut)
    print("sample_size\t: %d" % sample_size)
    print("num_in_frames\t: %d" % sample_duration)
    print("num_in_channels\t: %d" % num_in_channels)
    print("with_dropout\t: %f" % with_dropout)

    model = ResNet3D(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        shortcut_type=resnet_shortcut,
        sample_size=sample_size,
        sample_duration=sample_duration,
        num_in_channels=num_in_channels,
        with_dropout=with_dropout,
    )

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    if pretrain_path:
        print("loading pretrained model {}".format(pretrain_path))
        pretrain = torch.load(pretrain_path)

        pretrained_dict = pretrain["state_dict"]
        model_dict = model.state_dict()

        # This part handles ignoring the last layer weights
        # 1. filter out unnecessary keys
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if (k in model_dict) and v.shape == model_dict[k].shape
        }
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    ).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = torch.cat([out.data, zero_pads], dim=1)

    return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class ResNet3D(nn.Module):
    def __init__(
        self,
        block,
        layers,
        sample_size,
        sample_duration,
        shortcut_type="B",
        num_classes=400,
        expansion=4,
        num_in_channels=3,
        with_dropout=0,
    ):
        self.inplanes = 64
        self.expansion = expansion
        super(ResNet3D, self).__init__()
        assert with_dropout >= 0 and with_dropout < 1
        self.with_dropout = with_dropout
        if self.with_dropout > 0:
            self.dp = nn.Dropout(p=self.with_dropout)
        self.conv1 = nn.Conv3d(
            num_in_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)

        self.fc = nn.Linear(512 * self.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        batchnorm_freeze = False  # This was manually set to true for ntu+surreact CS bnfreeze experiments
        if batchnorm_freeze:
            print("Batchnorm3d layers in eval mode")
            for name, m in self.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()  # mean/var from training
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * self.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * self.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * self.expansion),
                )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, expansion=self.expansion)
        )
        self.inplanes = planes * self.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=self.expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.with_dropout > 0:
            x = self.dp(x)
        return self.fc(x)
