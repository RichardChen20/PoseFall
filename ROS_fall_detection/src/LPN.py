import torch
import torch.nn as nn
from context_block import ContextBlock


class  LBwithGCBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(LBwithGCBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels=inplanes,out_channels=planes,kernel_size=1,stride=1,padding=0)
        self.conv1_bn = nn.BatchNorm2d(planes)
        self.conv1_bn_relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.conv2_bn = nn.BatchNorm2d(planes)
        self.conv2_bn_relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * self.expansion, kernel_size=1, stride=1, padding=0)
        self.conv3_bn = nn.BatchNorm2d(planes * self.expansion)
        self.gcb = ContextBlock(planes * self.expansion, ratio=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1_bn_relu(self.conv1_bn(self.conv1(x)))
        out = self.conv2_bn_relu(self.conv2_bn(self.conv2(out)))
        out = self.conv3_bn(self.conv3(out))
        out = self.gcb(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

def computeGCD(a,b):
    while a != b:
        if a > b:
            a = a - b
        else:
            b = b - a
    return b

def GroupDeconv(inplanes, planes, kernel_size, stride, padding, output_padding):
    groups = computeGCD(inplanes, planes)
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=inplanes, out_channels=2*planes, kernel_size=kernel_size,
                           stride=stride, padding=padding, output_padding=output_padding, groups=groups),
        nn.Conv2d(2*planes, planes, kernel_size=1, stride=1, padding=0)
    )

class LPN(nn.Module):
    def __init__(self, nJoints):
        super(LPN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(LBwithGCBlock, 64, 3)
        self.layer2 = self._make_layer(LBwithGCBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(LBwithGCBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(LBwithGCBlock, 512, 3, stride=1)

        self.deconv_layers = self._make_deconv_group_layer()
        self.final_layer = nn.Conv2d(in_channels=self.inplanes,out_channels=nJoints,kernel_size=1,stride=1,padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_deconv_group_layer(self):
        layers = []
        planes = 256
        for i in range(2):
            planes = planes//2
            # layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,out_channels=256,kernel_size=4,stride=2,padding=1,output_padding=0,groups=computeGCD(self.inplanes,256)))
            layers.append(GroupDeconv(inplanes=self.inplanes, planes=planes, kernel_size=4, stride=2, padding=1, output_padding=0))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x





if __name__ == '__main__':
    model = LPN(nJoints=17)
    model.cuda()
    print('Test')
    #print(model)

    data = torch.randn(1, 3, 256, 192).cuda()
    out = model(data)
    #print(out)
    print(out.shape)




# tensor([[22.6559, 17.9853, 18.0003, 18.2738, 39.6131, 16.9065, 52.8072, 18.5335,
#          33.8715, 17.0900,  6.6834, 21.2778, 49.1681, 15.8688, 34.0064, 26.2579,
#          61.9474, 15.4221, 43.0450, 26.5805, 31.5281, 20.4255, 56.1089, 27.7003,
#          36.4057, 22.9371,  8.6800, 28.5253, 50.4714, 19.4043, 13.0295, 28.4742,
#          58.0933, 20.4258]], device='cuda:0')
#
#
# 16.0 23.25
# 16.0 23.25
# 16.0 22.75
# 8.25 25.0
# 13.0 22.75
# 4.0 27.75
# 10.25 22.0
# 6.25 35.0
# 15.0 21.25
# 26.75 35.0
# 40.0 26.75
# 9.0 37.75
# 11.75 30.0
# 23.25 37.0
# 26.25 26.0
# 27.0 36.75
# 41.75 27.0
