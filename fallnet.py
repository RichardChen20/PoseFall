import torch
import torch.nn as nn
import torch.nn.functional as F



class FallNetBase(nn.Module):
    """
    base model
    """
    # input: batchsize-frame-joints * xyz
    def __init__(self, num_joints_in, num_features_in, conv_channels, filter_widths=[3, 3, 3, 3, 3], dilation_sizes=[3, 9, 27, 81, 27],
                 dropout=0.25, num_frames=300, num_xyz=3, num_class=2):
        super().__init__()

        self.num_joints_in = num_joints_in
        self.num_features_in = num_features_in
        self.output_channels = conv_channels
        self.filter_widths = filter_widths
        self.dilation_sizes = dilation_sizes

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        #self.pad = [filter_widths[0] // 2]
        self.first_conv = nn.Conv1d(num_joints_in * num_xyz, conv_channels, filter_widths[0], bias=False)
        self.first_bn = nn.BatchNorm1d(conv_channels, momentum=0.1)
        self.average_pool = nn.AvgPool1d(58)
        self.fc = nn.Linear(conv_channels, num_class)

        layers_conv = []
        layers_bn = []

        for i in range(len(filter_widths) - 1):
            #self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            layers_conv.append(nn.Conv1d(conv_channels, conv_channels, filter_widths[i], dilation=dilation_sizes[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(conv_channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(conv_channels, conv_channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(conv_channels, momentum=0.1))

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)


    def forward(self, x):
        assert len(x.shape) == 4 # input: batchsize-frame-joints-xyz
        B, F, J, C = x.shape
        x = x.reshape((B, F, J*C))
        x = x.permute(0, 2, 1)

        # B, F, J*C to B, M (classification task)
        x = self._forward_blocks(x) # B, F, J*C to B, 4, 1024
        x = self.average_pool(x).reshape((B, self.output_channels))
        x = self.fc(x)

        return x


    def _forward_blocks(self, x):
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.relu(x)
        x = self.drop(x)

        for i in range(len(self.filter_widths) - 1):
            res = x[:, :, self.dilation_sizes[i]:x.shape[-1]-self.dilation_sizes[i]]
            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))
            x = x + res

        return x

    # useful functions
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames


class LiftNet(nn.Module):
    def __init__(self, num_joints=25, num_features=1024, dropout=0.5):
        super().__init__()

        self.joints = num_joints
        self.features = num_features
        self.dropout = dropout

        self.conv0 = nn.Conv1d(self.joints*2, self.features, 1)
        self.bn0 = nn.BatchNorm1d(self.features, momentum=0.1)
        self.relu0 = nn.ReLU(inplace=True)
        self.dropout0 = nn.Dropout(self.dropout)

        self.conv1 = nn.Conv1d(self.features, self.features, 1)
        self.bn1 = nn.BatchNorm1d(self.features, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(self.dropout)

        self.conv2 = nn.Conv1d(self.features, self.features, 1)
        self.bn2 = nn.BatchNorm1d(self.features, momentum=0.1)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(self.dropout)

        self.conv3 = nn.Conv1d(self.features, self.joints*3, 1)

    def forward(self, x):

        x = x.unsqueeze(2)
        x = self.dropout0(self.relu0(self.bn0(self.conv0(x))))
        res = x
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = x + res
        x = self.conv3(x)
        x = x.squeeze(2)

        return x


