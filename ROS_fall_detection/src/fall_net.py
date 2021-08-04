import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, device, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel
        self.device = device
        self.indices_c, self.indices_r = np.meshgrid(
            np.linspace(0, 1, width),
            np.linspace(0, 1, height),
            indexing='xy'
        )

        self.indices_r = torch.tensor(np.reshape(self.indices_r, (-1, height * width)))
        self.indices_c = torch.tensor(np.reshape(self.indices_c, (-1, height * width)))

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        *_, h, w = feature.shape

        self.indices_r = self.indices_r.to(self.device)
        self.indices_c = self.indices_c.to(self.device)
        feature = feature.to(self.device)

        feature = 100 * feature.reshape(*_, h * w)
        feature = F.softmax(feature, dim=-1)
        result_r = torch.sum((h - 1) * feature * self.indices_r, dim=-1)
        result_c = torch.sum((w - 1) * feature * self.indices_c, dim=-1)

        result = torch.stack([result_r, result_c], dim=-1)
        return result

class Fall_Net(torch.nn.Module):
    def __init__(self, height, width, channel, device):
        super(Fall_Net, self).__init__()
        self.arg_softmax = SpatialSoftmax(64, 48, 17, device=torch.device("cuda"))
        self.linear1 = nn.Linear(34, 1000)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(1000, 1000)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(1000, 2)
        #self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.arg_softmax(x)
        pose_cor = x
        x = self.linear1(x.view(34))
        x = self.relu1(x)
        x = self.dropout1(x)
        residual_x = x
        out = self.linear2(x)
        out = self.relu2(out)
        out = self.dropout2(out)
        out += residual_x
        out = self.linear3(out)
        #out = self.softmax(out)
        return out, pose_cor