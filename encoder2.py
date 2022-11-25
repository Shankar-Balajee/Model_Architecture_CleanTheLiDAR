import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# spatial transformer 

class Tnet(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.conv0 = nn.Conv1d(n, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.bn0 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc0 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, n * n)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.n = n

    def forward(self, input):
        out = F.relu(self.bn0(self.conv0(input)))
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = torch.max(input=out, dim=2, keepdim=True)[0]
        out = out.view(-1, 1024)
        out = F.relu(self.bn3(self.fc0(out)))
        out = F.relu(self.bn4(self.fc1(out)))
        out = self.fc2(out)
        weird_ass_thing = torch.from_numpy(np.eye(self.n).flatten().astype(np.float32)).view(-1, self.n * self.n).repeat(input.size()[0], 1)
        out = out + weird_ass_thing
        out = out.view(-1, self.n, self.n)
        return out

class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transformation = Tnet()
        self.feature_transformation = Tnet(n=64)
        self.conv0 = nn.Conv1d(3, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.bn0 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, input):
        x1 = self.input_transformation(input)
        out = torch.bmm(torch.transpose(input, 1, 2), x1).transpose(1, 2)
        out = F.relu(self.bn0(self.conv0(out)))
        x2 = self.feature_transformation(out)
        out = torch.bmm(torch.transpose(out, 1, 2), x2).transpose(1, 2)
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = torch.max(out, 2, keepdim=True)[0]
        out = out.view(-1, 1024)
        return out, x1, x2

# encoder

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = Transform()
        self.fc0 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.bn0 = nn.BatchNorm1d(512)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, input):
        out, x1, x2 = self.transform(input)
        out = F.relu(self.bn0(self.fc0(out)))
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        return out, x1, x2

input = torch.rand(32, 3, 1024)
encoder = Encoder()
out, x1, x2 = encoder(input)
print(out.shape, x1.shape, x2.shape)