import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F





# input =torch.randn(44,64,3)
# x1=TNET()(input)
# print(x1.shape)
# out=torch.bmm(torch.transpose(input,1,2),x1).transpose(1,2)
# out=F.relu(nn.BatchNorm1d(64)(nn.Conv1d(3,64,1)(out)))

# x2=TNET(n=64)(out)

# out=torch.bmm(torch.transpose(out,1,2),x2).transpose(1,2)
# out=F.relu(nn.BatchNorm1d(128)(nn.Conv1d(64,128,1)(out)))
# out=F.relu(nn.BatchNorm1d(1024)(nn.Conv1d(128,1024,1)(out)))
# out=torch.max(input=out,dim=2,keepdim=True)[0]
# out=out.view(-1,1024)
# print(out.shape)

#shared MLP

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
        return out,x1,x2

class Shared_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0=nn.Linear(1024,512)
        self.bn0=nn.BatchNorm1d(512)
        self.fc1=nn.Linear(512,256)
        self.bn1=nn.BatchNorm1d(256)
        self.fc2=nn.Linear(256,128)
        self.bn2=nn.BatchNorm1d(128)
    def forward(self,input):
        out=F.relu(self.bn0(self.fc0(input)))
        out=F.relu(self.bn1(self.fc1(out)))
        out=F.relu(self.bn2(self.fc2(out)))
        return out

#pointnet encoder
class PointNet_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform=Transform()
        self.shared_mlp=Shared_MLP()
    def forward(self,input):
        out,x1,x2=self.transform(input)
        out=self.shared_mlp(out)
        return out,x1,x2

#now check if it works
#not really sure

#must have nx3 size with n points
input =torch.randn(44,3,3)

encoder=PointNet_Encoder()
out,x1,x2=encoder(input)
print(out.shape)
print(x1.shape)
print(x2.shape)

#now we can use the encoder to encode the point cloud and then use the encoded vector to predict the class

# how to make 128 dim output to 1024 feature encoding




