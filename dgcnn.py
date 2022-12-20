import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# spatial transformer 

# class Tnet(nn.Module):
#     def __init__(self, n=3):
#         super(Tnet, self).__init__()
#         self.conv0 = nn.Conv1d(n, 64, 1)
#         self.conv1 = nn.Conv1d(64, 128, 1)
#         self.conv2 = nn.Conv1d(128, 1024, 1)
#         self.bn0 = nn.BatchNorm1d(64)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(1024)
#         self.fc0 = nn.Linear(1024, 512)
#         self.fc1 = nn.Linear(512, 256)
#         self.fc2 = nn.Linear(256, n * n)
#         self.bn3 = nn.BatchNorm1d(512)
#         self.bn4 = nn.BatchNorm1d(256)
#         self.n = n

#     def forward(self, input):
#         out = F.relu(self.bn0(self.conv0(input)))
#         out = F.relu(self.bn1(self.conv1(out)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = torch.max(input=out, dim=2, keepdim=True)[0]
#         out = out.view(-1, 1024)
#         out = F.relu(self.bn3(self.fc0(out)))
#         out = F.relu(self.bn4(self.fc1(out)))
#         out = self.fc2(out)
#         Identity = torch.from_numpy(np.eye(self.n).flatten().astype(np.float32)).view(-1, self.n * self.n).repeat(input.size()[0], 1).to("cuda")
#         out = out + Identity
#         out = out.view(-1, self.n, self.n)
#         return out

# class Transform(nn.Module):
#     def __init__(self):
#         super(Transform, self).__init__()
#         self.input_transformation = Tnet()
#         self.feature_transformation = Tnet(n=64)
#         self.conv0 = nn.Conv1d(3, 64, 1)
#         self.conv1 = nn.Conv1d(64, 128, 1)
#         self.conv2 = nn.Conv1d(128, 1024, 1)
#         self.bn0 = nn.BatchNorm1d(64)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(1024)

#     def forward(self, input):
#         x1 = self.input_transformation(input)
#         out = torch.bmm(torch.transpose(input, 1, 2), x1).transpose(1, 2)
#         out = F.relu(self.bn0(self.conv0(out)))
#         x2 = self.feature_transformation(out)
#         out = torch.bmm(torch.transpose(out, 1, 2), x2).transpose(1, 2)
#         out = F.relu(self.bn1(self.conv1(out)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = torch.max(out, 2, keepdim=True)[0]
#         out = out.view(-1, 1024)
#         out=out.type(torch.FloatTensor)
#         x1=x1.type(torch.FloatTensor)
#         x2=x2.type(torch.FloatTensor)
#         return out.to("cuda"), x1.to("cuda"), x2.to("cuda")

# class EdgeConv(nn.Module):
#   def __init__(self, k, in_features, out_features):
#     super(EdgeConv, self).__init__()
#     self.k = k
#     self.in_features = in_features
#     self.out_features = out_features
  
#   def knn(self, input):
#     inner = -2 * torch.matmul(input.transpose(2,1).contiguous(), input)
#     xx = torch.sum(input ** 2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2,1).contiguous()
#     idx = pairwise_distance.topk(self.k, dim=-1)[1]  # (batch_size, num_points, k)
#     idx_base = torch.arange(0, input.shape[0]).view(-1, 1, 1)*31
#     idx = idx + idx_base.to("cuda")
#     batch_size=input.shape[0]
#     num_points=input.shape[2]
#     idx = idx.view(-1)
#     _, num_dims, _ = input.size()
#     x=input
#     x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
#     feature = x.view(batch_size*num_points, -1)[idx, :]
#     feature = feature.view(batch_size, num_points, self.k, num_dims) 
#     x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)
    
#     feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
#     return feature
  
#   def forward(self, input):
#     input = self.knn(input)
#     self.edge_conv = nn.Sequential(
#         nn.Conv2d(self.in_features, self.out_features,1),
#         nn.LeakyReLU(0.1),
#         nn.BatchNorm2d(self.out_features), # not sure about this number
#     )
#     out = self.edge_conv(input)
#     out=out.max(dim=-1,keepdim=False)[0]
#     return out

# class DGCNN_encoder(nn.Module):
#   def __init__(self):
#     super(DGCNN_encoder, self).__init__()
#     self.edgeconv1 = EdgeConv(20, 6, 64)
#     self.edgeconv2 = EdgeConv(20, 128, 64)
#     self.edgeconv3 = EdgeConv(20, 128, 128)
#     self.edgeconv4 = EdgeConv(20, 256, 256)
#     self.transform = Transform()
  
#   def forward(self,input):
#     out,x1,x2 = self.transform(input)
#     new_out=torch.matmul(x1,input)
#     out1 = self.edgeconv1(new_out)
#     out2 = self.edgeconv2(out1)
#     out3 = self.edgeconv3(out2)
#     out4 = self.edgeconv4(out3)
#     encoded = torch.cat((out1, out2, out3, out4), 1)
#     return encoded

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN_encoder(nn.Module):
    def __init__(self, emb_dims = 512):
        super(DGCNN_encoder, self).__init__()
        # self.args = args
        self.k = 20
        self.emb_dims = emb_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        
        return x
      
if __name__ == '__main__':
    input =torch.rand(2, 3, 512).to("cuda")
    Final_Encoder=DGCNN_encoder().to("cuda")
    output_finale=Final_Encoder(input)
    print(output_finale.shape)