import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pointnet, dgcnn, content_agg, foldingnet

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.enc1 = pointnet.PointNetEncoder()
        self.enc2 = dgcnn.DGCNN_encoder()
        self.content_agg = content_agg.Content_Agg()
        self.dec = foldingnet.Decoder()
        
    def forward(self, x):
        x1, _, _ = self.enc1(x)
        x2 = self.enc2(x)
        out = self.content_agg(x1, x2)
        coarse, fine = self.dec(out)
        return coarse, fine
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = Model().to("cuda")
    input = torch.randn((2, 3, 512)).to("cuda")
    coarse, fine = model(input)
    print(coarse.shape, fine.shape)