import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Decoder(nn.Module):
    """
    Decoder Module of FoldingNet
    """

    def __init__(self, num_points = 512, grid_size=2):
        super(Decoder, self).__init__()

        self.num_points = num_points
        self.grid_size = grid_size
        self.num_dense = (self.grid_size**2) * self.num_points
        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

        self.final_conv = nn.Sequential(
            nn.Conv1d(512 + 512 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        
    def forward(self, x):
        """
        x: (B, C)
        """
        B = x.shape[0]
        
        coarse = self.mlp(x)                                                                 # (B, num_coarse, 3), coarse point cloud
        
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_points, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_repeat = x.repeat(1, self.grid_size**2, 1).transpose(1, 2)       # (B, 1024, num_fine)
        global_feat = torch.max(x, dim = 1)[0].unsqueeze(2).expand(-1, -1 ,self.num_dense)
        # print(feature_repeat.shape)
        # print(seed.shape)
        # print(point_feat.shape)
        # print(global_feat.shape)
        
        feat = torch.cat([feature_repeat.to("cuda"), seed.to("cuda"), point_feat.to("cuda"), global_feat.to("cuda")], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud
        
        return coarse, fine.transpose(1, 2)
    
if __name__ == '__main__':
    dec = Decoder().to("cuda")
    x = torch.randn((2, 512, 512)).to("cuda")
    out = dec(x)
    print(out.shape)
