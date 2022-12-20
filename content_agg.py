from re import S
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
import numpy as np
# from torchsummary import summary
import time

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.multiheadattn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, multiheadattn_mask=None):
        enc_output, enc_multiheadattn = self.multiheadattn(
            enc_input, enc_input, enc_input, mask=multiheadattn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_multiheadattn

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multiheadattn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, input_q, input_k, input_v, multiheadattn_mask=None):
        enc_output, enc_multiheadattn = self.multiheadattn(
            input_q, input_k, input_v, mask=multiheadattn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_multiheadattn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v):
        super(TransformerBlock, self).__init__()
        self.slfattn_pointnet = SelfAttention(d_model, d_inner, n_head, d_k, d_v)
        self.slfattn_dgcnn = SelfAttention(d_model, d_inner, n_head, d_k, d_v)
        self.crossattn = CrossAttention(d_model, d_inner, n_head, d_k, d_v)
        
    def forward(self, pointnet_feat, dgcnn_feat):
        self_slfattn_pointnet_output, _ = self.slfattn_pointnet(pointnet_feat)
        self_slfattn_dgcnn_output, _ = self.slfattn_dgcnn(dgcnn_feat)
        output, _ = self.crossattn(self_slfattn_dgcnn_output, self_slfattn_pointnet_output, self_slfattn_pointnet_output)
        return self_slfattn_pointnet_output, output

class Content_Agg(nn.Module):
    def __init__(self, d_model = 512, d_inner = 512, n_head = 4, d_k = 128, d_v = 128, num_trans_blocks = 4):
        super(Content_Agg, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.num_trans_blocks = num_trans_blocks
    
        trans_list = []
        for _ in range(self.num_trans_blocks):
            trans_list.append(TransformerBlock(d_model, d_inner, n_head, d_k, d_v))
        self.transformer_blocks = nn.ModuleList(trans_list)
        
        self.fc1 = nn.Linear(512, 512)   
        self.fc2 = nn.Linear(512, 512)
        
    def forward(self, pointnet_feat, dgcnn_feat):
        
        pointnet_feat = pointnet_feat.unsqueeze(2).repeat(1, 1, dgcnn_feat.shape[2])
        pointnet_feat = pointnet_feat.transpose(1, 2)
        dgcnn_feat = dgcnn_feat.transpose(1, 2)
        for num in range(self.num_trans_blocks):
            self_attn_pointnet, cross_attn_dgcnn = self.transformer_blocks[num](pointnet_feat, dgcnn_feat)
            pointnet_feat = self_attn_pointnet
            dgcnn_feat = cross_attn_dgcnn
        output = self.fc2(F.relu(self.fc1(cross_attn_dgcnn)))
        return output
    
        
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pointnet_feat, dgcnn_feat = torch.randn((2, 512)).to(device), torch.randn((2, 512, 512)).to(device)
    model = Content_Agg().to(device)
    start_time = time.time()
    out = model(pointnet_feat, dgcnn_feat)
    print("--- %s seconds ---" % (time.time() - start_time))

    print(f"model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}M")
    print(out.shape)    