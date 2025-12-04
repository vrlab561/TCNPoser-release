import torch
import torch.nn as nn
from torch.nn.functional import relu
import os
from functools import partial
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm



class TemporalSpatialBackbone(torch.nn.Module):
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        self.head_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size // 4), nn.LeakyReLU()),
                                                     nn.Sequential(nn.Linear(6, hidden_size // 4), nn.LeakyReLU()),
                                                     nn.Sequential(nn.Linear(3, hidden_size // 4), nn.LeakyReLU()),
                                                     nn.Sequential(nn.Linear(3, hidden_size // 4), nn.LeakyReLU())])
        self.lhand_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size // 4), nn.LeakyReLU()),
                                                      nn.Sequential(nn.Linear(6, hidden_size // 4), nn.LeakyReLU()),
                                                      nn.Sequential(nn.Linear(3, hidden_size // 4), nn.LeakyReLU()),
                                                      nn.Sequential(nn.Linear(3, hidden_size // 4), nn.LeakyReLU())])
        self.rhand_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size // 4), nn.LeakyReLU()),
                                                      nn.Sequential(nn.Linear(6, hidden_size // 4), nn.LeakyReLU()),
                                                      nn.Sequential(nn.Linear(3, hidden_size // 4), nn.LeakyReLU()),
                                                      nn.Sequential(nn.Linear(3, hidden_size // 4), nn.LeakyReLU())])
        self.lfoot_linear_embeddings = nn.ModuleList(
            [nn.Sequential(nn.Linear(6, (hidden_size - 16) // 2), nn.LeakyReLU()),
             nn.Sequential(nn.Linear(6, (hidden_size - 16) // 2), nn.LeakyReLU()),
             nn.Sequential(nn.Linear(3, 16), nn.LeakyReLU())])
        self.rfoot_linear_embeddings = nn.ModuleList(
            [nn.Sequential(nn.Linear(6, (hidden_size - 16) // 2), nn.LeakyReLU()),
             nn.Sequential(nn.Linear(6, (hidden_size - 16) // 2), nn.LeakyReLU()),
             nn.Sequential(nn.Linear(3, 16), nn.LeakyReLU())])
        self.pelvis_linear_embeddings = nn.ModuleList(
            [nn.Sequential(nn.Linear(6, (hidden_size - 16) // 2), nn.LeakyReLU()),
             nn.Sequential(nn.Linear(6, (hidden_size - 16) // 2), nn.LeakyReLU()),
             nn.Sequential(nn.Linear(3, 16), nn.LeakyReLU())])
        self.lhand_inhead_linear_embeddings = nn.ModuleList(
            [nn.Sequential(nn.Linear(6, hidden_size // 4), nn.LeakyReLU()),
             nn.Sequential(nn.Linear(6, hidden_size // 4), nn.LeakyReLU()),
             nn.Sequential(nn.Linear(3, hidden_size // 4), nn.LeakyReLU()),
             nn.Sequential(nn.Linear(3, hidden_size // 4), nn.LeakyReLU())])
        self.rhand_inhead_linear_embeddings = nn.ModuleList(
            [nn.Sequential(nn.Linear(6, hidden_size // 4), nn.LeakyReLU()),
             nn.Sequential(nn.Linear(6, hidden_size // 4), nn.LeakyReLU()),
             nn.Sequential(nn.Linear(3, hidden_size // 4), nn.LeakyReLU()),
             nn.Sequential(nn.Linear(3, hidden_size // 4), nn.LeakyReLU())])

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(hidden_size)

        self.num_block = block_num
        num_rnn_layer = 1

        self.time_encoder = nn.ModuleList(
            [nn.ModuleList([
                TCNBlock(input_size=hidden_size, output_size=hidden_size, kernel_size=3, dilation=2 ** layer,
                         dropout=dropout)
                for layer in range(number_layer)
            ]) for _ in range(8)]  
        )

        self.cbam_layers = nn.ModuleList(
            [CBAM(hidden_size) for _ in range(8)]
        )

        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=nhead, batch_first=True)
        self.spatial_encoder = nn.ModuleList(
            [nn.TransformerEncoder(encoder_layer, num_layers=number_layer) for _ in range(self.num_block)]
        )



    def forward(self, x_in, rnn_state=None):
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        head_feats = [x_in[..., 0:6], x_in[..., 36:42], x_in[..., 72:75], x_in[..., 81:84]]
        lhand_feats = [x_in[..., 6:12], x_in[..., 42:48], x_in[..., 75:78], x_in[..., 84:87]]
        rhand_feats = [x_in[..., 12:18], x_in[..., 48:54], x_in[..., 78:81], x_in[..., 87:90]]
        lfoot_feats = [x_in[..., 18:24], x_in[..., 54:60], x_in[..., 126:129]]
        rfoot_feats = [x_in[..., 24:30], x_in[..., 60:66], x_in[..., 129:132]]
        pelvis_feats = [x_in[..., 30:36], x_in[..., 66:72], x_in[..., 132:135]]
        lhand_inhead_feats = [x_in[..., 90:96], x_in[..., 102:108], x_in[..., 114:117], x_in[..., 120:123]]
        rhand_inhead_feats = [x_in[..., 96:102], x_in[..., 108:114], x_in[..., 117:120], x_in[..., 123:126]]

        head_emb = []
        for idx in range(len(head_feats)):
            head_emb.append(self.head_linear_embeddings[idx](head_feats[idx]))
        head_emb = self.norm(torch.cat(head_emb, dim=-1))

        lhand_emb = []
        for idx in range(len(lhand_feats)):
            lhand_emb.append(self.lhand_linear_embeddings[idx](lhand_feats[idx]))
        lhand_emb = self.norm(torch.cat(lhand_emb, dim=-1))

        rhand_emb = []
        for idx in range(len(rhand_feats)):
            rhand_emb.append(self.rhand_linear_embeddings[idx](rhand_feats[idx]))
        rhand_emb = self.norm(torch.cat(rhand_emb, dim=-1))

        lfoot_emb = []
        for idx in range(len(lfoot_feats)):
            lfoot_emb.append(self.lfoot_linear_embeddings[idx](lfoot_feats[idx]))
        lfoot_emb = self.norm(torch.cat(lfoot_emb, dim=-1))

        rfoot_emb = []
        for idx in range(len(rfoot_feats)):
            rfoot_emb.append(self.rfoot_linear_embeddings[idx](rfoot_feats[idx]))
        rfoot_emb = self.norm(torch.cat(rfoot_emb, dim=-1))

        pelvis_emb = []
        for idx in range(len(pelvis_feats)):
            pelvis_emb.append(self.pelvis_linear_embeddings[idx](pelvis_feats[idx]))
        pelvis_emb = self.norm(torch.cat(pelvis_emb, dim=-1))

        lhand_inhead_emb = []
        for idx in range(len(lhand_inhead_feats)):
            lhand_inhead_emb.append(self.lhand_inhead_linear_embeddings[idx](lhand_inhead_feats[idx]))
        lhand_inhead_emb = self.norm(torch.cat(lhand_inhead_emb, dim=-1))

        rhand_inhead_emb = []
        for idx in range(len(rhand_inhead_feats)):
            rhand_inhead_emb.append(self.rhand_inhead_linear_embeddings[idx](rhand_inhead_feats[idx]))
        rhand_inhead_emb = self.norm(torch.cat(rhand_inhead_emb, dim=-1))

        collect_feats = torch.stack([head_emb, lhand_emb, rhand_emb, lfoot_emb, rfoot_emb, pelvis_emb,
                                     lhand_inhead_emb, rhand_inhead_emb], dim=-2).reshape(batch_size, time_seq, 8, -1)

        for idx in range(self.num_block):
            collect_feats_temporal = []
            for idx_num in range(8):
                tcn_input = collect_feats[:, :, idx_num, :].permute(0, 2, 1)
                for layer in self.time_encoder[idx_num]:
                    tcn_input = layer(tcn_input)
                tcn_output = tcn_input.permute(0, 2, 1)
                collect_feats_temporal.append(tcn_output)

            collect_feats_temporal = torch.stack(collect_feats_temporal, dim=-2)

            cbam_outs = []
            for channel_idx in range(8):
                feats = collect_feats_temporal[:, :, channel_idx, :]
                feats = feats.permute(0, 2, 1).unsqueeze(-1)
                feats = self.cbam_layers[channel_idx](feats)
                feats = feats.squeeze(-1).permute(0, 2, 1)
                cbam_outs.append(feats)

            collect_feats_temporal = torch.stack(cbam_outs, dim=2)

            collect_feats = self.spatial_encoder[idx](
                collect_feats_temporal.reshape(batch_size * time_seq, 8, -1)
            ).reshape(batch_size, time_seq, 8, -1)

        return collect_feats


class HMD_imu_HME_Universe(torch.nn.Module):
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        self.backbone = TemporalSpatialBackbone(input_dim, number_layer, hidden_size, dropout, nhead, block_num)

        self.pose_est = nn.Sequential(
            nn.Linear(hidden_size * 8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 22 * 6)
        )

        self.shape_est = nn.Sequential(
            nn.Linear(hidden_size * 8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 16)
        )

    def forward(self, x_in, rnn_state=None):
        collect_feats = self.backbone(x_in, rnn_state)
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        collect_feats = collect_feats.reshape(batch_size, time_seq, -1)

        pred_pose = self.pose_est(collect_feats)
        pred_shapes = self.shape_est(collect_feats)

        return pred_pose, pred_shapes


class ChannelAttention(nn.Module):  
    def __init__(self, in_planes, scaling=16):  
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // scaling, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // scaling, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module): 
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module): 
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, scaling=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca  
        sa = self.spatial_attention(x)
        x = x * sa  
        return x

class TCNBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, dilation=1, dropout=0.05):
        super(TCNBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(input_size, output_size, kernel_size, padding=self.padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(output_size, output_size, kernel_size, padding=self.padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.residual = nn.Conv1d(input_size, output_size, kernel_size=1) if input_size != output_size else None

    def forward(self, x):
        res = self.residual(x) if self.residual else x  
        x = self.conv1(x)  
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)  
        x = self.relu2(x)
        x = self.dropout2(x)

        if x.size(-1) != res.size(-1):
            min_len = min(x.size(-1), res.size(-1))
            x = x[:, :, :min_len]
            res = res[:, :, :min_len]

        return x + res