import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d

weight_decay = 1e-4
eps_const_value = 1e-05
momentum_const_value = 0.1

class IdentityBlock2D(nn.Module):
    def __init__(self, input, filters):
        super(IdentityBlock2D, self).__init__()

        filters1, filters2, filters3 = filters

        # conv_1x1_reduce
        self.conv_1x1_reduce = Conv2d(in_channels=input,
                                      out_channels=filters1,
                                      kernel_size=(1, 1),
                                      stride=(1, 1), groups=1, bias=False)
        nn.init.orthogonal_(self.conv_1x1_reduce.weight)

        # conv_1x1_reduce_bn
        self.conv_1x1_reduce_bn = BatchNorm2d(num_features=filters1,
                                              eps=eps_const_value,
                                              momentum=momentum_const_value)

        # conv_3x3
        self.conv_3x3 = Conv2d(in_channels=filters1,
                               out_channels=filters2,
                               kernel_size=(3, 3),
                               stride=(1, 1), groups=1, bias=False)
        nn.init.orthogonal_(self.conv_3x3.weight)

        # conv_3x3_bn
        self.conv_3x3_bn = BatchNorm2d(num_features=filters2,
                                       eps=eps_const_value,
                                       momentum=momentum_const_value)

        # conv_1x1_increase
        self.conv_1x1_increase = Conv2d(in_channels=filters2,
                                        out_channels=filters3,
                                        kernel_size=(1, 1),
                                        stride=(1, 1), groups=1, bias=False)
        nn.init.orthogonal_(self.conv_1x1_increase.weight)

        # conv_1x1_increase_bn
        self.conv_1x1_increase_bn = BatchNorm2d(num_features=filters3,
                                                eps=eps_const_value,
                                                momentum=momentum_const_value)

    def forward(self, x):
        x1 = self.conv_1x1_reduce(x)
        x1 = self.conv_1x1_reduce_bn(x1)

        x1 = F.pad(F.relu(x1, inplace=True), (1, 1, 1, 1))
        x1 = self.conv_3x3(x1)
        x1 = self.conv_3x3_bn(x1)
        x1 = self.conv_1x1_increase(F.relu(x1, inplace=True))
        x1 = self.conv_1x1_increase_bn(x1)

        return F.relu(x1 + x, inplace=True)

class ConvBlock2D(nn.Module):
    def __init__(self, input, filters, stride=(2, 2)):
        super(ConvBlock2D, self).__init__()

        filters1, filters2, filters3 = filters

        # conv_1x1_reduce
        self.conv_1x1_reduce = Conv2d(in_channels=input,
                                      out_channels=filters1,
                                      kernel_size=(1, 1),
                                      stride=stride, groups=1, bias=False)
        nn.init.orthogonal_(self.conv_1x1_reduce.weight)

        # conv_1x1_reduce_bn
        self.conv_1x1_reduce_bn = BatchNorm2d(num_features=filters1,
                                              eps=eps_const_value,
                                              momentum=momentum_const_value)

        # conv_1x1_proj
        self.conv_1x1_proj = Conv2d(in_channels=input,
                                    out_channels=filters3,
                                    kernel_size=(1, 1),
                                    stride=stride, groups=1, bias=False)
        nn.init.orthogonal_(self.conv_1x1_proj.weight)

         # conv_1x1_proj_bn
        self.conv_1x1_proj_bn = BatchNorm2d(num_features=filters3,
                                            eps=eps_const_value,
                                            momentum=momentum_const_value)

        # conv_3x3
        self.conv_3x3 = Conv2d(in_channels=filters1,
                               out_channels=filters2,
                               kernel_size=(3, 3),
                               stride=(1, 1), groups=1, bias=False)
        nn.init.orthogonal_(self.conv_3x3.weight)

        # conv_3x3_bn
        self.conv_3x3_bn = BatchNorm2d(num_features=filters2,
                                       eps=eps_const_value,
                                       momentum=momentum_const_value)

        # conv_1x1_increase
        self.conv_1x1_increase = Conv2d(in_channels=filters2,
                                        out_channels=filters3,
                                        kernel_size=(1, 1),
                                        stride=(1, 1), groups=1, bias=False)
        nn.init.orthogonal_(self.conv_1x1_increase.weight)

        # conv_1x1_increase_bn
        self.conv_1x1_increase_bn = BatchNorm2d(num_features=filters3,
                                                eps=eps_const_value,
                                                momentum=momentum_const_value)

    def forward(self, x):
        x1 = self.conv_1x1_reduce(x)

        x2 = self.conv_1x1_proj(x)
        x2 = self.conv_1x1_proj_bn(x2)

        x1 = self.conv_1x1_reduce_bn(x1)
        x1 = F.pad(F.relu(x1, inplace=True), (1, 1, 1, 1))
        x1 = self.conv_3x3(x1)
        x1 = self.conv_3x3_bn(x1)
        x1 = self.conv_1x1_increase(F.relu(x1, inplace=True))
        x1 = self.conv_1x1_increase_bn(x1)

        return F.relu(x1 + x2, inplace=True)

class ThinResnet34(nn.Module):
    def __init__(self):
        super(ThinResnet34, self).__init__()

        # ===============================================
        #            Convolution Block 1
        # ===============================================
        self.conv1_1_3x3_s1 = Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7),
                                     stride=(1, 1), groups=1, bias=False)
        nn.init.orthogonal_(self.conv1_1_3x3_s1.weight)
        self.conv1_1_3x3_s1_bn = BatchNorm2d(
            num_features=64, eps=eps_const_value, momentum=momentum_const_value)

        # ===============================================
        #            Convolution Section 2
        # ===============================================
        self.conv_block_2_a = ConvBlock2D(
            input=64, filters=[48, 48, 96], stride=(1, 1))
        self.identity_block_2_b = IdentityBlock2D(
            input=96, filters=[48, 48, 96])

        # ===============================================
        #            Convolution Section 3
        # ===============================================
        self.conv_block_3_a = ConvBlock2D(
            input=96, filters=[96, 96, 128])
        self.identity_block_3_b = IdentityBlock2D(
            input=128, filters=[96, 96, 128])
        self.identity_block_3_c = IdentityBlock2D(
            input=128, filters=[96, 96, 128])

        # ===============================================
        #            Convolution Section 4
        # ===============================================
        self.conv_block_4_a = ConvBlock2D(
            input=128, filters=[128, 128, 256])
        self.identity_block_4_b = IdentityBlock2D(
            input=256, filters=[128, 128, 256])
        self.identity_block_4_c = IdentityBlock2D(
            input=256, filters=[128, 128, 256])

        # ===============================================
        #            Convolution Section 5
        # ===============================================
        self.conv_block_5_a = ConvBlock2D(
            input=256, filters=[256, 256, 512])
        self.identity_block_5_b = IdentityBlock2D(
            input=512, filters=[256, 256, 512])
        self.identity_block_5_c = IdentityBlock2D(
            input=512, filters=[256, 256, 512])

    def forward(self, x):
        x = self.conv1_1_3x3_s1(F.pad(x, (3, 3, 3, 3)))
        x = self.conv1_1_3x3_s1_bn(x)

        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)

        x = self.conv_block_2_a(x)
        x = self.identity_block_2_b(x)

        x = self.conv_block_3_a(x)
        x = self.identity_block_3_b(x)
        x = self.identity_block_3_c(x)

        x = self.conv_block_4_a(x)
        x = self.identity_block_4_b(x)
        x = self.identity_block_4_c(x)

        x = self.conv_block_5_a(x)
        x = self.identity_block_5_b(x)
        x = self.identity_block_5_c(x)

        x = F.max_pool2d(x, kernel_size=(3, 1), stride=(2, 2), padding=0, ceil_mode=False)
        return x
    
class GhostVLAD(nn.Module):
    def __init__(self, in_dim, vlad_cluster, ghost_cluster):
        super(GhostVLAD, self).__init__()

        self.vlad_cluster = vlad_cluster
        self.ghost_cluster = ghost_cluster
        self.conv1 = nn.Conv2d(in_dim,
                               vlad_cluster + ghost_cluster,
                               kernel_size=(1, 1))
        self.centers = nn.Parameter(torch.rand(vlad_cluster + ghost_cluster,
                                               in_dim))

    def forward(self, x):
        N, C = x.shape[:2]

        soft_assign = self.conv1(x).view(N, self.vlad_cluster + self.ghost_cluster, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x = x.view(N, C, -1)
        x = x.expand(self.vlad_cluster + self.ghost_cluster, -1, -1, -1).permute(1, 0, 2, 3)
        c = self.centers.expand(x.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        feat_res = x - c
        weighted_res = feat_res * soft_assign.unsqueeze(2)
        cluster_res = weighted_res.sum(dim=-1)

        cluster_res = cluster_res[:, :self.vlad_cluster, :]  # ghost
        cluster_res = F.normalize(cluster_res, p=2, dim=-1)
        vlad_feats = cluster_res.view(N, -1)
        vlad_feats = F.normalize(vlad_feats, p=2, dim=-1)

        return vlad_feats
    
class ThinResnet34GhostVLAD(nn.Module):
    def __init__(self, vlad_cluster, ghost_cluster, num_classes):
        super(ThinResnet34GhostVLAD, self).__init__()

        self.vlad_cluster = vlad_cluster
        self.ghost_cluster = ghost_cluster
        self.num_classes = num_classes

        # ===============================================
        #                 ThinResnet34
        # ===============================================
        self.thin_resnet_34 = ThinResnet34()

        # ===============================================
        #              Convolution Block
        # ===============================================

        self.conv_1x7 = Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 7),
                               stride=(1, 1), groups=1, bias=False)
        nn.init.orthogonal_(self.conv_1x7.weight)
        
        # ===============================================
        #            Feature Aggregation
        # ===============================================
        self.gvlad_center = Conv2d(in_channels=512, out_channels=self.vlad_cluster+self.ghost_cluster,
                                   kernel_size=(1, 7), stride=(1, 1), groups=1, bias=False)
        nn.init.orthogonal_(self.gvlad_center.weight)

    def forward(self, x):
        # x: (batch, 1, num_feature, n_frame)

        x = self.thin_resnet_34(x)
        x_fc = self.conv_1x7(x)
        x_center = self.gvlad_center(x)

        # GhostVLAD
        x_fc_center = torch.cat((x_fc, x_center), 1)
        ghost_vlad = GhostVLAD(in_dim=x_fc_center.shape[1], vlad_cluster=self.vlad_cluster, ghost_cluster=self.ghost_cluster)
        x = ghost_vlad(x_fc_center)

        # Fully Connected
        fc_layer = nn.Linear(in_features=x.shape[1], out_features=512)
        nn.init.orthogonal_(fc_layer.weight)
        embedding = fc_layer(x)

        relu = nn.ReLU()
        embedding = relu(embedding)

        # Output
        output_layer = nn.Linear(in_features=embedding.shape[1], out_features=self.num_classes)
        nn.init.orthogonal_(output_layer.weight)
        pred_logits = output_layer(embedding)

        softmax = nn.Softmax()
        pred_logits = softmax(pred_logits)

        # return
        return pred_logits, embedding