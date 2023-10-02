# https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
# https://github.com/cvqluu/TDNN

import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNN(nn.Module):
    def __init__(self, input_dim=23, output_dim=512, context_size=5,
                stride=1, dilation=1, batch_norm=False, dropout_p=0.2):
        
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: (batch, seq_len, input_features)
        output: (batch, new_seq_len, output_features)
        '''
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(x, (self.context_size, self.input_dim), 
                    stride=(1, self.input_dim), 
                    dilation=(self.dilation, 1))

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x.float())
        x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x
    
class XVector(nn.Module):
    def __init__(self, input_dim=39, num_classes=46):
        super(XVector, self).__init__()
        self.frame1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1, dropout_p=0)
        self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2, dropout_p=0)
        self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3, dropout_p=0)
        self.frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, dropout_p=0)
        self.frame5 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1, dropout_p=0)

        self.segment6 = nn.Linear(3000, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # frame level
        frame1_out = self.frame1(inputs)
        frame2_out = self.frame2(frame1_out)
        frame3_out = self.frame3(frame2_out)
        frame4_out = self.frame4(frame3_out)
        frame5_out = self.frame5(frame4_out)

        # stats pooling
        mean = torch.mean(frame5_out, 1)
        std = torch.std(frame5_out, 1)
        stat_pooling = torch.cat((mean, std), 1)

        # segment level
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)

        # pred_logits = self.softmax(self.output(x_vec))
        pred_logits = self.output(x_vec)

        # return
        return pred_logits, x_vec
    
# class XVector(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super().__init__()

#         self.frame1 = nn.Sequential(
#             nn.Conv1d(input_dim, 512, kernel_size=5, dilation=1),
#             nn.BatchNorm1d(512),
#             nn.ReLU()
#          )
        
#         self.frame2 = nn.Sequential(
#             nn.Conv1d(512, 512, kernel_size=3, dilation=1),
#             nn.BatchNorm1d(512),
#             nn.ReLU()
#             )
        
#         self.frame3 = nn.Sequential(
#             nn.Conv1d(512, 512, kernel_size=3, dilation=2),
#             nn.BatchNorm1d(512),
#             nn.ReLU()
#             )
        
#         self.frame4 = nn.Sequential(
#             nn.Conv1d(512, 512, kernel_size=1, dilation=1),
#             nn.BatchNorm1d(512),
#             nn.ReLU()
#             )
        
#         self.frame5 = nn.Sequential(
#             nn.Conv1d(512, 1500, kernel_size=1, dilation=1),
#             nn.BatchNorm1d(1500),
#             nn.ReLU()
#             )
        
#         self.segment6 = nn.Sequential(
#             nn.Linear(3000, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU()
#             )
        
#         self.segment7 = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU()
#             )

#         self.output = nn.Sequential(
#             nn.Linear(512, num_classes),
#             # nn.Softmax()
#             )
        
#     def forward(self, inputs):
#         # frame level
#         frame1_out = self.frame1(inputs.T)
#         frame2_out = self.frame2(frame1_out)
#         frame3_out = self.frame3(frame2_out)
#         frame4_out = self.frame4(frame3_out)
#         frame5_out = self.frame5(frame4_out)

#         # stats pooling
#         mean = torch.mean(frame5_out, 2) 
#         std = torch.std(frame5_out, 2)
#         stat_pooling = torch.cat((mean, std), 1)

#         # segment level
#         segment6_out = self.segment6(stat_pooling)
#         x_vec = self.segment7(segment6_out)
#         pred_logits = self.output(x_vec)

#         # return
#         return pred_logits, x_vec