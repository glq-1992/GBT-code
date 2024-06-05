import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules.tconv import TemporalConv, TemporalConv2SignDict
from modules.tconv import TemporalConv_TLP
from modules import BiLSTMLayer
from modules.criterions import SeqKD
import modules.resnet as resnet
import math
from .downsampled_multihead_attention import DownsampledMultiHeadAttention
from embeddings import Embeddings, SpatialEmbeddings
from torch import Tensor
# from encoder import Encoder, TransformerEncoder

from thop import profile
from torchvision.models import AlexNet


def Linear(in_features, out_features, dropout=0.):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return m
class SelfAttention(nn.Module):

    def __init__(self, out_channels, embed_dim, num_heads, project_input=False, gated=False, downsample=False):
        super().__init__()
        self.attention = DownsampledMultiHeadAttention(
            out_channels, embed_dim, num_heads, dropout=0, bias=True,
            project_input=project_input, gated=gated, downsample=downsample,
        )
        self.in_proj_q = Linear(out_channels, embed_dim)
        self.in_proj_k = Linear(out_channels, embed_dim)
        self.in_proj_v = Linear(out_channels, embed_dim)
        self.ln = Linear(out_channels, embed_dim)

    def forward(self, x):
        residual = x
        query = self.in_proj_q(x)
        key = self.in_proj_k(x)
        value = self.in_proj_v(x)
        x, _ = self.attention(query, key, value, mask_future_timesteps=True, use_scalar_bias=True)
        # aa=x + residual
        # bb=self.ln(x + residual)
        return self.ln(x + residual)
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        # self.fc = nn.Linear(1024,768)

    def forward(self, x):
        # x = self.fc(x)
        return x
class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs
class VACModel(nn.Module):
    def __init__(self, num_classes,c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size= 1024, gloss_dict=None):
        super(VACModel, self).__init__()
        self.num_classes = num_classes
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        ### lianyu
        # self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)
        self.gloss_dict=gloss_dict
        self.gloss_dict_word2index=dict((v, k) for k, v in gloss_dict.items())
        self.classifier_sign = nn.Linear(hidden_size, self.num_classes)
        # if weight_norm:
        #     self.classifier_sign = NormLinear(hidden_size, self.num_classes)
        #     self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        # else:
        #     self.classifier_sign = nn.Linear(hidden_size, self.num_classes)
        #     self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        # if share_classifier:
        #     self.conv1d.fc = self.classifier_sign
        self.register_backward_hook(self.backward_hook)
        self.position_embeddings = nn.Embedding(1024, 512)
        self.attention = SelfAttention(512, 512, 4)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[max(len_x) * idx:max(len_x) * idx + lgt] for idx, lgt in enumerate(len_x)])

        x = self.conv2d(x)

        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], max(len_x))
                       for idx, lgt in enumerate(len_x)])

        # x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # x = self.conv2d(x)
        # x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
        #                for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x_all):
            # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)
        # lianyu
        # framewise = self.conv2d(x.permute(0, 2, 1, 3, 4))
        # framewise_all = framewise.reshape(batch, temp, -1).transpose(1, 2)

        framewise = framewise_all
        len_x = len_x_all

        # position_ids = torch.arange(max(len_x), dtype=torch.long, device=framewise.device)
        # position_ids = position_ids.expand(framewise.size(0), framewise.size(1))
        # position_embeddings = self.position_embeddings(position_ids)
        # framewise = framewise + position_embeddings
        # framewise = self.attention(framewise)

        framewise = framewise.transpose(1, 2)
        # bs*dim*length

        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)

        outputs_sign = self.classifier_sign(tm_outputs['predictions'])


        return {
            "feat_len": lgt,
            "feat_frame": framewise,
            "conv_logits" : conv1d_outputs['conv_logits'],
            "sign_feat": tm_outputs['predictions'],
            "sequence_logits_sign" : outputs_sign,
        }
    def forward_framewise(self, x, len_x_all):
            # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)


        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)
        outputs_sign = self.classifier_sign(tm_outputs['predictions'])

        return {
            "framewise" : framewise,
            "feat_len": lgt,
            "conv_logits" : conv1d_outputs['conv_logits'],
            "sign_feat": tm_outputs['predictions'],
            "sequence_logits_sign" : outputs_sign,
        }
    
    def forward_framewiseOnly(self, x, len_x_all):
            # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)


        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length

        return {
            "framewise" : framewise,

        }

    def forward_noRNN(self, x, len_x_all):
            # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)


        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)
        outputs_sign = self.classifier_sign(tm_outputs['predictions'])

        return {
            "feat_len": lgt,
            "conv_logits" : conv1d_outputs['conv_logits'],
            "sign_feat": conv_output,
            "sequence_logits_sign" : outputs_sign,
        }


class VACModel_TLP(nn.Module):
    def __init__(self, num_classes, c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size=1024, gloss_dict=None):
        super(VACModel_TLP, self).__init__()
        self.num_classes = num_classes
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        ### lianyu
        # self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()
        # self.conv1d = TemporalConv(input_size=512,
        #                            hidden_size=hidden_size,
        #                            conv_type=conv_type,
        #                            use_bn=use_bn,
        #                            num_classes=num_classes)
        self.conv1d = TemporalConv_TLP(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        self.gloss_dict = gloss_dict
        self.gloss_dict_word2index = dict((v, k) for k, v in gloss_dict.items())
        self.classifier_sign = nn.Linear(hidden_size, self.num_classes)
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[max(len_x) * idx:max(len_x) * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], max(len_x))
                       for idx, lgt in enumerate(len_x)])

        # x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # x = self.conv2d(x)
        # x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
        #                for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x_all):
        # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)
        # lianyu
        # framewise = self.conv2d(x.permute(0, 2, 1, 3, 4))
        # framewise_all = framewise.reshape(batch, temp, -1).transpose(1, 2)

        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)

        outputs_sign = self.classifier_sign(tm_outputs['predictions'])

        return {
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sign_feat": tm_outputs['predictions'],
            "sequence_logits_sign": outputs_sign,
            "loss_LiftPool_u": conv1d_outputs['loss_LiftPool_u'],
            "loss_LiftPool_p": conv1d_outputs['loss_LiftPool_p'],
        }

    def forward_framewise(self, x, len_x_all):
        # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)

        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)
        outputs_sign = self.classifier_sign(tm_outputs['predictions'])

        return {
            "framewise": framewise,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sign_feat": tm_outputs['predictions'],
            "sequence_logits_sign": outputs_sign,
        }

    def forward_framewiseOnly(self, x, len_x_all):
        # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)

        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length

        return {
            "framewise": framewise,

        }

    def forward_noRNN(self, x, len_x_all):
        # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)

        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)
        outputs_sign = self.classifier_sign(tm_outputs['predictions'])

        return {
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sign_feat": conv_output,
            "sequence_logits_sign": outputs_sign,
        }
class VACModel2SignDict(nn.Module):
    def __init__(self, num_classes_fine,num_classes_coarse,c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size= 1024, gloss_dict_fine=None,gloss_dict_coarse=None):
        super(VACModel2SignDict, self).__init__()
        self.num_classes_fine = num_classes_fine
        self.num_classes_coarse = num_classes_coarse
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv2SignDict(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes_fine=num_classes_fine,
                                   num_classes_coarse=num_classes_coarse)
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)
        # self.gloss_dict_fine=gloss_dict_fine
        # self.gloss_dict_word2index=dict((v, k) for k, v in gloss_dict.items())
        self.classifier_sign_fine = nn.Linear(hidden_size, self.num_classes_fine)
        self.classifier_sign_coarse= nn.Linear(hidden_size, self.num_classes_coarse)
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[max(len_x) * idx:max(len_x) * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], max(len_x))
                       for idx, lgt in enumerate(len_x)])

        # x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # x = self.conv2d(x)
        # x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
        #                for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x_all):
            # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)


        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)
        outputs_sign_fine = self.classifier_sign_fine(tm_outputs['predictions'])
        outputs_sign_coarse = self.classifier_sign_coarse(tm_outputs['predictions'])

        return {
            "feat_len": lgt,
            "conv_logits_fine" : conv1d_outputs['conv_logits_fine'],
            "conv_logits_coarse": conv1d_outputs['conv_logits_coarse'],
            "sign_feat": tm_outputs['predictions'],
            "sequence_logits_sign_fine" : outputs_sign_fine,
            "sequence_logits_sign_coarse" : outputs_sign_coarse,
        }



class VACModelNoCNN(nn.Module):
    def __init__(self, num_classes,c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size= 1024, gloss_dict=None):
        super(VACModelNoCNN, self).__init__()
        self.num_classes = num_classes
        # self.conv2d = getattr(models, c2d_type)(pretrained=True)
        # self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)
        self.gloss_dict=gloss_dict
        self.gloss_dict_word2index=dict((v, k) for k, v in gloss_dict.items())
        self.classifier_sign = nn.Linear(hidden_size, self.num_classes)
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[max(len_x) * idx:max(len_x) * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], max(len_x))
                       for idx, lgt in enumerate(len_x)])

        # x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # x = self.conv2d(x)
        # x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
        #                for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, framewise, len_x):
            # videos
        # batch, temp, channel, height, width = x.shape
        # inputs = x.reshape(batch * temp, channel, height, width)
        # framewise_all = self.masked_bn(inputs, len_x_all)
        # framewise_all = framewise_all.reshape(batch, temp, -1)


        # framewise = framewise_all
        # len_x = len_x_all
        # framewise = framewise.transpose(1, 2)
        # bs*dim*length
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)
        outputs_sign = self.classifier_sign(tm_outputs['predictions'])

        return {
            "feat_len": lgt,
            "conv_logits" : conv1d_outputs['conv_logits'],
            "sign_feat": tm_outputs['predictions'],
            "sequence_logits_sign" : outputs_sign,
        }




class VACModel_SEN(nn.Module):
    def __init__(self, num_classes,c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size= 1024, gloss_dict=None):
        super(VACModel_SEN, self).__init__()
        self.num_classes = num_classes
        # self.conv2d = getattr(models, c2d_type)(pretrained=True)
        ### lianyu
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)
        self.gloss_dict=gloss_dict
        self.gloss_dict_word2index=dict((v, k) for k, v in gloss_dict.items())
        self.classifier_sign = nn.Linear(hidden_size, self.num_classes)
        # if weight_norm:
        #     self.classifier_sign = NormLinear(hidden_size, self.num_classes)
        #     self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        # else:
        #     self.classifier_sign = nn.Linear(hidden_size, self.num_classes)
        #     self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        # if share_classifier:
        #     self.conv1d.fc = self.classifier_sign
        self.register_backward_hook(self.backward_hook)
        self.position_embeddings = nn.Embedding(1024, 512)
        self.attention = SelfAttention(512, 512, 4)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[max(len_x) * idx:max(len_x) * idx + lgt] for idx, lgt in enumerate(len_x)])

        x = self.conv2d(x)

        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], max(len_x))
                       for idx, lgt in enumerate(len_x)])

        # x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # x = self.conv2d(x)
        # x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
        #                for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x_all):
            # videos
        batch, temp, channel, height, width = x.shape
        # inputs = x.reshape(batch * temp, channel, height, width)
        # framewise_all = self.masked_bn(inputs, len_x_all)
        # framewise_all = framewise_all.reshape(batch, temp, -1)
        # lianyu
        framewise = self.conv2d(x.permute(0, 2, 1, 3, 4))
        # input = torch.FloatTensor(2,3,280,224,224).cuda()
        # flops, params = profile(self.conv2d, inputs=(input,))
        # print('flops:', flops)
        # print('params:', params)
        framewise_all = framewise.reshape(batch, temp, -1).transpose(1, 2)

        framewise = framewise_all
        len_x = len_x_all

        # position_ids = torch.arange(max(len_x), dtype=torch.long, device=framewise.device)
        # position_ids = position_ids.expand(framewise.size(0), framewise.size(1))
        # position_embeddings = self.position_embeddings(position_ids)
        # framewise = framewise + position_embeddings
        # framewise = self.attention(framewise)

        # framewise = framewise.transpose(1, 2)
        # bs*dim*length

        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)

        outputs_sign = self.classifier_sign(tm_outputs['predictions'])


        return {
            "feat_len": lgt,
            "feat_frame": framewise,
            "conv_logits" : conv1d_outputs['conv_logits'],
            "sign_feat": tm_outputs['predictions'],
            "sequence_logits_sign" : outputs_sign,
        }