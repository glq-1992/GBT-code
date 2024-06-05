#-*-coding:utf-8-*- 
import torch
import torch.nn as nn
import torch.nn.functional as F



import numpy as np


def my_log_softmax(x):
    '''只能处理3维的'''
    size = x.size()
    res = F.log_softmax(x.squeeze())
    res = res.view(size[0], size[1], -1)
    return res


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """

    def __init__(self, dim,batch_size):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)
        self.mask = None
        # self.attn = nn.Linear(dim, dim)
        # self.concat = nn.Linear(dim * 2, dim)
        # self.out = nn.Linear(dim, dim)
        # self.attn_cat = nn.Linear(dim* 2, dim)
        self.batch_size=batch_size
        # self.v = nn.Parameter(torch.FloatTensor(self.batch_size,1, dim))
        # nn.init.uniform_(self.v, -0.1, 0.1)
        # self.v = nn.Parameter(torch.rand(dim))
        # self.v1 = nn.Parameter(torch.rand(batch_size,1))
        # self.v = nn.Parameter(torch.FloatTensor(dim))

    def set_mask(self, input_lengths, context):
        """
        sets self.mask which is applied before softmax
        ones for inactive context fields, zeros for active context fields
        :param context_len: b
        :param context: if batch_first: (b x t_k x n) else: (t_k x b x n)
        self.mask: (b x t_k)
        """
        # input_lengths_array = np.array(input_lengths, np.int64)
        # input_lengths_tensor = torch.from_numpy(input_lengths_array)
        # input_lengths_tensor = input_lengths_tensor.cuda(non_blocking=True)
        # input_lengths_tensor = input_lengths

        max_len = context.size(1)
        indices = torch.arange(0, max_len, dtype=torch.int64,
                               device=context.device)
        self.mask = indices >= (input_lengths.unsqueeze(1))





    def forward(self, output, context):
        # output decoder
        # context encoder hiddens
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            t_q = output.size(1)
            mask = self.mask.unsqueeze(1).expand(batch_size, t_q, input_size)
            attn.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        # aa=combined.view(-1, 2 * hidden_size)
        # bb=self.linear_out(combined.view(-1, 2 * hidden_size))
        # cc=torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size)))
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)


        ####增加的 general
        # batch_size = output.size(0)
        # hidden_size = output.size(2)
        # input_size = context.size(1)
        # print('output:',output.size())
        # # (b, ts, h) (b, is, h)
        # rnn_outputs = output
        # encoder_outputs = context
        # encoder_outputs = self.attn(encoder_outputs).transpose(1, 2)
        # print('***************',rnn_outputs.size())
        # print('%%%%%%%%%%%%',encoder_outputs.size())
        # attn_energies = rnn_outputs.bmm(encoder_outputs)
        # print('@@@@@@@@@@@@',attn_energies.size())
        # print('&&&&&&&&&&&&&&',attn_energies.view(-1, input_size).size())
        # print('################', attn_energies.view(-1, input_size).view(batch_size, -1, input_size).size())
        #
        #
        # attn_weights = F.softmax(attn_energies.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        # print('?????????',attn_weights.size())
        # mix = torch.bmm(attn_weights,context)
        #
        # output_context = torch.cat((output, mix), 2)
        # # [ts, b, h]
        # output_context = self.concat(output_context)
        # concat_output = F.tanh(output_context)
        #
        # # [ts, b, o]
        # output = self.out(concat_output)
        # print('__________',output.size())


        ####增加的 general

        ####增加的 concat
        # batch_size = output.size(0)
        # hidden_size = output.size(2)
        # word_size=output.size(1)
        # input_size = context.size(1)
        #
        #
        # # aa=output.expend(batch_size,input_size,input_size)
        # # hidden = output.unsqueeze(1).repeat(1, input_size, 1)
        #
        # if output.size(1)!=1:
        #
        #
        #     output_un = output.unsqueeze(2)
        #
        #
        #     output_un=output_un.repeat(1,1,input_size,1)
        #
        #     context_un=context.unsqueeze(1)
        #
        #
        #     context_un=context_un.repeat(1,word_size,1,1)
        #
        #     h_o = torch.cat((output_un, context_un), 3)
        #     energy = self.attn_cat(h_o).tanh()
        #
        #     energy=energy.view(batch_size,-1, hidden_size)
        #
        #     energy = energy.permute(0, 2, 1)
        #
        #     v = self.v.repeat(batch_size, 1).unsqueeze(1).repeat(1, energy.size(2),1)
        #
        #     attention = torch.bmm(v, energy)
        #     v1=self.v1.unsqueeze(2).repeat(1, 1,energy.size(2))
        #
        #     attention1=torch.bmm(v1,attention)
        #
        #     attn_weights = F.softmax(attention1.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        #
        #     # energy = self.v.bmm(energy.transpose(1, 2))
        #     # bb= F.softmax(energy.view(-1, input_size), dim=1)
        #     # attn_weights = F.softmax(energy.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        #     mix = torch.bmm(attn_weights, context)
        #
        #
        #     output_context = torch.cat((output, mix), 2)
        #
        #     # [ts, b, h]
        #     output_context = self.concat(output_context)
        #     concat_output = F.tanh(output_context)
        #
        #     # [ts, b, o]
        #     output = self.out(concat_output)
        # else:
        #     output_re = output.repeat(1, input_size, 1)
        #     h_o = torch.cat((output_re, context), 2)
        #
        #     energy = self.attn_cat(h_o).tanh()
        #     energy = energy.permute(0, 2, 1)
        #     v = self.v.repeat(batch_size, 1).unsqueeze(1)
        #     attention = torch.bmm(v, energy).squeeze(1)
        #     attn_weights=F.softmax(attention, dim=1).view(batch_size, -1, input_size)
        #     # energy = self.v.bmm(energy.transpose(1, 2))
        #     # bb= F.softmax(energy.view(-1, input_size), dim=1)
        #     # attn_weights = F.softmax(energy.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        #     mix = torch.bmm(attn_weights, context)
        #
        #     output_context = torch.cat((output, mix), 2)
        #     # [ts, b, h]
        #     output_context = self.concat(output_context)
        #     concat_output = F.tanh(output_context)
        #
        #     # [ts, b, o]
        #     output = self.out(concat_output)
        ####增加的 concat





        return output, attn, mix

        # return output, attn_weights, mix


class FrameAttention_a2v(nn.Module):
    def __init__(self):
        super(FrameAttention_a2v, self).__init__()
        self.mask = None

    def set_mask(self, video_length, max_len, dim):
        input_lengths_array = np.array([video_length] * dim, np.int64)
        input_lengths_tensor = torch.from_numpy(input_lengths_array)

        indices = torch.arange(0, max_len, dtype=torch.int64)
        self.mask = indices >= (input_lengths_tensor.unsqueeze(1))
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def forward(self, audio_encoder_output, video_encoder_output):

        attn = torch.matmul(audio_encoder_output, video_encoder_output.transpose(1, 0))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn, dim=1)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.matmul(attn, video_encoder_output)

        return mix, attn


class FrameAttention_v2a(nn.Module):
    def __init__(self):
        super(FrameAttention_v2a, self).__init__()
        self.mask = None

    def set_mask(self, video_length, max_len, dim):
        input_lengths_array = np.array([video_length] * dim, np.int64)
        input_lengths_tensor = torch.from_numpy(input_lengths_array)

        indices = torch.arange(0, max_len, dtype=torch.int64)
        self.mask = indices >= (input_lengths_tensor.unsqueeze(1))
        self.mask = self.mask.transpose()

        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def forward(self, audio_encoder_output, video_encoder_output):

        attn = torch.matmul(video_encoder_output, audio_encoder_output.transpose(1, 0))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn, dim=1)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.matmul(attn, audio_encoder_output)

        return mix, attn

# class FrameAttention(nn.Module):
#     def __init__(self):
#         super(FrameAttention, self).__init__()
#         self.mask = None
#
#     def set_mask(self, audio_input_lengths, video_input_lengths):
#         """
#         sets self.mask which is applied before softmax
#         ones for inactive context fields, zeros for active context fields
#         """
#         batch_size = len(audio_input_lengths)
#         audio_max_length = max(audio_input_lengths)
#         video_max_length = max(video_input_lengths)
#
#         self.mask = torch.zeros(batch_size, audio_max_length, video_max_length)
#         for i in range(batch_size):
#             indices = torch.arange(0, video_max_length, dtype=torch.int64)
#             input_lengths = [0] * audio_max_length
#             input_lengths[:audio_input_lengths[i]] = [video_input_lengths[i]] * audio_input_lengths[i]
#
#             input_lengths_array = np.array(input_lengths, np.int64)
#             input_lengths_tensor = torch.from_numpy(input_lengths_array)
#             self.mask[i] = indices >= (input_lengths_tensor.unsqueeze(1))
#
#         self.mask = self.mask.byte()
#
#         if torch.cuda.is_available():
#             self.mask = self.mask.cuda()
#
#     def forward(self, audio_encoder_outputs, video_encoder_outputs):
#         # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
#         attn = torch.bmm(audio_encoder_outputs, video_encoder_outputs.transpose(1, 2))
#         if self.mask is not None:
#             attn.data.masked_fill_(self.mask, -float('inf'))
#         # attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
#         attn = F.softmax(attn, dim=2)
#
#         # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
#         mix = torch.bmm(attn, video_encoder_outputs)
#
#         return mix
