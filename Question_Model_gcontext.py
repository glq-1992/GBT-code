#-*-coding:utf-8-*- 
#-*-coding:utf-8-*- 
import torch
import torch.nn as nn
from attention_context import Attention
import torch.nn.functional as F






class QuestionModel(nn.Module) :

    def __init__(self, vocab_size, vocab_nums, hidden_size, p, use_attention=False,batch_size=None):
        super(QuestionModel,self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_nums,embedding_dim=vocab_size)
        self.dropout = nn.Dropout(p=p)
        self.qsm = nn.GRU(input_size=vocab_size,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.max_length = 50
        self.batch_size=batch_size
        if use_attention:
            # use attention
            self.attention = Attention(dim=hidden_size,batch_size=self.batch_size)

    def forward(self, video_features, video_length, questions, question_length):
        questions = questions.long()

        word_embed = self.dropout(self.embed(questions))

        # question RNN encoding
        # size outputs [word_len,bsz,num_directions*hidden_size]
        # size hidden  [num_layers*num_directions,bsz,hidden_size]
        self.qsm.flatten_parameters()
        sorted_seq_lengths,indices = torch.sort(question_length,descending=True)
        word_embed = word_embed[indices]
        word_embed = nn.utils.rnn.pack_padded_sequence(word_embed,sorted_seq_lengths, batch_first=True)
        outputs , hidden = self.qsm(word_embed)
        _, desorted_indices = torch.sort(indices,descending = False)
        outputs , _ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        outputs = outputs[desorted_indices]  # not use

        hidden = hidden.transpose(0,1)

        # Attention
        self.attention.set_mask(input_lengths=video_length, context=video_features)
        q, attn_scores, _ = self.attention(hidden, video_features)    ######将question的最后一个状态和encoder的所有输出做attention

        # q = torch.cat((hidden, q), dim=2)   ##### 已经在attention里面做过cat了

        # return q, attn_scores
        return q, attn_scores,hidden,outputs





class QuestionEmbedding(nn.Module) :

    def __init__(self, vocab_size, vocab_nums, hidden_size, p, use_attention=False,batch_size=None):
        super(QuestionEmbedding,self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_nums,embedding_dim=vocab_size)
        self.dropout = nn.Dropout(p=p)
        self.qsm = nn.GRU(input_size=vocab_size,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.max_length = 50
        self.batch_size=batch_size


    def forward(self, questions, question_length):
        questions = questions.long()

        word_embed = self.dropout(self.embed(questions))

        # question RNN encoding
        # size outputs [word_len,bsz,num_directions*hidden_size]
        # size hidden  [num_layers*num_directions,bsz,hidden_size]
        self.qsm.flatten_parameters()
        sorted_seq_lengths,indices = torch.sort(question_length,descending=True)
        word_embed = word_embed[indices]
        word_embed = nn.utils.rnn.pack_padded_sequence(word_embed,sorted_seq_lengths.cpu(), batch_first=True)
        outputs , hidden = self.qsm(word_embed)
        _, desorted_indices = torch.sort(indices,descending = False)
        outputs , _ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        outputs = outputs[desorted_indices]  # not use

        hidden = hidden.transpose(0,1)

        # Attention
        # self.attention.set_mask(input_lengths=video_length, context=video_features)
        # q, attn_scores, _ = self.attention(hidden, video_features)    ######将question的最后一个状态和encoder的所有输出做attention

        # q = torch.cat((hidden, q), dim=2)   ##### 已经在attention里面做过cat了

        # return q, attn_scores
        return hidden,outputs




# class Context_QuestionModel(nn.Module) :
#
#     def __init__(self, vocab_size, vocab_nums, hidden_size, p, use_attention=False,batch_size=None):
#         super(Context_QuestionModel,self).__init__()
#
#         self.batch_size=batch_size
#         self.fc_q=nn.Linear(hidden_size,hidden_size)
#         self.fc_v=nn.Linear(hidden_size,hidden_size)
#
#         self.fc_q_mix=nn.Linear(500,1)
#         self.fc_v_mix=nn.Linear(78,1)
#         self.fc_combine=nn.Linear(hidden_size*2,hidden_size)
#
#
#     def forward(self, video_features, video_length, questions, question_length,q_features):
#
#         ##### 以前的
#         # questions = questions.long()
#         #
#         # word_embed = self.dropout(self.embed(questions))
#         #
#         # # question RNN encoding
#         # # size outputs [word_len,bsz,num_directions*hidden_size]
#         # # size hidden  [num_layers*num_directions,bsz,hidden_size]
#         # self.qsm.flatten_parameters()
#         # sorted_seq_lengths,indices = torch.sort(question_length,descending=True)
#         # word_embed = word_embed[indices]
#         # word_embed = nn.utils.rnn.pack_padded_sequence(word_embed,sorted_seq_lengths, batch_first=True)
#         # outputs , hidden = self.qsm(word_embed)
#         # _, desorted_indices = torch.sort(indices,descending = False)
#         # outputs , _ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
#         # outputs = outputs[desorted_indices]  # not use
#         #
#         # hidden = hidden.transpose(0,1)
#         #
#         # # Attention
#         # self.attention.set_mask(input_lengths=video_length, context=video_features)
#         # q, attn_scores, _ = self.attention(hidden, video_features)    ######将question的最后一个状态和encoder的所有输出做attention
#         #
#         # # q = torch.cat((hidden, q), dim=2)   ##### 已经在attention里面做过cat了
#         #
#         # # return q, attn_scores
#
#
#
#         ####### global-context
#         q_features_fc=torch.tanh(self.fc_q(q_features))
#         video_features_fc=torch.tanh(self.fc_v(video_features))
#
#         # q_mean=torch.mean(q_features_fc,dim=1).unsqueeze(1)
#         # v_mean = torch.mean(video_features_fc, dim=1).unsqueeze(1)
#         # combined = torch.cat((q_mean, v_mean), dim=2)
#         # q_v = self.fc_combine(combined)
#
#         C2 = torch.bmm(q_features_fc, video_features_fc.transpose(2,1))
#
#         C2_max = torch.zeros(self.batch_size, 78, 500, dtype=torch.float32, requires_grad=False).cuda()   ####不减帧时：(8,48)
#         C2_max[:,:C2.size(1),:C2.size(2)]=C2
#
#         q_features_mix=torch.tanh(self.fc_q_mix(C2_max))
#         video_features_mix = torch.tanh(self.fc_v_mix(C2_max.transpose(2,1)))
#
#         ###### 截取与输入长度相同的向量
#         q_features_mix_final = q_features_mix[:, :C2.size(1), :]
#         video_features_mix_final = video_features_mix[:, :C2.size(2), :]
#
#         ##### 哈达玛积 两个相同维度的矩阵相乘
#         # q_features_mix=torch.mean(q_features_mix, dim=1).unsqueeze(1)
#         # video_features_mix=torch.mean(video_features_mix, dim=1).unsqueeze(1)
#         # q_features_mix_final = q_features_mix.repeat(1, C2.size(1), 1)
#         # video_features_mix_final = video_features_mix.repeat(1, C2.size(2), 1)
#
#         q_attn = F.softmax(q_features_mix_final, dim=1)
#         v_attn = F.softmax(video_features_mix_final, dim=1)
#
#
#
#         q_mix= torch.bmm(q_features.transpose(2,1),q_attn).transpose(2,1)
#         v_mix = torch.bmm(video_features.transpose(2, 1), v_attn).transpose(2,1)
#         combined = torch.cat((q_mix, v_mix), dim=2)
#         q_v=self.fc_combine(combined)
#
#         # return q, attn_scores,hidden,outputs
#         return q_mix, v_mix, q_v


class Context_QuestionModel(nn.Module) :

    def __init__(self, vocab_size, vocab_nums, hidden_size, p, use_attention=False,batch_size=None):

        # Local Context Memory (LCM)
        super(Context_QuestionModel,self).__init__()

        self.batch_size=batch_size
        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_v = nn.Linear(hidden_size, hidden_size)

        self.fc_q_mix=nn.Linear(200,1)      #####读取灰度图像时为46，读取RGB图像时为150
        self.fc_v_mix=nn.Linear(10,1)
        self.fc_combine=nn.Linear(hidden_size*2,hidden_size)


    def forward(self, video_features, q_features):
        """

        :param video_features:
        :param video_length:
        :param questions:
        :param question_length:
        :param q_features:
        :return:
        """
        ####### global-context
        # q_features_fc = torch.tanh(self.fc_q(q_features))
        # # q_hat_features_fc = torch.tanh(self.fc_q(question_hat_features))
        # video_features_fc = torch.tanh(self.fc_v(video_features))
        # C2 = torch.bmm(q_features_fc, video_features_fc.transpose(2,1))
        # # C2_hat = torch.bmm(q_hat_features_fc, video_features_fc.transpose(2,1))
        #
        # # C2_max_hat = torch.zeros(self.batch_size, 10, 200, dtype=torch.float32).cuda()
        # # C2_max_hat[:, :C2_hat.size(1), :C2_hat.size(2)] = C2_hat
        #
        # C2_max = torch.zeros(self.batch_size, 10, 200, dtype=torch.float32).cuda()   ####不减帧时：(8,48)
        # C2_max[:,:C2.size(1),:C2.size(2)]=C2
        #
        #
        # q_features_mix = torch.tanh(self.fc_q_mix(C2_max))
        # video_features_mix = torch.tanh(self.fc_v_mix(C2_max.transpose(2,1)))
        #
        # # q_features_mix_hat = torch.tanh(self.fc_q_mix(C2_max_hat))
        # # video_features_mix_hat = torch.tanh(self.fc_v_mix(C2_max_hat.transpose(2,1)))
        #
        #
        #
        # ###### 截取与输入长度相同的向量
        # q_features_mix_final = q_features_mix[:, :C2.size(1), :]
        # video_features_mix_final = video_features_mix[:, :C2.size(2), :]
        # # q_features_mix_final_hat = q_features_mix_hat[:, :C2_hat.size(1), :]
        # # video_features_mix_final_hat = video_features_mix_hat[:, :C2_hat.size(2), :]
        #
        #
        # ##### 哈达玛积 两个相同维度的矩阵相乘
        # # q_features_mix=torch.mean(q_features_mix, dim=1).unsqueeze(1)
        # # video_features_mix=torch.mean(video_features_mix, dim=1).unsqueeze(1)
        # # q_features_mix_final = q_features_mix.repeat(1, C2.size(1), 1)
        # # video_features_mix_final = video_features_mix.repeat(1, C2.size(2), 1)
        #
        # q_attn = F.softmax(q_features_mix_final, dim=1)
        # v_attn = F.softmax(video_features_mix_final, dim=1)
        #
        # # q_attn_hat = F.softmax()
        #
        #
        # q_mix= torch.bmm(q_features.transpose(2,1),q_attn).transpose(2,1)
        # v_mix = torch.bmm(video_features.transpose(2, 1), v_attn).transpose(2,1)
        # combined = torch.cat((q_mix, v_mix), dim=2)
        # q_v = self.fc_combine(combined)

        ####### global-context

        # return q, attn_scores,hidden,outputs
        ####### global-context
        m = nn.ReLU(inplace=True)
        q_features_fc = m(self.fc_q(q_features))
        # q_hat_features_fc = torch.tanh(self.fc_q(question_hat_features))
        video_features_fc = m(self.fc_v(video_features))
        C2 = torch.bmm(q_features_fc, video_features_fc.transpose(2, 1))

        ##### now
        v_attn = F.softmax(C2.transpose(1,2), dim=2)
        v_mix = torch.bmm(v_attn, q_features_fc)
        # C2_v = torch.mean(C2, dim=1)  ##### video
        # C2 = C2.permute(0, 2, 1)
        # C2_q = torch.mean(C2, dim=1)  #### question
        #
        # v_attn = F.softmax(C2_v, dim=1)
        # q_attn = F.softmax(C2_q, dim=1)
        #
        # q_mix = torch.bmm(q_features.transpose(2, 1), q_attn.unsqueeze(2)).transpose(2, 1)
        # v_mix = torch.bmm(video_features.transpose(2, 1), v_attn.unsqueeze(2)).transpose(2, 1)
        combined = torch.cat((video_features, v_mix), dim=2)
        q_v = self.fc_combine(combined)


        return v_attn, v_mix, q_v