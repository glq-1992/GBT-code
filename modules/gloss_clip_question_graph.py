import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
# from layers import GraphAttentionLayer, SpGraphAttentionLayer

class GCQEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, nlayers):
        super(GCQEncoder, self).__init__()
        layer = GCQEncoderLayer(nfeat = nfeat, nhid = nhid, dropout = dropout, alpha = alpha, nheads = nheads)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(nlayers)])
    def forward(self, gloss, clip, question, gloss2clipadj, gloss2questionadj):
        for i, layer_module in enumerate(self.layer):
            gloss, clip, question = layer_module(gloss, clip, question, gloss2clipadj, gloss2questionadj)
        return gloss, clip, question
    # def forward(self, gloss, clip, question):
    #     for i, layer_module in enumerate(self.layer):
    #         gloss, clip, question = layer_module(gloss, clip, question)
    #     return gloss, clip, question

# class GCQEncoderLayer(nn.Module):
#     def __init__(self, nfeat, nhid, dropout, alpha, nheads):
#         super(GCQEncoderLayer, self).__init__()
#
#         self.Gloss2ClipFusionLayer1 = FusionLayer1(nfeat=nfeat, nhid = nhid, dropout= dropout)
#         self.Clip2GlossFusionLayer1 = FusionLayer(nfeat=nfeat, nhid = nhid, dropout= dropout)
#         self.Clip2GlossLayer1 = [GraphCrossAttentionLayer2(in_features = nfeat, out_features = nhid // nheads, dropout = dropout, alpha = alpha, concat=True) for _ in range(nheads)]
#         self.Gloss2ClipLayer1=[GraphCrossAttentionLayer1(in_features = nfeat, out_features = nhid // nheads, dropout = dropout, alpha = alpha, concat=True) for _ in range(nheads)]
#         self.Question2GlossLayer1=[GraphCrossAttentionLayer(in_features = nfeat, out_features = nhid // nheads, dropout = dropout, alpha = alpha, concat=True) for _ in range(nheads)]
#         self.Question2GlossFusionLayer1 = FusionLayer(nfeat=nfeat, nhid = nhid, dropout= dropout)
#         self.Gloss2QuestionLayer1 = [GraphCrossAttentionLayer3(in_features = nfeat, out_features = nhid // nheads, dropout = dropout, alpha = alpha, concat=True) for _ in range(nheads)]
#         self.L=nn.Linear(nhid // nheads,nfeat)
#         self.Gloss2QuestionFusionLayer1=FusionLayer(nfeat=nfeat, nhid = nhid, dropout= dropout)
#         self.Question_clip2GlossLayer=GraphCrossAttentionLayer_final(in_features = nfeat, out_features = nhid // nheads, dropout = dropout, alpha = alpha, concat=True)
#     def forward(self, gloss, clip, question, gloss2clipadj, gloss2questionadj):
#         # original
#         gloss_sameshape = torch.bmm(gloss2clipadj.transpose(1,2).cuda(),gloss) ###gloss给到clip
#         clip_1 = self.Gloss2ClipFusionLayer1(clip, gloss_sameshape)  #### 更新后的clip
#
#         clip_aggregation = torch.cat([att(gloss, clip, gloss2clipadj.cuda()) for att in self.Clip2GlossLayer1], dim= -1) ###clip给到gloss, 多头注意力不共享
#         gloss_1 = self.Clip2GlossFusionLayer1(gloss, clip_aggregation)  #### 更新后的gloss
#
#         # gloss2questionadj = torch.ones()
#         gloss_aggregation = torch.cat([att(question, gloss, gloss2questionadj.transpose(1,2).cuda()) for att in self.Gloss2QuestionLayer1], dim= -1)
#         question_1 = self.Gloss2QuestionFusionLayer1(question, gloss_aggregation)
#
#         question_aggregation = torch.cat([att(gloss, question, gloss2questionadj.cuda()) for att in self.Question2GlossLayer1], dim= -1)
#         gloss_2 = self.Question2GlossFusionLayer1(gloss_1, question_aggregation)
#         return gloss_2, clip_1, question_1

    # def forward(self, gloss, clip, question, gloss2clipadj, gloss2questionadj):
    #     # original
    #
    #     question_aggregation = torch.cat(
    #         [att(gloss, question, gloss2questionadj.cuda()) for att in self.Question2GlossLayer1], dim=-1)   ##利用gloss-question矩阵聚合question
    #     gloss_1 = self.Question2GlossFusionLayer1(gloss, question_aggregation)  ##利用聚合后的question更新gloss
    #
    #     gloss_sameshape = torch.bmm(gloss2clipadj.transpose(1, 2).cuda(), gloss_1)  ###gloss给到clip
    #     clip_1 = self.Gloss2ClipFusionLayer1(clip, gloss_sameshape)  #### 更新后的clip
    #
    #     clip_aggregation = torch.cat([att(gloss, clip, gloss2clipadj.cuda()) for att in self.Clip2GlossLayer1],
    #                                  dim=-1)  ###clip给到gloss, 多头注意力不共享
    #     gloss_2 = self.Clip2GlossFusionLayer1(gloss, clip_aggregation)  #### 更新后的gloss
    #
    #     # gloss2questionadj = torch.ones()
    #     gloss_aggregation = torch.cat(
    #         [att(question, gloss_2, gloss2questionadj.transpose(1, 2).cuda()) for att in self.Gloss2QuestionLayer1],
    #         dim=-1)    ###gloss给到question
    #     question_1 = self.Gloss2QuestionFusionLayer1(question, gloss_aggregation)   ###更新后的question
    #
    #     gloss_3=self.Question_clip2GlossLayer(gloss_1, gloss_2, gloss)
    #
    #     return gloss, clip_1, question_1
        #one-way (question-gloss-video)
        # question_aggregation = torch.cat(
        #     [att(gloss, question, gloss2questionadj.cuda()) for att in self.Question2GlossLayer1], dim=-1)
        # gloss_2 = self.Question2GlossFusionLayer1(gloss, question_aggregation)
        #
        # gloss_sameshape = torch.bmm(gloss2clipadj.transpose(1, 2).cuda(), gloss_2)  ###gloss给到clip
        # clip_1 = self.Gloss2ClipFusionLayer1(clip, gloss_sameshape)  #### 更新后的clip
        # return gloss_2, clip_1, question

        # graph update 1
        #question to gloss
        # question_aggregation = torch.cat(
        #     [att(gloss, question, gloss2questionadj.cuda()) for att in self.Question2GlossLayer1], dim=-1)
        # gloss_1 = self.Question2GlossFusionLayer1(gloss, question_aggregation)    ##### cat就行
        # # gloss to clip
        # gloss_sameshape = torch.bmm(gloss2clipadj.transpose(1, 2).cuda(), gloss_1)  ###gloss给到clip
        # clip_1 = self.Gloss2ClipFusionLayer1(clip, gloss_sameshape)  #### 更新后的clip
        # # clip to gloss
        # clip_aggregation = torch.cat([att(gloss, clip, gloss2clipadj.cuda()) for att in self.Clip2GlossLayer1],
        #                              dim=-1)  ###clip给到gloss, 多头注意力不共享     ,,, 这里要改
        # gloss_2 = self.Clip2GlossFusionLayer1(gloss, clip_aggregation)  #### 更新后的gloss   ，，，这里要改
        #
        # # gloss to question，，，，要改
        # gloss_aggregation = torch.cat(
        #     [att(question, gloss_2, gloss2questionadj.transpose(1, 2).cuda()) for att in self.Gloss2QuestionLayer1],
        #     dim=-1)
        # question_1 = self.Gloss2QuestionFusionLayer1(question, gloss_aggregation)
        #
        # gloss_3=gloss_1+gloss_2
        # return gloss_3, clip_1, question_1

        # graph update 2
        # question to gloss
        # question_aggregation = self.Question2GlossLayer1[0](gloss, question, gloss2questionadj.cuda())
        # gloss_1 = gloss + self.L(question_aggregation)  ##### or cat
        # # gloss to clip
        # gloss_sameshape = torch.bmm(gloss2clipadj.transpose(1, 2).cuda(), gloss_1)  ###gloss给到clip
        # clip_1 = self.Gloss2ClipFusionLayer1(clip, gloss_sameshape)  #### 更新后的clip
        # # clip to gloss
        # clip_sameshape = torch.bmm(gloss2clipadj.cuda(), clip_1)
        # gloss_2 = self.Gloss2ClipFusionLayer1(gloss_1, clip_sameshape)
        # # clip_aggregation = torch.cat([att(gloss, clip, gloss2clipadj.cuda()) for att in self.Clip2GlossLayer1],
        # #                              dim=-1)  ###clip给到gloss, 多头注意力不共享    gloss2clipadj有问题
        # # gloss_2 = self.Clip2GlossFusionLayer1(gloss, clip_aggregation)  #### 更新后的gloss   ，，，这里要改
        #
        # # gloss to question，，，，要改
        # gloss_aggregation=self.Gloss2QuestionLayer1[0](question, gloss_2, gloss2questionadj.transpose(1, 2).cuda())
        # # gloss_aggregation = torch.cat(
        # #     [att(question, gloss_2, gloss2questionadj.transpose(1, 2).cuda()) for att in self.Gloss2QuestionLayer1],
        # #     dim=-1)
        # question_1 = question + self.L(gloss_aggregation)
        # # question_1 = self.Gloss2QuestionFusionLayer1(question, gloss_aggregation)
        #
        # # gloss_3 = gloss_1 + gloss_2
        # return gloss_2, clip_1, question_1

class GCQEncoderLayer(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GCQEncoderLayer, self).__init__()

        self.Gloss2ClipFusionLayer1 = FusionLayer(nfeat=nfeat, nhid=nhid, dropout=dropout)
        self.Clip2GlossFusionLayer1 = FusionLayer(nfeat=nfeat, nhid=nhid, dropout=dropout)
        self.Clip2GlossLayer1 = [GraphCrossAttentionLayer(in_features=nfeat, out_features=nhid // nheads, dropout=dropout, alpha=alpha,
                                     concat=True) for _ in range(nheads)]

        self.Question2GlossFusionLayer1 = FusionLayer(nfeat=nfeat, nhid=nhid, dropout=dropout)
        self.Question2GlossLayer1 = [GraphCrossAttentionLayer(in_features=nfeat, out_features=nhid // nheads, dropout=dropout, alpha=alpha,
                                     concat=True) for _ in range(nheads)]
        self.Gloss2QuestionFusionLayer1 = FusionLayer(nfeat=nfeat, nhid=nhid, dropout=dropout)
        self.Gloss2QuestionLayer1 = [GraphCrossAttentionLayer(in_features=nfeat, out_features=nhid // nheads, dropout=dropout, alpha=alpha,
                                     concat=True) for _ in range(nheads)]
        self.L = nn.Linear(nhid // nheads, nfeat)
    def forward(self, gloss, clip, question, gloss2clipadj, gloss2questionadj):
        # original
        gloss_sameshape = torch.bmm(gloss2clipadj.transpose(1, 2).cuda(), gloss)  ###gloss给到clip
        clip_1 = self.Gloss2ClipFusionLayer1(clip, gloss_sameshape)  #### 更新后的clip

        clip_aggregation = torch.cat([att(gloss, clip, gloss2clipadj.cuda()) for att in self.Clip2GlossLayer1],
                                     dim=-1)  ###clip给到gloss, 多头注意力不共享
        gloss_1 = self.Clip2GlossFusionLayer1(gloss, clip_aggregation)  #### 更新后的gloss

        # gloss2questionadj = torch.ones()
        gloss_aggregation = torch.cat(
            [att(question, gloss, gloss2questionadj.transpose(1, 2).cuda()) for att in self.Gloss2QuestionLayer1],
            dim=-1)
        question_1 = self.Gloss2QuestionFusionLayer1(question, gloss_aggregation)

        question_aggregation = torch.cat(
            [att(gloss, question, gloss2questionadj.cuda()) for att in self.Question2GlossLayer1], dim=-1)
        gloss_2 = self.Question2GlossFusionLayer1(gloss_1, question_aggregation)
        return gloss_2, clip_1, question_1


# def forward(self, gloss, clip, question, gloss2clipadj, gloss2questionadj):
    #     # original
    #
    #     question_aggregation = torch.cat(
    #         [att(gloss, question, gloss2questionadj.cuda()) for att in self.Question2GlossLayer1], dim=-1)   ##利用gloss-question矩阵聚合question
    #     gloss_1 = self.Question2GlossFusionLayer1(gloss, question_aggregation)  ##利用聚合后的question更新gloss
    #
    #     gloss_sameshape = torch.bmm(gloss2clipadj.transpose(1, 2).cuda(), gloss_1)  ###gloss给到clip
    #     clip_1 = self.Gloss2ClipFusionLayer1(clip, gloss_sameshape)  #### 更新后的clip
    #
    #     clip_aggregation = torch.cat([att(gloss, clip, gloss2clipadj.cuda()) for att in self.Clip2GlossLayer1],
    #                                  dim=-1)  ###clip给到gloss, 多头注意力不共享
    #     gloss_2 = self.Clip2GlossFusionLayer1(gloss, clip_aggregation)  #### 更新后的gloss
    #
    #     # gloss2questionadj = torch.ones()
    #     gloss_aggregation = torch.cat(
    #         [att(question, gloss_2, gloss2questionadj.transpose(1, 2).cuda()) for att in self.Gloss2QuestionLayer1],
    #         dim=-1)    ###gloss给到question
    #     question_1 = self.Gloss2QuestionFusionLayer1(question, gloss_aggregation)   ###更新后的question
    #
    #     gloss_3=self.Question_clip2GlossLayer(gloss_1, gloss_2, gloss)
    #
    #     return gloss, clip_1, question_1
        #one-way (question-gloss-video)
        # question_aggregation = torch.cat(
        #     [att(gloss, question, gloss2questionadj.cuda()) for att in self.Question2GlossLayer1], dim=-1)
        # gloss_2 = self.Question2GlossFusionLayer1(gloss, question_aggregation)
        #
        # gloss_sameshape = torch.bmm(gloss2clipadj.transpose(1, 2).cuda(), gloss_2)  ###gloss给到clip
        # clip_1 = self.Gloss2ClipFusionLayer1(clip, gloss_sameshape)  #### 更新后的clip
        # return gloss_2, clip_1, question

        # graph update 1
        #question to gloss
        # question_aggregation = torch.cat(
        #     [att(gloss, question, gloss2questionadj.cuda()) for att in self.Question2GlossLayer1], dim=-1)
        # gloss_1 = self.Question2GlossFusionLayer1(gloss, question_aggregation)    ##### cat就行
        # # gloss to clip
        # gloss_sameshape = torch.bmm(gloss2clipadj.transpose(1, 2).cuda(), gloss_1)  ###gloss给到clip
        # clip_1 = self.Gloss2ClipFusionLayer1(clip, gloss_sameshape)  #### 更新后的clip
        # # clip to gloss
        # clip_aggregation = torch.cat([att(gloss, clip, gloss2clipadj.cuda()) for att in self.Clip2GlossLayer1],
        #                              dim=-1)  ###clip给到gloss, 多头注意力不共享     ,,, 这里要改
        # gloss_2 = self.Clip2GlossFusionLayer1(gloss, clip_aggregation)  #### 更新后的gloss   ，，，这里要改
        #
        # # gloss to question，，，，要改
        # gloss_aggregation = torch.cat(
        #     [att(question, gloss_2, gloss2questionadj.transpose(1, 2).cuda()) for att in self.Gloss2QuestionLayer1],
        #     dim=-1)
        # question_1 = self.Gloss2QuestionFusionLayer1(question, gloss_aggregation)
        #
        # gloss_3=gloss_1+gloss_2
        # return gloss_3, clip_1, question_1

        # graph update 2
        # question to gloss
        # question_aggregation = self.Question2GlossLayer1[0](gloss, question, gloss2questionadj.cuda())
        # gloss_1 = gloss + self.L(question_aggregation)  ##### or cat
        # # gloss to clip
        # gloss_sameshape = torch.bmm(gloss2clipadj.transpose(1, 2).cuda(), gloss_1)  ###gloss给到clip
        # clip_1 = self.Gloss2ClipFusionLayer1(clip, gloss_sameshape)  #### 更新后的clip
        # # clip to gloss
        # clip_sameshape = torch.bmm(gloss2clipadj.cuda(), clip_1)
        # gloss_2 = self.Gloss2ClipFusionLayer1(gloss_1, clip_sameshape)
        # # clip_aggregation = torch.cat([att(gloss, clip, gloss2clipadj.cuda()) for att in self.Clip2GlossLayer1],
        # #                              dim=-1)  ###clip给到gloss, 多头注意力不共享    gloss2clipadj有问题
        # # gloss_2 = self.Clip2GlossFusionLayer1(gloss, clip_aggregation)  #### 更新后的gloss   ，，，这里要改
        #
        # # gloss to question，，，，要改
        # gloss_aggregation=self.Gloss2QuestionLayer1[0](question, gloss_2, gloss2questionadj.transpose(1, 2).cuda())
        # # gloss_aggregation = torch.cat(
        # #     [att(question, gloss_2, gloss2questionadj.transpose(1, 2).cuda()) for att in self.Gloss2QuestionLayer1],
        # #     dim=-1)
        # question_1 = question + self.L(gloss_aggregation)
        # # question_1 = self.Gloss2QuestionFusionLayer1(question, gloss_aggregation)
        #
        # # gloss_3 = gloss_1 + gloss_2
        # return gloss_2, clip_1, question_1
class FusionLayer(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        """Refer to Dense version of GAT."""
        super(FusionLayer, self).__init__()

        self.W = nn.Linear(nfeat, nhid, bias=False)
        self.U = nn.Linear(nfeat, nhid, bias=False)
        self.Wf = nn.Linear(nfeat, nhid, bias=False)
        self.Uf = nn.Linear(nfeat, nhid, bias=False)
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()

    def forward(self, a, b):
        # x = F.dropout(x, self.dropout, training=self.training)
        # gloss_sameshape = torch.bmm(adj.transpose(1,2).cuda(),gloss)
        new_a = self.W(a) + self.U(b)
        f = self.sigmoid(self.Wf(a) + self.Uf(b))
        output = f * new_a + (1 - f) * a
        output = F.dropout(output, self.dropout, training=self.training)
        return output

class FusionLayer1(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        """Refer to Dense version of GAT."""
        super(FusionLayer1, self).__init__()

        self.W = nn.Linear(nfeat, nhid, bias=False)
        self.U = nn.Linear(nfeat, nhid, bias=False)
        self.Wf = nn.Linear(nfeat, nhid, bias=False)
        self.Uf = nn.Linear(nfeat, nhid, bias=False)
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
    def forward(self, a, b):
        # x = F.dropout(x, self.dropout, training=self.training)
        # gloss_sameshape = torch.bmm(adj.transpose(1,2).cuda(),gloss)
        new_a = self.W(a) + self.U(b)
        f = self.sigmoid(self.Wf(a) + self.Uf(b))
        output = f * new_a + (1 - f) * b
        output = F.dropout(output, self.dropout, training=self.training)
        return output
    # def forward(self, a, b):
    #     # x = F.dropout(x, self.dropout, training=self.training)
    #     # gloss_sameshape = torch.bmm(adj.transpose(1,2).cuda(),gloss)
    #     new_a = self.W(a) + self.U(b)
    #     f = self.sigmoid(self.Wf(a) + self.Uf(b))
    #     # output = f * a + (1 - f) * b
    #     output = a + f * b
    #     output = F.dropout(output, self.dropout, training=self.training)
    #     return output
class GraphCrossAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphCrossAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)).cuda())
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)).cuda())
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, a, b, adj):

        Wa = torch.matmul(a, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wb = torch.matmul(b, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wa, Wb)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wb)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wa, Wb):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wha = torch.matmul(Wa, self.a[:self.out_features, :])
        Whb = torch.matmul(Wb, self.a[self.out_features:, :])
        # broadcast add
        e = Wha + Whb.transpose(1,2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphCrossAttentionLayer3(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphCrossAttentionLayer3, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)).cuda())
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)).cuda())
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, a, b, adj):

        Wa = torch.matmul(a, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wb = torch.matmul(b, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wa, Wb)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wb)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wa, Wb):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wha = torch.matmul(Wa, self.a[:self.out_features, :])
        Whb = torch.matmul(Wb, self.a[self.out_features:, :])
        # broadcast add
        e = Wha + Whb.transpose(1,2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphCrossAttentionLayer2(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphCrossAttentionLayer2, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)).cuda())
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)).cuda())
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, a, b, adj):

        Wa = torch.matmul(a, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wb = torch.matmul(b, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wa, Wb)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wb)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wa, Wb):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wha = torch.matmul(Wa, self.a[:self.out_features, :])
        Whb = torch.matmul(Wb, self.a[self.out_features:, :])
        # broadcast add
        e = Wha + Whb.transpose(1,2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class GraphCrossAttentionLayer1(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphCrossAttentionLayer1, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)).cuda())
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)).cuda())
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, a, b, adj):

        Wa = torch.matmul(a, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wb = torch.matmul(b, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wa, Wb)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wb)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wa, Wb):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wha = torch.matmul(Wa, self.a[:self.out_features, :])
        Whb = torch.matmul(Wb, self.a[self.out_features:, :])
        # broadcast add
        e = Wha + Whb.transpose(1,2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphCrossAttentionLayer_final(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphCrossAttentionLayer_final, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = in_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, in_features)).cuda())
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)).cuda())
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, a, b, c):

        Wa = torch.matmul(a, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wc = torch.matmul(c, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # e = self._prepare_attentional_mechanism_input(Wa, Wc)
        e = torch.bmm(Wa, Wc.transpose(2,1))


        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wa)

        Wb = torch.matmul(b, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wc = torch.matmul(c, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # e1 = self._prepare_attentional_mechanism_input(Wb, Wc)
        e1 = torch.bmm(Wb, Wc.transpose(2, 1))

        attention1 = F.softmax(e1, dim=1)
        attention1 = F.dropout(attention1, self.dropout, training=self.training)
        h_prime1 = torch.matmul(attention1, Wb)

        h_prime2=h_prime+h_prime1
        if self.concat:
            return F.elu(h_prime2)
        else:
            return h_prime2

    def _prepare_attentional_mechanism_input(self, Wa, Wb):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wha = torch.matmul(Wa, self.a[:self.out_features, :])
        Whb = torch.matmul(Wb, self.a[self.out_features:, :])
        # broadcast add
        e = Wha + Whb.transpose(1,2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
