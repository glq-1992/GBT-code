import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import choice
import random
class ContrastiveLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        # self.kdloss = nn.KLDivLoss(reduction='batchmean')
        # self.T = T
        self.temperature=0.5
    def forward(self, feature_ori, feature_pos, feature_neg, feature_neg2):
        feature_ori=feature_ori.mean(dim=1)
        feature_pos=feature_pos.mean(dim=1)
        feature_neg=feature_neg.mean(dim=1)
        feature_neg2=feature_neg2.mean(dim=1)
        # pos = torch.cosine_similarity(feature_ori, feature_pos, dim=1)/self.temperature  # [length,batch]
        # neg = torch.cosine_similarity(feature_ori, feature_neg, dim=1)/self.temperature  # [length,batch]
        # neg2 = torch.cosine_similarity(feature_ori, feature_neg2, dim=1)/self.temperature  # [length,batch]
        pos = torch.cosine_similarity(feature_ori, feature_pos, dim=1)   # [length,batch]
        neg = torch.cosine_similarity(feature_ori, feature_neg, dim=1)   # [length,batch]
        neg2 = torch.cosine_similarity(feature_ori, feature_neg2, dim=1)   # [length,batch]

        ### gai
        # logit = torch.stack((pos, neg, neg2), 1)
        # pone=torch.sum(logit,dim=1)
        # postive=torch.exp(pos)
        # num=torch.exp(pone)
        # contras_loss1 = - torch.log(postive/num)

        ### original
        logit = torch.stack((pos, neg , neg2), 1) # [length,batch,3]
        softmax_logit = nn.functional.softmax(logit, 1) # [length,batch,3] 沿着最后一个维度做softmax
# softmax_logit[:,:,0] 表示exp(pos)/exp(pos)+exp(neg)+exp(neg2) 
        contras_loss = - torch.log(softmax_logit[:,0])
                    # contras_loss += torch.log(softmax_logit[:, 1]) # add contras_neg
        contras_loss = contras_loss.mean()
        return contras_loss
# class ContrastiveLoss_local(nn.Module):
#     """
#     NLL loss with label smoothing.
#     """
#
#     def __init__(self):
#         super(ContrastiveLoss_local, self).__init__()
#         # self.kdloss = nn.KLDivLoss(reduction='batchmean')
#         # self.T = T
#         self.temperature = 0.5
#     def forward(self, feature_v, feature_q):
#         contras_loss=0
#         for i in range(feature_v.size(1)):
#             list = []
#             aa = feature_v.size(1)
#             for j in range(feature_v.size(1)):
#                 list = list + [j]
#             feature_ori=feature_v[:,i,:]
#             feature_pos = feature_q[:,i,:]
#             list_neg=list.pop(i)
#             choice=random.sample(list,2)
#             feature_neg=feature_q[:,choice[0],:]
#             feature_neg1 = feature_q[:, choice[1], :]
#             # pos = torch.cosine_similarity(feature_ori, feature_pos, dim=1)/self.temperature  # [length,batch]
#             # neg = torch.cosine_similarity(feature_ori, feature_neg, dim=1)/self.temperature  # [length,batch]
#             # neg2 = torch.cosine_similarity(feature_ori, feature_neg1, dim=1)/self.temperature  # [length,batch]
#             pos = torch.cosine_similarity(feature_ori, feature_pos, dim=1)   # [length,batch]
#             neg = torch.cosine_similarity(feature_ori, feature_neg, dim=1)   # [length,batch]
#             neg2 = torch.cosine_similarity(feature_ori, feature_neg1, dim=1)   # [length,batch]
#             logit = torch.stack((pos, neg, neg2), 1)  # [length,batch,3]
#             softmax_logit = nn.functional.softmax(logit, 1)  # [length,batch,3] 沿着最后一个维度做softmax
#             # softmax_logit[:,:,0] 表示exp(pos)/exp(pos)+exp(neg)+exp(neg2)
#             contras_loss = - torch.log(softmax_logit[:, 0])
#             # contras_loss += torch.log(softmax_logit[:, 1]) # add contras_neg
#             contras_loss = contras_loss.mean()
#             contras_loss+=contras_loss
#
#         return contras_loss

class ContrastiveLoss_local(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self):
        super(ContrastiveLoss_local, self).__init__()
        # self.kdloss = nn.KLDivLoss(reduction='batchmean')
        # self.T = T
        self.temperature = 0.5
    def forward(self, feature_v, feature_q):
        contras_loss=0
        for i in range(feature_v.size(1)):
            list = []
            aa = feature_v.size(1)
            for j in range(feature_v.size(1)):
                list = list + [j]
            feature_ori=feature_v[:,i,:]
            feature_pos = feature_q[:,i,:]
            list_neg=list.pop(i)
            if len(list)>=2:
                choice=random.sample(list,2)
                feature_neg=feature_q[:,choice[0],:]
                feature_neg1 = feature_q[:, choice[1], :]
                # pos = torch.cosine_similarity(feature_ori, feature_pos, dim=1)/self.temperature  # [length,batch]
                # neg = torch.cosine_similarity(feature_ori, feature_neg, dim=1)/self.temperature  # [length,batch]
                # neg2 = torch.cosine_similarity(feature_ori, feature_neg1, dim=1)/self.temperature  # [length,batch]
                pos = torch.cosine_similarity(feature_ori, feature_pos, dim=1)   # [length,batch]
                neg = torch.cosine_similarity(feature_ori, feature_neg, dim=1)   # [length,batch]
                neg2 = torch.cosine_similarity(feature_ori, feature_neg1, dim=1)   # [length,batch]
                logit = torch.stack((pos, neg, neg2), 1)  # [length,batch,3]
                softmax_logit = nn.functional.softmax(logit, 1)  # [length,batch,3] 沿着最后一个维度做softmax
                # softmax_logit[:,:,0] 表示exp(pos)/exp(pos)+exp(neg)+exp(neg2)
                contras_loss = - torch.log(softmax_logit[:, 0])
                # contras_loss += torch.log(softmax_logit[:, 1]) # add contras_neg
                contras_loss = contras_loss.mean()
                contras_loss+=contras_loss
            elif len(list)==1:
                choice = random.sample(list, 1)
                feature_neg = feature_q[:, choice[0], :]
                # feature_neg1 = feature_q[:, choice[1], :]
                # pos = torch.cosine_similarity(feature_ori, feature_pos, dim=1) / self.temperature  # [length,batch]
                # neg = torch.cosine_similarity(feature_ori, feature_neg, dim=1) / self.temperature  # [length,batch]
                pos = torch.cosine_similarity(feature_ori, feature_pos, dim=1)   # [length,batch]
                neg = torch.cosine_similarity(feature_ori, feature_neg, dim=1)   # [length,batch]
                # neg2 = torch.cosine_similarity(feature_ori, feature_neg1, dim=1) / self.temperature  # [length,batch]
                logit = torch.stack((pos, neg), 1)  # [length,batch,3]
                softmax_logit = nn.functional.softmax(logit, 1)  # [length,batch,3] 沿着最后一个维度做softmax
                # softmax_logit[:,:,0] 表示exp(pos)/exp(pos)+exp(neg)+exp(neg2)
                contras_loss = - torch.log(softmax_logit[:, 0])
                # contras_loss += torch.log(softmax_logit[:, 1]) # add contras_neg
                contras_loss = contras_loss.mean()
                contras_loss += contras_loss
            elif len(list) == 0:
                contras_loss += 0
        return contras_loss