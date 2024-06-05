# 残差结构，sign-encoder的结果用ctc

import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
from modules.tconv import TemporalConv
from modules import BiLSTMLayer
from modules.criterions import SeqKD
from modules.gloss_clip_question_graph_gai import GCQEncoder
from modules.criterions_contrastive import ContrastiveLoss,ContrastiveLoss_local

from pytorch_pretrained_bert.modeling import BertConfig, BertEmbeddings, BertForSignOnly,BertNoEmbedding, BertHybridFusion
# from modules.transformer import TransformerDecoder
from modules.vac import VACModel

from signjoey.decoders import Decoder, RecurrentDecoder, TransformerDecoder2Encoder,TransformerDecoder3Encoder
from signjoey.search import beam_search, greedy,transformer_greedy_2Encoder, transformer_greedy_3Encoder
from signjoey.loss import XentLoss
from signjoey.helpers import tile



import random
import itertools
import math

import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
# matplotlib.use('pdf')


import matplotlib.pyplot as plt
import torch

plt.close('all')
def seq_mask(seq):
    batch_size,seq_len=seq.size()
    sub_seq_mask = (1 - torch.triu(torch.ones((1,seq_len,seq_len)),diagonal=1)).bool()
    return sub_seq_mask

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        # self.fc = nn.Linear(1024,768)

    def forward(self, x):
        # x = self.fc(x)
        return x



class SLRModel(nn.Module):
    def __init__(self, num_classes,c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size= 768,bert_arg = None, slt_arg = None, gloss_dict=None,text_dict = None,loss_weights=None):
        super(SLRModel, self).__init__()
        # self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.gloss_dict=gloss_dict
        self.gloss_dict_word2index=dict((v, k) for k, v in gloss_dict.items())

        self.sign_encoder = VACModel(num_classes,c2d_type, conv_type, use_bn=use_bn, tm_type=tm_type,
                 hidden_size= hidden_size,gloss_dict=gloss_dict)

        self.decoder_combine = utils.Decode(gloss_dict, num_classes, 'beam')
        self.classifier_combine = nn.Linear(hidden_size, self.num_classes)
        self.classifier_sign = nn.Linear(hidden_size, self.num_classes)

        self.vis_position_embedding = nn.Embedding(
            512, hidden_size)

        # 0层 只用embedding
        self.bert_text = BertForSignOnly.from_pretrained(
            bert_arg['bert_model'], state_dict=None, num_labels=2,
            type_vocab_size=bert_arg['type_vocab_size'], relax_projection=bert_arg['relax_projection'],
            config_path=bert_arg['config_path'], task_idx=bert_arg['task_idx_proj'],
            max_position_embeddings=bert_arg['max_position_embeddings'], label_smoothing=bert_arg['label_smoothing'],
            fp32_embedding=bert_arg['fp32_embedding'],
            cache_dir=bert_arg['output_dir'] + '/.pretrained_model_{}'.format(bert_arg['global_rank']),
            drop_prob=bert_arg['drop_prob'], enable_butd=bert_arg['enable_butd'],
            len_vis_input=bert_arg['len_vis_input'], visdial_v=bert_arg['visdial_v'], loss_type=bert_arg['loss_type'],
            neg_num=bert_arg['neg_num'], adaptive_weight=bert_arg['adaptive_weight'], add_attn_fuse=bert_arg['add_attn_fuse'],
            no_h0=bert_arg['no_h0'], no_vision=bert_arg['no_vision'],num_hidden_layers=0,
            use_high_layer = None)

        self.GCQEncoder = GCQEncoder(nfeat = hidden_size, nhid = hidden_size, dropout = 0.1, alpha = 0.2, nheads = 8, nlayers = 2)
        
        self.attn_clip = BertNoEmbedding.from_pretrained(
            bert_arg['bert_model'], state_dict={}, num_labels=2,
            type_vocab_size=bert_arg['type_vocab_size'], relax_projection=bert_arg['relax_projection'],
            config_path=bert_arg['config_path'], task_idx=bert_arg['task_idx_proj'],
            max_position_embeddings=bert_arg['max_position_embeddings'], label_smoothing=bert_arg['label_smoothing'],
            fp32_embedding=bert_arg['fp32_embedding'],
            cache_dir=bert_arg['output_dir'] + '/.pretrained_model_{}'.format(bert_arg['global_rank']),
            drop_prob=bert_arg['drop_prob'], enable_butd=bert_arg['enable_butd'],
            len_vis_input=bert_arg['len_vis_input'], visdial_v=bert_arg['visdial_v'], loss_type="ctc",
            neg_num=bert_arg['neg_num'], adaptive_weight=bert_arg['adaptive_weight'], add_attn_fuse=bert_arg['add_attn_fuse'],
            no_h0=bert_arg['no_h0'], no_vision=bert_arg['no_vision'],num_hidden_layers=bert_arg['num_hidden_layers_clip'],
            use_high_layer = None)

        self.attn_gloss = BertNoEmbedding.from_pretrained(
            bert_arg['bert_model'], state_dict={}, num_labels=2,
            type_vocab_size=bert_arg['type_vocab_size'], relax_projection=bert_arg['relax_projection'],
            config_path=bert_arg['config_path'], task_idx=bert_arg['task_idx_proj'],
            max_position_embeddings=bert_arg['max_position_embeddings'], label_smoothing=bert_arg['label_smoothing'],
            fp32_embedding=bert_arg['fp32_embedding'],
            cache_dir=bert_arg['output_dir'] + '/.pretrained_model_{}'.format(bert_arg['global_rank']),
            drop_prob=bert_arg['drop_prob'], enable_butd=bert_arg['enable_butd'],
            len_vis_input=bert_arg['len_vis_input'], visdial_v=bert_arg['visdial_v'], loss_type="ctc",
            neg_num=bert_arg['neg_num'], adaptive_weight=bert_arg['adaptive_weight'], add_attn_fuse=bert_arg['add_attn_fuse'],
            no_h0=bert_arg['no_h0'], no_vision=bert_arg['no_vision'],num_hidden_layers=bert_arg['num_hidden_layers_gloss'],
            use_high_layer = None)

        self.attn_question = BertNoEmbedding.from_pretrained(
            bert_arg['bert_model'], state_dict={}, num_labels=2,
            type_vocab_size=bert_arg['type_vocab_size'], relax_projection=bert_arg['relax_projection'],
            config_path=bert_arg['config_path'], task_idx=bert_arg['task_idx_proj'],
            max_position_embeddings=bert_arg['max_position_embeddings'], label_smoothing=bert_arg['label_smoothing'],
            fp32_embedding=bert_arg['fp32_embedding'],
            cache_dir=bert_arg['output_dir'] + '/.pretrained_model_{}'.format(bert_arg['global_rank']),
            drop_prob=bert_arg['drop_prob'], enable_butd=bert_arg['enable_butd'],
            len_vis_input=bert_arg['len_vis_input'], visdial_v=bert_arg['visdial_v'], loss_type="ctc",
            neg_num=bert_arg['neg_num'], adaptive_weight=bert_arg['adaptive_weight'], add_attn_fuse=bert_arg['add_attn_fuse'],
            no_h0=bert_arg['no_h0'], no_vision=bert_arg['no_vision'],num_hidden_layers=bert_arg['num_hidden_layers_question'],
            use_high_layer = None)
        self.tau = 5
        self.tau_decay = -0.05

        

        bert_embedding_config = BertConfig(len(text_dict))

        self.text_embedding = BertEmbeddings(bert_embedding_config)

# 需要额外加1，因为gloss_dict中没有blank
        bert_embedding_config_gloss = BertConfig(len(gloss_dict) + 1)
        # bert_embedding_config = {'vocab_size':137,'hidden_size':768,'max_position_embeddings':512,'type_vocab_size':2,'hidden_dropout_prob':0.1}
        self.gloss_embedding = BertEmbeddings(bert_embedding_config_gloss)

        if not slt_arg == None:
            self.decoder = TransformerDecoder3Encoder(
                    num_layers = slt_arg['num_layers'],
                    num_heads = slt_arg['num_heads'],
                    hidden_size = slt_arg['hidden_size'],
                    ff_size = slt_arg['ff_size'],
                    dropout = slt_arg['dropout'],
                    # emb_dropout = slt_arg['emb_dropout'],
                    vocab_size = len(text_dict),
                    # freeze = slt_arg['freeze'],
                )


        self.register_backward_hook(self.backward_hook)
    def train(self, mode=True):
        super(SLRModel, self).train(mode)
        if mode:
            self.tau = self.tau * np.exp(self.tau_decay)
            print('current tau: ', self.tau)
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

        return x




    def get_attn_mask(self, one_len, max_len):
        # self-attention mask
        input_mask = torch.zeros(max_len, max_len, dtype=torch.long)
        input_mask[:, :one_len].fill_(1)
        return input_mask
    def get_attn_mask_boolean(self, one_len, max_len):
        # self-attention mask
        input_mask = torch.zeros(max_len, max_len, dtype=torch.bool)
        input_mask[:, :one_len].fill_(True)
        return input_mask 

    def get_attn_mask_multipart(self, len_v, max_len_v, len_q, max_len_q):
        # self-attention mask
        input_mask = torch.zeros(max_len_q + max_len_v, max_len_q + max_len_v, dtype=torch.long)

        # 输入为[CLS] TEXT [SEP] VIDEO  但处理后的text中包括了[CLS][SEP]
        input_mask[:, 0: len_q].fill_(1)
        input_mask[:, max_len_q: max_len_q + len_v].fill_(1)
        return input_mask

    def get_attn_mask_cross(self, len_v, max_len_v, len_q, max_len_q):
        mask = torch.zeros(max_len_v, max_len_q, dtype=torch.long)
        mask[:, 0: len_q].fill_(1)
        return mask


    def forward_pretrain(self, x, len_x_all, label, label_lgt, label_word, label_word_lgt, text, text_lgt,next_sentence_label):
            # videos
        batch, temp, channel, height, width = x.shape
        res_dict = self.sign_encoder(x, len_x_all)
        lgt = res_dict['feat_len']
        conv_logits = res_dict['conv_logits']
        sequence_logits_sign = res_dict['sequence_logits_sign']
        sign_feat = res_dict['sign_feat']
        sign_feat = sign_feat.transpose(0,1)

        # 获取表示手语视频帧位置的特征
        position_ids = torch.arange(
                max(lgt) , dtype=torch.long, device=sequence_logits_sign.device)
        position_ids = position_ids.expand(batch,max(lgt) )

        # 获取mask以遮蔽被补背景帧的帧和被补0的文本
        mask_sign = torch.zeros(batch,max(lgt), max(lgt) , dtype=torch.long,device=sequence_logits_sign.device)
        for i in range(batch):
            mask_sign[i] = self.get_attn_mask(lgt[i],max(lgt))

        mask_text = torch.zeros(batch,max(text_lgt), max(text_lgt) , dtype=torch.long,device=text.device)
        for i in range(batch):
            mask_text[i] = self.get_attn_mask(text_lgt[i],max(text_lgt))

        mask_VasQ = torch.zeros(batch, max(lgt), max(text_lgt) , dtype=torch.long,device=text.device)
        for i in range(batch):
            mask_VasQ[i] = self.get_attn_mask_cross(lgt[i], max(lgt), text_lgt[i], max(text_lgt))

       
        # outputs_sign = self.classifier_sign(bert_output_sign_video.transpose(0,1))
        pred_sign = self.decoder_combine.decode(sequence_logits_sign, lgt, batch_first=False, probs=False)

        return {
            # "framewise_features": bert_output_fusion,
            "feat_len": lgt,
            "conv_logits" : conv_logits,
            "sequence_logits_sign" : sequence_logits_sign,
            "recognized_sents_sign": pred_sign,
            # "next_sentence_loss": next_sentence_loss,
        }

    def forward_train(self, x, len_x_all, label, label_lgt, label_word, label_word_lgt, text, text_lgt,text_neg1, text_lgt_neg1,text_neg2, text_lgt_neg2,next_sentence_label,info):
            # videos
        batch, temp, channel, height, width = x.shape
        res_dict = self.sign_encoder(x, len_x_all)
        lgt = res_dict['feat_len']
        conv_logits = res_dict['conv_logits']
        sequence_logits_sign = res_dict['sequence_logits_sign']
        sign_feat = res_dict['sign_feat']
        sign_feat = sign_feat.transpose(0,1)

       # 获取表示手语视频帧位置的特征
        position_ids = torch.arange(
                max(lgt) , dtype=torch.long, device=sequence_logits_sign.device)
        position_ids = position_ids.expand(batch,max(lgt) )

        # 获取mask以遮蔽被补背景帧的帧和被补0的文本
        mask_sign = torch.zeros(batch,max(lgt), max(lgt) , dtype=torch.long,device=sequence_logits_sign.device)
        for i in range(batch):
            mask_sign[i] = self.get_attn_mask(lgt[i],max(lgt))

        mask_text = torch.zeros(batch,max(text_lgt), max(text_lgt) , dtype=torch.long,device=text.device)
        for i in range(batch):
            mask_text[i] = self.get_attn_mask(text_lgt[i],max(text_lgt))

        mask_text_neg1 = torch.zeros(batch, max(text_lgt_neg1), max(text_lgt_neg1), dtype=torch.long, device=text_neg1.device)
        for i in range(batch):
            mask_text_neg1[i] = self.get_attn_mask(text_lgt_neg1[i], max(text_lgt_neg1))
        mask_text_neg2 = torch.zeros(batch, max(text_lgt_neg2), max(text_lgt_neg2), dtype=torch.long, device=text_neg2.device)
        for i in range(batch):
            mask_text_neg2[i] = self.get_attn_mask(text_lgt_neg2[i], max(text_lgt_neg2))

        mask_VasQ = torch.zeros(batch, max(lgt), max(text_lgt) , dtype=torch.long,device=text.device)
        for i in range(batch):
            mask_VasQ[i] = self.get_attn_mask_cross(lgt[i], max(lgt), text_lgt[i], max(text_lgt))


        bert_output_text = self.bert_text(vis_feats=None,vis_pe=None,input_ids=text,attention_mask=mask_text)
        bert_output_text_neg1 = self.bert_text(vis_feats=None, vis_pe=None, input_ids=text_neg1, attention_mask=mask_text_neg1)
        bert_output_text_neg2 = self.bert_text(vis_feats=None, vis_pe=None, input_ids=text_neg2, attention_mask=mask_text_neg2)
        # outputs_sign = self.classifier_sign(bert_output_sign_video.transpose(0,1))
        pred_sign = self.decoder_combine.decode(sequence_logits_sign, lgt, batch_first=False, probs=False)
        
        pred_sign_notext = self.decoder_combine.decode(sequence_logits_sign, lgt, batch_first=False, probs=False,search_mode = "no_text")
        pred_len = max([len(i) for i in pred_sign_notext])
        padded_pred_sign_notext = []
        for i in range(0,batch):
            # [PAD]对应0
            padded_pred_sign_notext.append( pred_sign_notext[i] + [0] * (pred_len - len(pred_sign_notext[i])))
        pred_sign_notext_tensor = torch.LongTensor(padded_pred_sign_notext).cuda()

        mask_gloss = torch.zeros(batch, pred_len, pred_len, dtype=torch.long,device=sequence_logits_sign.device)
        for i in range(batch):
            mask_gloss[i] = self.get_attn_mask(len(pred_sign_notext[i]), pred_len)

        gloss_input = self.gloss_embedding(vis_feats=None,vis_pe=None,input_ids=pred_sign_notext_tensor)

        try:
            pred_sign_max = self.decoder_combine.decode(sequence_logits_sign, lgt, batch_first=False, probs=False,search_mode = "max")
        except Exception as e:
            print(e)
        # finally:
        #     pass

        blank_mask = torch.ones(batch, max(lgt)).cuda()
        gloss_clip_mask = torch.zeros(batch, pred_len, max(lgt)) 

        for batch_idx in range(0,batch):
            # gloss_input_one = []
            start_blank = True
            end_blank = True

            for clip_idx in range(0,len(pred_sign_max[batch_idx])):
                clip_gloss = pred_sign_max[batch_idx][clip_idx][1]
                if start_blank and clip_gloss == 0:
                    blank_mask[batch_idx,clip_idx] = 0
                elif not clip_gloss == 0:
                    start_blank = False
                    if clip_gloss in pred_sign_notext[batch_idx]:
                        gloss_clip_mask[batch_idx, pred_sign_notext[batch_idx].index(clip_gloss), clip_idx] = 1 
                    # gloss_input_one
                # 道中的blank，可以考虑赋予他们gloss
                elif clip_gloss == 0:
                    blank_mask[batch_idx,clip_idx] = 0
                else:
                    continue
            for clip_idx in range(len(pred_sign_max[batch_idx]) - 1, 0, -1):
                clip_gloss = pred_sign_max[batch_idx][clip_idx][1]
                if end_blank and clip_gloss == 0:
                    blank_mask[batch_idx, clip_idx] = 0
                    # gloss_clip_mask[batch_idx, :, clip_idx] = 0
                else:
                    break

        #### fuse clip based on the gloss:
        clip_fuse_batch=torch.zeros(batch, gloss_clip_mask.size(1), sign_feat.size(2))
        for i in range(batch):
            gloss_clip_mask_batch = gloss_clip_mask[i, :, :]
            clip_batch=sign_feat[i,:,:]
            clip_fuse_batch[i,:,:]=torch.mm(gloss_clip_mask_batch.cuda(),clip_batch)


        gloss_question_mask = torch.zeros(batch, pred_len, max(text_lgt) , dtype=torch.long,device=text.device)
        for i in range(batch):
            gloss_question_mask[i] = self.get_attn_mask_cross(len(pred_sign_notext[i]), pred_len, text_lgt[i], max(text_lgt))
        gloss_question_mask_gai=torch.zeros(batch, pred_len, max(text_lgt))
        ## select word based on gloss
        word_fuse_batch = torch.zeros(batch, pred_len, gloss_input.size(2))
        gloss_question_matrix=[]
        for i in range(batch):
            gloss_question_mask_batch = gloss_input[i, :, :]
            question_batch = bert_output_text[i, :, :]
            weight = torch.mm(gloss_question_mask_batch.cuda(), question_batch.transpose(1,0))
            weight=nn.functional.softmax(weight, 1)
            gloss_question_matrix.append(weight)
            location = torch.log(weight.clamp(min=1e-8))
            if self.training:
                action = F.gumbel_softmax(location, self.tau, hard=True, dim=1)  # B*(M+1)
            else:
                action = F.gumbel_softmax(location, 1e-5, hard=True, dim=1)  # B*(M+1)
            word_fuse_batch[i,:,:]=torch.mm(action,question_batch)
            gloss_question_mask_gai[i,:,:]=action




        gloss_output, clip_output_withblank, question_output = self.GCQEncoder(gloss_input, sign_feat, bert_output_text, gloss_clip_mask,gloss_question_mask_gai)



        # 被预测为blank的clip仍使用原特征，再加上positiosnembedding
        clip_output = clip_output_withblank * blank_mask.unsqueeze(2) + sign_feat * (1 - blank_mask.unsqueeze(2)) + self.vis_position_embedding(position_ids)


        gloss_output_self = self.attn_gloss(gloss_output, mask_gloss)
        clip_output_self = self.attn_clip(clip_output, mask_sign)
        question_output_self = self.attn_question(question_output, mask_text)


        bert_output_classifier = clip_output_self.transpose(0,1)
        sequence_logits = self.classifier_combine(bert_output_classifier)
        # greedyPred = self.decoder_combine.decode(outputs, lgt, batch_first=False, probs=False,search_mode='max')

        pred = self.decoder_combine.decode(sequence_logits, lgt, batch_first=False, probs=False)

        # 输入不包括<eos>

        trg_embed = self.text_embedding(vis_feats=None,vis_pe=None,input_ids=label_word)
        trg_mask = torch.zeros(batch,max(label_word_lgt), max(label_word_lgt) , dtype=torch.bool,device=text.device)
        for i in range(batch):
            trg_mask[i] = self.get_attn_mask_boolean(label_word_lgt[i],max(label_word_lgt))
        

        src_mask1 = torch.zeros(batch, pred_len, pred_len , dtype=torch.bool,device=sequence_logits_sign.device)
        for i in range(batch):
            src_mask1[i] = self.get_attn_mask_boolean(len(pred_sign_notext[i]),pred_len) 
        src_mask2 = torch.zeros(batch,max(lgt), max(lgt) , dtype=torch.bool,device=sequence_logits_sign.device)
        for i in range(batch):
            src_mask2[i] = self.get_attn_mask_boolean(lgt[i],max(lgt)) 

        src_mask3 = torch.zeros(batch,max(text_lgt), max(text_lgt) , dtype=torch.bool,device=sequence_logits_sign.device)
        for i in range(batch):
            src_mask3[i] = self.get_attn_mask_boolean(text_lgt[i],max(text_lgt)) 


        decoder_outputs = self.decoder(
            trg_embed=trg_embed, 
            encoder_output1 = gloss_output_self,
            encoder_output2 = clip_output_self,
            encoder_output3 = question_output_self,
            src_mask1=src_mask1[:,0:1,:],
            src_mask2=src_mask2[:,0:1,:],
            src_mask3=src_mask3[:,0:1,:],
            trg_mask=trg_mask,
        )

        word_outputs, _, _, _ = decoder_outputs
        # Calculate Translation Loss
        txt_log_probs = F.log_softmax(word_outputs, dim=-1)



        return {
            "feat_len": lgt,
            "conv_logits" : conv_logits,
            "sequence_logits": sequence_logits,
            "sequence_logits_sign" : sequence_logits_sign,
            "recognized_sents": pred,
            "recognized_sents_sign": pred_sign,
            "txt_log_probs": txt_log_probs,
            "feature_v": clip_output_self,
            "feature_q": question_output_self,
            "feature_q_neg1": bert_output_text_neg1,
            "feature_q_neg2": bert_output_text_neg2,
            "feature_v_all": clip_fuse_batch,
            "feature_q_all": word_fuse_batch,
        }

    def transformer_greedy(
        src_mask,
        embed,
        bos_index: int,
        eos_index: int,
        max_output_length: int,
        decoder: Decoder,
        encoder_output,
        encoder_hidden,
    ) :
        """
        Special greedy function for transformer, since it works differently.
        The transformer remembers all previous states and attends to them.

        :param src_mask: mask for source inputs, 0 for positions after </s>
        :param embed: target embedding layer
        :param bos_index: index of <s> in the vocabulary
        :param eos_index: index of </s> in the vocabulary
        :param max_output_length: maximum length for the hypotheses
        :param decoder: decoder to use for greedy decoding
        :param encoder_output: encoder hidden states for attention
        :param encoder_hidden: encoder final state (unused in Transformer)
        :return:
            - stacked_output: output hypotheses (2d array of indices),
            - stacked_attention_scores: attention scores (3d array)
        """

        batch_size = src_mask.size(0)

        # start with BOS-symbol for each sentence in the batch
        ys = encoder_output.new_full([batch_size, 1], bos_index, dtype=torch.long)

        # a subsequent mask is intersected with this in decoder forward pass
        trg_mask = src_mask.new_ones([1, 1, 1])
        finished = src_mask.new_zeros((batch_size)).byte()

        for _ in range(max_output_length):

            trg_embed = embed(ys)  # embed the previous tokens

            # pylint: disable=unused-variable
            with torch.no_grad():
                logits, out, _, _ = decoder(
                    trg_embed=trg_embed,
                    encoder_output=encoder_output,
                    encoder_hidden=None,
                    src_mask=src_mask,
                    unroll_steps=None,
                    hidden=None,
                    trg_mask=trg_mask,
                )

                logits = logits[:, -1]
                _, next_word = torch.max(logits, dim=1)
                next_word = next_word.data
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

            # check if previous symbol was <eos>
            is_eos = torch.eq(next_word, eos_index)
            finished += is_eos
            # stop predicting if <eos> reached for all elements in batch
            if (finished >= 1).sum() == batch_size:
                break

        ys = ys[:, 1:]  # remove BOS-symbol
        return ys.detach().cpu().numpy(), None

    def beam_search(
        decoder: Decoder,
        size: int,
        bos_index: int,
        eos_index: int,
        pad_index: int,
        encoder_output,
        encoder_hidden,
        src_mask,
        max_output_length: int,
        alpha: float,
        embed,
        n_best: int = 1,
    ):
        """
        Beam search with size k.
        Inspired by OpenNMT-py, adapted for Transformer.

        In each decoding step, find the k most likely partial hypotheses.

        :param decoder:
        :param size: size of the beam
        :param bos_index:
        :param eos_index:
        :param pad_index:
        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param max_output_length:
        :param alpha: `alpha` factor for length penalty
        :param embed:
        :param n_best: return this many hypotheses, <= beam (currently only 1)
        :return:
            - stacked_output: output hypotheses (2d array of indices),
            - stacked_attention_scores: attention scores (3d array)
        """
        assert size > 0, "Beam size must be >0."
        assert n_best <= size, "Can only return {} best hypotheses.".format(size)

        # init
        transformer = isinstance(decoder, TransformerDecoder3Encoder)
        batch_size = src_mask.size(0)
        att_vectors = None  # not used for Transformer

        # Recurrent models only: initialize RNN hidden state
        # pylint: disable=protected-access
        if not transformer:
            hidden = decoder._init_hidden(encoder_hidden)
        else:
            hidden = None

        # tile encoder states and decoder initial states beam_size times
        if hidden is not None:
            hidden = tile(hidden, size, dim=1)  # layers x batch*k x dec_hidden_size

        encoder_output = tile(
            encoder_output.contiguous(), size, dim=0
        )  # batch*k x src_len x enc_hidden_size
        src_mask = tile(src_mask, size, dim=0)  # batch*k x 1 x src_len

        # Transformer only: create target mask
        if transformer:
            trg_mask = src_mask.new_ones([1, 1, 1])  # transformer only
        else:
            trg_mask = None

        # numbering elements in the batch
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=encoder_output.device
        )

        # numbering elements in the extended batch, i.e. beam size copies of each
        # batch element
        beam_offset = torch.arange(
            0, batch_size * size, step=size, dtype=torch.long, device=encoder_output.device
        )

        # keeps track of the top beam size hypotheses to expand for each element
        # in the batch to be further decoded (that are still "alive")
        alive_seq = torch.full(
            [batch_size * size, 1],
            bos_index,
            dtype=torch.long,
            device=encoder_output.device,
        )

        # Give full probability to the first beam on the first step.
        topk_log_probs = torch.zeros(batch_size, size, device=encoder_output.device)
        topk_log_probs[:, 1:] = float("-inf")

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]

        results = {
            "predictions": [[] for _ in range(batch_size)],
            "scores": [[] for _ in range(batch_size)],
            "gold_score": [0] * batch_size,
        }

        for step in range(max_output_length):

            # This decides which part of the predicted sentence we feed to the
            # decoder to make the next prediction.
            # For Transformer, we feed the complete predicted sentence so far.
            # For Recurrent models, only feed the previous target word prediction
            if transformer:  # Transformer
                decoder_input = alive_seq  # complete prediction so far
            else:  # Recurrent
                decoder_input = alive_seq[:, -1].view(-1, 1)  # only the last word

            # expand current hypotheses
            # decode one single step
            # logits: logits for final softmax
            # pylint: disable=unused-variable
            trg_embed = embed(decoder_input)
            logits, hidden, att_scores, att_vectors = decoder(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                trg_embed=trg_embed,
                hidden=hidden,
                prev_att_vector=att_vectors,
                unroll_steps=1,
                trg_mask=trg_mask,  # subsequent mask for Transformer only
            )

            # For the Transformer we made predictions for all time steps up to
            # this point, so we only want to know about the last time step.
            if transformer:
                logits = logits[:, -1]  # keep only the last time step
                hidden = None  # we don't need to keep it for transformer

            # batch*k x trg_vocab
            log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

            # multiply probs by the beam probability (=add logprobs)
            log_probs += topk_log_probs.view(-1).unsqueeze(1)
            curr_scores = log_probs.clone()

            # compute length penalty
            if alpha > -1:
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
                curr_scores /= length_penalty

            # flatten log_probs into a list of possibilities
            curr_scores = curr_scores.reshape(-1, size * decoder.output_size)

            # pick currently best top k hypotheses (flattened order)
            topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

            if alpha > -1:
                # recover original log probs
                topk_log_probs = topk_scores * length_penalty
            else:
                topk_log_probs = topk_scores.clone()

            # reconstruct beam origin and true word ids from flattened order
            topk_beam_index = topk_ids.div(decoder.output_size)
            topk_ids = topk_ids.fmod(decoder.output_size)

            # map beam_index to batch_index in the flat representation
            batch_index = topk_beam_index + beam_offset[
                : topk_beam_index.size(0)
            ].unsqueeze(1)
            select_indices = batch_index.view(-1)

            # append latest prediction
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
            )  # batch_size*k x hyp_len

            is_finished = topk_ids.eq(eos_index)
            if step + 1 == max_output_length:
                is_finished.fill_(True)
            # end condition is whether the top beam is finished
            end_condition = is_finished[:, 0].eq(True)

            # save finished hypotheses
            if is_finished.any():
                predictions = alive_seq.view(-1, size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(True)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # store finished hypotheses for this batch
                    for j in finished_hyp:
                        # Check if the prediction has more than one EOS.
                        # If it has more than one EOS, it means that the prediction should have already
                        # been added to the hypotheses, so you don't have to add them again.
                        if (predictions[i, j, 1:] == eos_index).nonzero().numel() < 2:
                            hypotheses[b].append(
                                (
                                    topk_scores[i, j],
                                    predictions[i, j, 1:],
                                )  # ignore start_token
                            )
                    # if the batch reached the end, save the n_best hypotheses
                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                non_finished = end_condition.eq(False).nonzero().view(-1)
                # if all sentences are translated, no need to go further
                # pylint: disable=len-as-condition
                if len(non_finished) == 0:
                    break
                # remove finished batches for the next step
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(
                    -1, alive_seq.size(-1)
                )

            # reorder indices, outputs and masks
            select_indices = batch_index.view(-1)
            encoder_output = encoder_output.index_select(0, select_indices)
            src_mask = src_mask.index_select(0, select_indices)

            if hidden is not None and not transformer:
                if isinstance(hidden, tuple):
                    # for LSTMs, states are tuples of tensors
                    h, c = hidden
                    h = h.index_select(1, select_indices)
                    c = c.index_select(1, select_indices)
                    hidden = (h, c)
                else:
                    # for GRUs, states are single tensors
                    hidden = hidden.index_select(1, select_indices)

            if att_vectors is not None:
                att_vectors = att_vectors.index_select(0, select_indices)

        def pad_and_stack_hyps(hyps, pad_value):
            filled = (
                np.ones((len(hyps), max([h.shape[0] for h in hyps])), dtype=int) * pad_value
            )
            for j, h in enumerate(hyps):
                for k, i in enumerate(h):
                    filled[j, k] = i
            return filled

        # from results to stacked outputs
        assert n_best == 1
        # only works for n_best=1 for now
        final_outputs = pad_and_stack_hyps(
            [r[0].cpu().numpy() for r in results["predictions"]], pad_value=pad_index
        )

        return final_outputs, None


    def forward_test(self, x, len_x_all, label, label_lgt, label_word, label_word_lgt, text, text_lgt,translation_beam_size,translation_beam_alpha):
        batch, temp, channel, height, width = x.shape
        res_dict = self.sign_encoder(x, len_x_all)
        lgt = res_dict['feat_len']
        conv_logits = res_dict['conv_logits']
        sequence_logits_sign = res_dict['sequence_logits_sign']
        sign_feat = res_dict['sign_feat']
        sign_feat = sign_feat.transpose(0,1)

       # 获取表示手语视频帧位置的特征
        position_ids = torch.arange(
                max(lgt) , dtype=torch.long, device=sequence_logits_sign.device)
        position_ids = position_ids.expand(batch,max(lgt) )

        # 获取mask以遮蔽被补背景帧的帧和被补0的文本
        mask_sign = torch.zeros(batch,max(lgt), max(lgt) , dtype=torch.long,device=sequence_logits_sign.device)
        for i in range(batch):
            mask_sign[i] = self.get_attn_mask(lgt[i],max(lgt))

        mask_text = torch.zeros(batch,max(text_lgt), max(text_lgt) , dtype=torch.long,device=text.device)
        for i in range(batch):
            mask_text[i] = self.get_attn_mask(text_lgt[i],max(text_lgt))

        mask_VasQ = torch.zeros(batch, max(lgt), max(text_lgt) , dtype=torch.long,device=text.device)
        for i in range(batch):
            mask_VasQ[i] = self.get_attn_mask_cross(lgt[i], max(lgt), text_lgt[i], max(text_lgt))

        bert_output_text = self.bert_text(vis_feats=None,vis_pe=None,input_ids=text,attention_mask=mask_text)
        # outputs_sign = self.classifier_sign(bert_output_sign_video.transpose(0,1))
        pred_sign = self.decoder_combine.decode(sequence_logits_sign, lgt, batch_first=False, probs=False)
        # print('gloss:',pred_sign)
        pred_sign_notext = self.decoder_combine.decode(sequence_logits_sign, lgt, batch_first=False, probs=False,search_mode = "no_text")
        pred_len = max([len(i) for i in pred_sign_notext])
        padded_pred_sign_notext = []
        for i in range(0,batch):
            # [PAD]对应0
            padded_pred_sign_notext.append( pred_sign_notext[i] + [0] * (pred_len - len(pred_sign_notext[i])))
        pred_sign_notext_tensor = torch.LongTensor(padded_pred_sign_notext).cuda()

        mask_gloss = torch.zeros(batch, pred_len, pred_len, dtype=torch.long,device=sequence_logits_sign.device)
        for i in range(batch):
            mask_gloss[i] = self.get_attn_mask(len(pred_sign_notext[i]), pred_len)

        gloss_input = self.gloss_embedding(vis_feats=None,vis_pe=None,input_ids=pred_sign_notext_tensor)

        pred_sign_max = self.decoder_combine.decode(sequence_logits_sign, lgt, batch_first=False, probs=False,search_mode = "max")


        blank_mask = torch.ones(batch, max(lgt)).cuda()
        gloss_clip_mask = torch.zeros(batch, pred_len, max(lgt)) 
        # print('question_len:',text_lgt)
        for batch_idx in range(0,batch):
            # gloss_input_one = []
            start_blank = True
            end_blank = True
            for clip_idx in range(0,len(pred_sign_max[batch_idx])):
                clip_gloss = pred_sign_max[batch_idx][clip_idx][1]
                if start_blank and clip_gloss == 0:
                    blank_mask[batch_idx,clip_idx] = 0
                elif not clip_gloss == 0:
                    start_blank = False
                    if clip_gloss in pred_sign_notext[batch_idx]:
                        gloss_clip_mask[batch_idx, pred_sign_notext[batch_idx].index(clip_gloss), clip_idx] = 1 
                    # gloss_input_one
                # 道中的blank，可以考虑赋予他们gloss
                elif clip_gloss == 0:
                    blank_mask[batch_idx,clip_idx] = 0
                else:
                    continue
            for clip_idx in range(len(pred_sign_max[batch_idx]) - 1, 0, -1):
                clip_gloss = pred_sign_max[batch_idx][clip_idx][1]
                if end_blank and clip_gloss == 0:
                    blank_mask[batch_idx, clip_idx] = 0
                    # gloss_clip_mask[batch_idx, :, clip_idx] = 0
                else:
                    break

        gloss_question_mask = torch.zeros(batch, pred_len, max(text_lgt) , dtype=torch.long,device=text.device)
        for i in range(batch):
            gloss_question_mask[i] = self.get_attn_mask_cross(len(pred_sign_notext[i]), pred_len, text_lgt[i], max(text_lgt))
        gloss_question_mask_gai = torch.zeros(batch, pred_len, max(text_lgt))
        ## select word based on gloss
        word_fuse_batch = torch.zeros(batch, pred_len, gloss_input.size(2))
        gloss_question_matrix = []
        for i in range(batch):
            gloss_question_mask_batch = gloss_input[i, :, :]
            question_batch = bert_output_text[i, :, :]
            weight = torch.mm(gloss_question_mask_batch.cuda(), question_batch.transpose(1, 0))
            weight = nn.functional.softmax(weight, 1)

            gloss_question_matrix.append(weight)
            location = torch.log(weight.clamp(min=1e-8))
            if self.training:
                action = F.gumbel_softmax(location, self.tau, hard=True, dim=1)  # B*(M+1)
            else:
                action = F.gumbel_softmax(location, 1e-5, hard=True, dim=1)  # B*(M+1)
            # print('gloss_question_matrix:', np.round(weight.cpu().numpy(), 5))
            idx = action.argmax(axis=0)
            # print('gloss_question_matrix:', idx)
            # print('matrix_size:', weight.size())
            word_fuse_batch[i, :, :] = torch.mm(action, question_batch)
            gloss_question_mask_gai[i, :, :] = action

        gloss_output, clip_output_withblank, question_output = self.GCQEncoder(gloss_input, sign_feat, bert_output_text, gloss_clip_mask,gloss_question_mask_gai )

        # 被预测为blank的clip仍使用原特征，再加上positionembedding
        clip_output = clip_output_withblank * blank_mask.unsqueeze(2) + sign_feat * (1 - blank_mask.unsqueeze(2)) + self.vis_position_embedding(position_ids)
        # clip_output = clip_output_withblank 

        gloss_output_self = self.attn_gloss(gloss_output, mask_gloss)
        clip_output_self = self.attn_clip(clip_output, mask_sign)
        question_output_self = self.attn_question(question_output, mask_text)


        bert_output_classifier = clip_output_self.transpose(0,1)
        sequence_logits = self.classifier_combine(bert_output_classifier)
        # greedyPred = self.decoder_combine.decode(outputs, lgt, batch_first=False, probs=False,search_mode='max')

        pred = self.decoder_combine.decode(sequence_logits, lgt, batch_first=False, probs=False)

        # 输入不包括<eos>
        unroll_steps = label_word.size(1)
        trg_embed = self.text_embedding(vis_feats=None,vis_pe=None,input_ids=label_word)
        trg_mask = torch.zeros(batch,max(label_word_lgt), max(label_word_lgt) , dtype=torch.bool,device=text.device)
        for i in range(batch):
            trg_mask[i] = self.get_attn_mask_boolean(label_word_lgt[i],max(label_word_lgt))
        

        src_mask1 = torch.zeros(batch, pred_len, pred_len , dtype=torch.bool,device=sequence_logits_sign.device)
        for i in range(batch):
            src_mask1[i] = self.get_attn_mask_boolean(len(pred_sign_notext[i]),pred_len) 
        src_mask2 = torch.zeros(batch,max(lgt), max(lgt) , dtype=torch.bool,device=sequence_logits_sign.device)
        for i in range(batch):
            src_mask2[i] = self.get_attn_mask_boolean(lgt[i],max(lgt)) 

        src_mask3 = torch.zeros(batch,max(text_lgt), max(text_lgt) , dtype=torch.bool,device=sequence_logits_sign.device)
        for i in range(batch):
            src_mask3[i] = self.get_attn_mask_boolean(text_lgt[i],max(text_lgt))


        if translation_beam_size < 2:
            stacked_txt_output, stacked_attention_scores = transformer_greedy_3Encoder(
                encoder_hidden = None,
                encoder_output1 = gloss_output_self,
                encoder_output2 = clip_output_self,
                encoder_output3 = question_output_self,
                src_mask1=src_mask1[:,0:1,:],
                src_mask2=src_mask2[:,0:1,:],
                src_mask3=src_mask3[:,0:1,:],
                embed = self.text_embedding,
                bos_index = 1,
                eos_index = 2,
                decoder=self.decoder,
                max_output_length = 30,
            )
                # batch, time, max_sgn_length
        else:  # beam size
            stacked_txt_output, stacked_attention_scores = beam_search(
                size=translation_beam_size,
                encoder_hidden = None,
                encoder_output = bert_output_classifier.transpose(0,1),
                src_mask = src_mask1[:,0:1,:],
                embed = self.text_embedding,
                max_output_length = 30,
                alpha=translation_beam_alpha,
                eos_index=2,
                pad_index=0,
                bos_index=1,
                decoder=self.decoder,
            )

        return {
            # "framewise_features": bert_output_fusion,
            "feat_len": lgt,
            "sequence_logits": sequence_logits,
            "sequence_logits_sign" : sequence_logits_sign,
            "recognized_sents": pred,
            "recognized_sents_sign": pred_sign,
            'stacked_txt_output': stacked_txt_output,
            'stacked_attention_scores': stacked_attention_scores
        }

    def forward_pretrain_test(self, x, len_x_all, label, label_lgt, label_word, label_word_lgt, text, text_lgt,translation_beam_size,translation_beam_alpha):
        batch, temp, channel, height, width = x.shape
        res_dict = self.sign_encoder(x, len_x_all)
        lgt = res_dict['feat_len']
        conv_logits = res_dict['conv_logits']
        sequence_logits_sign = res_dict['sequence_logits_sign']
        sign_feat = res_dict['sign_feat']
        sign_feat = sign_feat.transpose(0,1)


        pred_sign = self.decoder_combine.decode(sequence_logits_sign, lgt, batch_first=False, probs=False)

        return {
            # "framewise_features": bert_output_fusion,
            "feat_len": lgt,
            # "sequence_logits": sequence_logits,
            "sequence_logits_sign" : sequence_logits_sign,
            # "recognized_sents": pred,
            "recognized_sents_sign": pred_sign,
            # 'stacked_txt_output': stacked_txt_output,
            # 'stacked_attention_scores': stacked_attention_scores
        }

    def criterion_calculation(self, ret_dict, label, label_lgt,label_text,label_text_lgt,epoch):
        loss = 0

        for k, weight in self.loss_weights.items():
            if k == 'ConvCTCSign':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTCSign':
                # if epoch>2:
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits_sign"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits_sign"].detach(),
                                                           use_blank=False)

            elif k == 'SeqCTC':
                # if epoch > 2:
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'TranslationCrossEntropy':
                translation_loss = self.loss['translation'](
                    ret_dict['txt_log_probs'], label_text,)
                # loss += weight * self.loss['translation'](
                #     ret_dict['txt_log_probs'], label_text,
                # )
                if np.isinf(translation_loss.item()) or np.isnan(translation_loss.item()): 
                    print(translation_loss)
                loss += weight * translation_loss
            elif k == 'Contractive_gobal':
                loss += weight * self.loss['contrastive'](ret_dict["feature_v"],
                                                           ret_dict["feature_q"],
                                                           ret_dict["feature_q_neg1"],
                                                           ret_dict["feature_q_neg2"],
                                                           )
            elif k == 'Contractive_local':
                loss += weight * self.loss['contrastive_local'](ret_dict["feature_v_all"],
                                                           ret_dict["feature_q_all"],

                                                           )

        # return loss, loss_question, loss_question_gai, loss_video, loss_video_gai

        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['translation'] = XentLoss(
            pad_index=0, smoothing=0.0
        )
        # self.loss['distillation_disentangle'] = SeqKD(T=8)
        self.loss['contrastive'] = ContrastiveLoss()
        self.loss['contrastive_local'] = ContrastiveLoss_local()
        return self.loss

    
