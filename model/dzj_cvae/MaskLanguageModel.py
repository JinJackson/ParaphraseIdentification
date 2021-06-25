# coding=utf-8
"""PyTorch RoBERTa model. """

import math
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN, gelu
from transformers.configuration_roberta import RobertaConfig


from transformers.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel
)

from .Attention import AttentionInArgs
from .SelfAttention import SelfAttention
from .GATModel import GAT

import logging
import random
logger = logging.getLogger(__name__)

import numpy as np

import fitlog

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaPDTBModel(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.roberta_for_mlm = RobertaModel(config, add_pooling_layer=False)

        self.lm_head = RobertaLMHead(config)

        self.classifier = RobertaClassificationHead(config)

        self.laynorm = nn.LayerNorm(config.hidden_size)
        
        # 加attention
        self.self_attention = SelfAttention(input_size=768,
                                         embedding_dim=256,
                                         output_size=768
                                         )
        # self.self_atten = nn.MultiheadAttention(768, 2)
        self.attention = AttentionInArgs(input_size=768,            # 768
                                         embedding_dim=256,         # 256
                                         output_size=768            # 256, 768
                                         )
        self.self_attention2 = nn.MultiheadAttention(768, 2, dropout=0.2)
        self.attention2 = nn.MultiheadAttention(768, 2, dropout=0.2)

        self.attn_fc = nn.Linear(768, 128)
        self.atten_fc = nn.Linear(128, 256)

        # torch.Size([8, 256, 768]) torch.Size([8, 256, 256])
        # self.gat = GAT(in_features=768, n_dim=256, n_class=768)

        self.projector = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 768)            
        )

        self.init_weights()

    def _select_attention(self, sequence_output, attention_mask, arg1_first=False):
        # 加attention
        arg_len = sequence_output.shape[1] // 2
        # X: [8, 128, 768], [8, 128, 768] --> [8, 256, 768]
        # X: [8, 82, 768], [8, 82, 768] --> [8, 164, 768]
        arg1, arg2 = sequence_output[:, :arg_len, :], sequence_output[:, arg_len:, :]
        # arg1_mask, arg2_mask = attention_mask[:, :arg_len], attention_mask[:, :arg_len]
        arg1 = self.self_attention(arg1, arg1)
        arg2 = self.self_attention(arg2, arg2)
        # arg1, _ = self.self_atten(arg1, arg1, arg1)
        # arg2, _ = self.self_atten(arg2, arg2, arg2)
        
        # print(arg1.shape)

        if arg1_first:
            # seq_out, _ = self.inter_atten(arg1, arg2, arg2)  # [8, 128, 768]
            seq_out = self.attention(arg1, arg2)
        else:
            # seq_out, _ = self.inter_atten(arg2, arg1, arg1)  # [8, 128, 768]
            seq_out = self.attention(arg2, arg1)

        # print(seq_out.shape)
        return self.attn_fc(seq_out)
        # return seq_out

    def _random_mask(self, sequence_output, arg1_first=False):
        # 加attention
        arg_len = sequence_output.shape[1] // 2
        # X: [8, 128, 768], [8, 128, 768] --> [8, 256, 768]
        # X: [8, 82, 768], [8, 82, 768] --> [8, 164, 768]
        arg1, arg2 = sequence_output[:, :arg_len, :], sequence_output[:, arg_len:, :]

        if arg1_first:
            return self.attn_fc(arg1)
        else:
            return self.attn_fc(arg2)


    def _add_attention(self, sequence_output, attention_mask):
        # 加attention
        arg_len = sequence_output.shape[1] // 2
        # X: [8, 128, 768], [8, 128, 768] --> [8, 256, 768]
        # X: [8, 82, 768], [8, 82, 768] --> [8, 164, 768]
        arg1, arg2 = sequence_output[:, :arg_len, :], sequence_output[:, arg_len:, :]
        # arg1_mask, arg2_mask = attention_mask[:, :arg_len], attention_mask[:, :arg_len]
        # 目标: [8, 256, 256]
        arg1 = self.self_attention(arg1, arg1)
        arg2 = self.self_attention(arg2, arg2)
        sequence_output = self.attention(arg1, arg2, attention_mask)  # [8, 256]
        
        # logging.info('adj: ' + str(adj[0]) + ' ' + str(adj.shape))
        # sequence_output = self.gat(sequence_output, adj)
        return sequence_output

    def _add_attention_main(self, sequence_output, attention_mask=None):
        # 加attention
        arg_len = sequence_output.shape[1] // 2
        # X: [8, 128, 768], [8, 128, 768] --> [8, 256, 768]
        # X: [8, 82, 768], [8, 82, 768] --> [8, 164, 768]
        arg1, arg2 = sequence_output[:, :arg_len, :], sequence_output[:, arg_len:, :]
        # arg1_mask, arg2_mask = attention_mask[:, :arg_len], attention_mask[:, :arg_len]
        # 目标: [8, 256, 256]
        arg1, _ = self.self_attention2(arg1.transpose(0, 1), arg1.transpose(0, 1), arg1.transpose(0, 1))
        # print(arg1.shape)
        arg2, _ = self.self_attention2(arg2.transpose(0, 1), arg2.transpose(0, 1), arg2.transpose(0, 1))
        arg1_attn, _ = self.attention2(arg1, arg1, arg2) 
        arg2_attn, _ = self.attention2(arg2, arg2, arg1) 
        
        arg1_attn = arg1_attn.transpose(0, 1)
        arg2_attn = arg2_attn.transpose(0, 1)
        sequence_output = torch.cat([arg1_attn, arg2_attn], dim=1)
        
        return sequence_output

    def _mlm_attention(self, sequence_output, attention_mask, input_ids, args=None, tokenizer=None):
        torch.set_printoptions(profile="full")

        masked_input_ids = input_ids.clone().detach()
        
        sequence_for_mask = self._select_attention(sequence_output, attention_mask, arg1_first=True)
        # sequence_for_mask =  self._random_mask(sequence_output, arg1_first=True)
        sequence_for_mask = torch.sum(sequence_for_mask, dim=2)
        # sequence_for_mask = torch.argmax(sequence_for_mask, dim=2)  # [8, 128]
        # print('seq mask: ', sequence_for_mask)
        
        mask_idx = torch.argsort(sequence_for_mask, dim=1, descending=True)[:, :args.mask_num]
        # print(mask_idx.shape, mask_idx)

        for elem, idx in zip(masked_input_ids, mask_idx):
            elem[idx] = 50264    # '<mask>': 50264
        # for elem in masked_input_ids:
        #     idx = list(range(0, 128))
        #     random.shuffle(idx)
        #     idx = idx[:args.mask_num]        
        #     idx = sequence_for_mask[:, idx]            
        #     elem[128 + idx] = 50264

        # print('masked input_ids_arg2: ', masked_input_ids, '\n')
        # print('input_ids: ', input_ids)
        # print(tokenizer.convert_ids_to_tokens(masked_input_ids[0]))
        # print(tokenizer.convert_ids_to_tokens(input_ids[0]))

        sequence_for_mask = self._select_attention(sequence_output, attention_mask)
        # sequence_for_mask =  self._random_mask(sequence_output)
        sequence_for_mask = torch.sum(sequence_for_mask, dim=2)
        # sequence_for_mask = torch.argmax(sequence_for_mask, dim=2)
        mask_idx = torch.argsort(sequence_for_mask, dim=1, descending=True)[:, :args.mask_num]

        for elem, idx in zip(masked_input_ids, mask_idx):
            elem[idx] = 50264    # '<mask>': 50264

        # for elem in masked_input_ids:
        #     idx = list(range(0, 128))
        #     random.shuffle(idx)
        #     idx = idx[:args.mask_num]
        #     idx = sequence_for_mask[:, idx]
        #     elem[idx] = 50264

        # print(mask_idx.shape, mask_idx)
        # print('masked input_ids_arg1: ', masked_input_ids, '\n')
        # print('input_ids: ', input_ids)
        # print(tokenizer.convert_ids_to_tokens(masked_input_ids[0]))
        # print(tokenizer.convert_ids_to_tokens(input_ids[0]))

        return masked_input_ids, input_ids

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        Training=False,
        tokenizer=None,
        args=None,
        global_step=0
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        
        # 在这修改代码
        sequence_output = self.laynorm(sequence_output)
        
        ###########################################################################################
        # MLM任务
        input_ids_label = None
        if Training and args.do_mlm > 0:
            masked_ids, input_ids_label = self._mlm_attention(sequence_output, attention_mask, input_ids, args, tokenizer=tokenizer)
            mlm_tasks = self.roberta(masked_ids, 
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict,
                                    )
            mlm_sequence_output = mlm_tasks[0]
   
            prediction_scores = self.lm_head(mlm_sequence_output)

            # 测试
            if global_step % 100 == 0:
                pre = torch.argmax(prediction_scores, dim=2)
                print('pre: ', tokenizer.convert_ids_to_tokens(pre[0]))
                print('label: ', tokenizer.convert_ids_to_tokens(input_ids[0]))

        masked_mlm_loss = None
        if input_ids_label is not None:
            loss_fct_mlm = CrossEntropyLoss()
            masked_mlm_loss = loss_fct_mlm(prediction_scores.view(-1, self.config.vocab_size), input_ids_label.view(-1))
        ###############################################################################################

        # 主任务
        sequence_output = self._add_attention(sequence_output, attention_mask)
        sequence_output = self.atten_fc(sequence_output.transpose(1, 2)).transpose(1, 2)

        # sequence_output = self._add_attention_main(sequence_output)


        sequence_output = self.laynorm(sequence_output)

        logits = self.classifier(sequence_output)
        

        # 计算loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                # [8, 2], [8]
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
                # loss_主 + loss_mlm
                if Training and args.do_mlm > 0 and masked_mlm_loss is not None: 
                    loss = loss + args.mlm_theta * masked_mlm_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




