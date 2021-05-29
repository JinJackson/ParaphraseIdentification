# coding=utf-8
"""PyTorch RoBERTa model. """

import math
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

# from transformers.activations import ACT2FN, gelu
# from transformers.configuration_roberta import RobertaConfig


from transformers.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaLMHead
)

from .CVAEModel import CVAEModel, Similarity
from .Attention import AttentionInArgs
from .SelfAttention import SelfAttention
from .GATModel import GAT

import logging
logger = logging.getLogger(__name__)


import fitlog


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
        self.classifier = RobertaClassificationHead(config)

        self.laynorm = nn.LayerNorm(config.hidden_size)

        self.cvae = CVAEModel(config.hidden_size, config.hidden_size)

        # 加attention
        self.self_attention = SelfAttention(input_size=768,
                                         embedding_dim=256,
                                         output_size=768
                                         )
        # self.self_atten = nn.MultiheadAttention(768, 2)
        self.attention = AttentionInArgs(input_size=768,
                                         embedding_dim=256,
                                         output_size=768
                                         )
        # torch.Size([8, 256, 768]) torch.Size([8, 256, 256])
        self.gat = GAT(in_features=768, n_dim=256, n_class=768)

        self.projector = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 768)            
        )

        self.init_weights()

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
        # 只是做个 mm
        # sequence_output = self.gat(sequence_output, adj)
        return sequence_output

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
        do_cvae=0,              # 先做几次cvae
        global_step=0,
        args=None
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

        # 数据增强
        sequence_output = self._add_attention(sequence_output, attention_mask)
        

        # logger.info('seq.shape: ' + str(sequence_output.shape))
        if do_cvae > 0:
            # 训练CVAE
            out, mu, logvar = self.cvae(x=sequence_output, y=labels, Training=True, device=args.device)
            # 原始数据+con
            y_c = self.cvae.to_categrical(labels, device=args.device)
            # y_c = y_c.unsqueeze(1)
            # 输入样本和标签y的one-hot向量连接
            con = nn.Dropout(args.cvae_dropout)(sequence_output) + y_c

            y_n = self.cvae.to_categrical_neg(labels, device=args.device)
            # y_n = y_n.unsqueeze(1)
            neg_con = nn.Dropout(args.cvae_dropout)(sequence_output ) + y_n

            # update: 4/26
            # sim = Similarity(1000)
            # pos_sim = sim(out.unsqueeze(2), con.unsqueeze(2))
            # neg_sim = sim(out.unsqueeze(2), neg_con.unsqueeze(2)) # [8, 256]
            
            # cos_sim = torch.cat([pos_sim, neg_sim], 2)
            # pos_sim = pos_sim.expand(-1, -1, 2)    
            # logger.info(str(pos_sim.shape))
            # logger.info(str(cos_sim.shape))

            # version1: cvae的loss，seq
            cvae_loss = CVAEModel.loss_function(recon_x=out, x=con, mu=mu, logvar=logvar) 
            cvae_loss_neg = CVAEModel.loss_function(recon_x=out, x=neg_con, mu=mu, logvar=logvar)

            # version2: Cos_sim loss
            # print(con.shape)
            # sim_contrast_loss = CVAEModel.loss_function(recon_x=pos_sim, x=neg_sim, mu=mu, logvar=logvar)
            # logger.info(str(sim_contrast_loss))

            sequence_output = self.cvae(sequence_output)
            sequence_output = self.projector(sequence_output)
        else:
            # 训练完毕，使用训练好的编码器
            sequence_output = self.cvae(sequence_output)
            sequence_output = self.projector(sequence_output)
            pass
        
        sequence_output = self.laynorm(sequence_output)
        logits = self.classifier(sequence_output)
        # logger.info('label: ' + str(labels) + str(labels.shape))

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
                # 多任务？
                if do_cvae > 0:
                    # print(loss, cvae_loss, cvae_loss_neg)
                    loss = loss + args.cvae_beta * cvae_loss - args.cvae_theta * cvae_loss_neg
                    # loss = loss + args.cvae_theta * sim_contrast_loss
                    if global_step % args.logging_steps == 0:
                        # fitlog.add_loss(sim_contrast_loss, name='sim loss', step=global_step)
                        fitlog.add_loss(cvae_loss*1000%1000, name = 'positive_loss', step=global_step)
                        fitlog.add_loss(cvae_loss_neg*1000%1000, name = 'negative_loss', step=global_step)
                        fitlog.add_loss(cvae_loss - cvae_loss_neg, name = 'pos - neg', step=global_step)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




