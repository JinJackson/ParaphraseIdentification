from transformers import BertModel, BertPreTrainedModel, BertForMaskedLM, BertTokenizer, BertLMHeadModel
from transformers.modeling_bert import BertOnlyMLMHead
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from all_dataset import TrainData
from attention import attention

class TestModel(BertPreTrainedModel):
    def __init__(self, config):
        super(TestModel, self).__init__(config)
        self.bert = BertModel(config)
        self.input_size = config.hidden_size
        self.GRU_Layer = nn.GRU(input_size=self.input_size,
                                hidden_size=self.input_size//2,
                                num_layers=2,
                                bias=True,
                                batch_first=True,
                                dropout=config.hidden_dropout_prob,
                                bidirectional=True)
        self.cls = BertOnlyMLMHead(config)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_outputs, cls = outputs[:2]
        gru_outputs, _ = self.GRU_Layer(sequence_outputs)
        prediction_scores = self.cls(gru_outputs)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            print(prediction_scores.shape, labels.shape)
            print(prediction_scores.view(-1, self.config.vocab_size).shape, labels.view(-1))
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        output = (prediction_scores,) + outputs[2:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output


model = TestModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = 'I love New [MASK] city.'
labels = tokenizer("I love New York city.", return_tensors="pt")["input_ids"]

inputs = tokenizer(text=text, return_tensors='pt')
outputs = model(**inputs, labels=labels)

# print(len(outputs))
# print(outputs[0].shape, outputs[1].shape)
# print(outputs[0])
# print(outputs[1])



# from torch.utils.data import DataLoader
# text1 = '英雄联盟什么英雄最好？'
# text2 = '英雄联盟最好英雄是什么'
#
# model = BertModel.from_pretrained('bert-base-chinese')
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#
# max_length = 128
# mask_rate = 0.3
# train_data = TrainData(data_file='../data/LCQMC/clean/train_clean.txt',
#                        max_length=max_length,
#                        tokenizer=tokenizer,
#                        model_type='bert-base-chinese')
#
#
# train_dataloader = DataLoader(dataset=train_data,
#                               batch_size=8,
#                               shuffle=True)
#
# for batch in train_dataloader:
#     query1, query2 = batch[-2:]
#     mask_nums = [int(mask_rate * (len(q1) + len(q2))) for q1, q2 in zip(query1, query2)]
#     batch = tuple(t for t in batch[:-2])
#     input_ids, token_type_ids, attention_mask, labels = batch
#     masked_input_ids = input_ids.clone().detach().numpy()
#     # print(len(masked_input_ids))
#     outputs = model(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask)
#     sequence_outputs, cls = outputs[:2]
#
#     # print(sequence_outputs.shape)
#     # print(attention_mask.shape)
#     # print(attention_mask)
#
#     outputs, p_attn = attention(query=sequence_outputs, key=sequence_outputs, value=sequence_outputs, mask=attention_mask)
#
#     # 每个词共获得的注意力大小
#     attn_word = torch.sum(p_attn, dim=1)
#
#     # 按照注意力大小对单词排序
#     sort_idx = torch.argsort(attn_word, dim=1, descending=True).numpy()
#
#     for mask_num, elem, sort_idx in zip(mask_nums, masked_input_ids, sort_idx):
#         for i in range(mask_num):
#             mask_idx = sort_idx[i]
#             elem[mask_idx] = 103


# inputs = tokenizer(text=text1, text_pair=text2, return_tensors='pt')
# print(inputs)
# outputs = model(**inputs)
# sequence_outputs, cls = outputs[:2]
# outputs, p_attn = attention(query=sequence_outputs, key=sequence_outputs, value=sequence_outputs)
#
#
# print(p_attn.shape)
#
# # 每个词共获得的注意力大小
# attn_word = torch.sum(p_attn, dim=1)
# print(attn_word.shape)
# print(attn_word)
#
# # 一共要mask的单词个数
# mask_rate = 0.3
# print('len_attn_word', attn_word.size(1))
# mask_nums = int(mask_rate * attn_word.size(1))
# print(mask_nums)
#
# # 按照注意力大小对单词排序
# sort_idx = torch.argsort(attn_word, dim=1, descending=True)[:, :mask_nums]
# print(sort_idx.shape)
# print(sort_idx)


