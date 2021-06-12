# from transformers import BertModel, BertTokenizer
# from model.VAEMatchModel import VaeBertMatchModel
# import torch
# import torch.nn as nn
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# Text = 'I feel good today.'
#
# inputs = tokenizer(Text, max_length=128, padding='max_length', return_tensors='pt')
# labels = torch.tensor([[1]]).float()
# # inputs.to(device)
# model = VaeBertMatchModel.from_pretrained('bert-base-uncased')
# # model.to(device)
#
# loss, logits = model(**inputs, labels=labels)
#
# # loss.backward()
#
#
# print(loss, logits)
#
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# text = '这是什么蔬菜啊[SEP]事物	这是什么蔬菜呀？[SEP]事物	1'
# data = text.strip().split('\t')
# query1, query2, label = data[0], data[1], int(data[2])
# tokenzied_dict = tokenizer.encode_plus(text=query1,
#                                        text_pair=query2,
#                                        max_length=128,
#                                        truncation=True,
#                                        padding='max_length')
# input_ids = tokenzied_dict['input_ids']
# token_type_ids = tokenzied_dict['token_type_ids']
# attention_mask = tokenzied_dict['attention_mask']
# print(input_ids)
# print(token_type_ids)
# print(token_type_ids.index(1))
# print(attention_mask)

datas = []
with open('../data/BQ/clean/train_clean.txt', 'r', encoding='utf-8') as reader:
    lines = reader.readlines()
    for line in lines:
        pair = line.strip().split('\t')
        datas.append(pair)
        print(pair)

for data in datas:
    if len(data) != 3:
        print(data)