# from transformers import BertModel, BertTokenizer
# from model.CVAEMatchModel import GRUEncoder
# from model.CVAEMatchModel import CVaeModel, CVaeBertMatchModel
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
# model = CVaeBertMatchModel.from_pretrained('bert-base-uncased')
# # model.to(device)
#
# loss, logits = model(**inputs, labels=labels)
#
# loss.backward()
#
#
# print(loss, logits)

import torch

t1 = torch.randn((4, 20))
print(t1.shape)
linear = torch.nn.Linear(20, 1)
print(linear(t1))
