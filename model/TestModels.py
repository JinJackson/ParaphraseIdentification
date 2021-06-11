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



