from transformers import BertModel, BertTokenizer
from CVAEMatchModel import GRUEncoder
from CVAEMatchModel import CVaeModel
import torch
import torch.nn as nn
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

Text = 'I feel good today.'

inputs = tokenizer(Text, return_tensors='pt')
outputs = model(**inputs)

input_size = 768
hidden_size = 256
num_layers = 2
dropout = 0.1
seq_len = 7

# GRU_Layer = nn.GRU(input_size=input_size,
#                    hidden_size=hidden_size,
#                    num_layers=num_layers,
#                    bias=True,
#                    batch_first=True,
#                    dropout=dropout,
#                    bidirectional=True)
#
# res, _ = GRU_Layer(outputs[0])
# print(type(res))
# print(res.shape)
#
# res = res.view((res.size(0), seq_len * 2 * hidden_size))
# print(res.shape)
# print(res.size(0), res.size(1))
# fc = nn.Linear(seq_len * 2 * hidden_size, (seq_len * 2 * hidden_size)//2)
# res = fc(res)
# print(res.shape)


model = CVaeModel(input_size=input_size,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  dropout=dropout,
                  seq_len=seq_len)

# print(outputs[0])
print(outputs[0].shape)

res = model(representation=outputs[0])

print(type(res))

print(res.shape)