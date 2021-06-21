from model.MatchModel import ErnieMatchModel
from transformers import AutoTokenizer
import torch

model = ErnieMatchModel.from_pretrained("nghuyong/ernie-1.0")
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")

#print(model)

text = '今天的天气真不错'
label = torch.tensor([[1]])

inputs = tokenizer.encode_plus(text=text, return_tensors='pt')

input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']
attention_mask = inputs['attention_mask']

outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=label)

loss, logits = outputs

print(loss)
print(logits)