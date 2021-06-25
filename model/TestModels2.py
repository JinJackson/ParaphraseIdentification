from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from model.VAEMatchModel import VaeBertMatchModel
from all_dataset import TrainData
from parser1 import args
import torch
from attention import attention

text1 = '英雄联盟什么英雄最好？'
text2 = '英雄联盟最好英雄是什么'

model = VaeBertMatchModel.from_pretrained('bert-base-chinese')
model = model.to('cuda')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

max_length = 128
mask_rate = 0.3
mask_rate = 0.1
train_data = TrainData(data_file='../data/LCQMC/clean/train_clean.txt',
                       max_length=max_length,
                       tokenizer=tokenizer,
                       model_type='bert-base-chinese')


train_dataloader = DataLoader(dataset=train_data,
                              batch_size=8,
                              shuffle=True)

optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

for batch in train_dataloader:
    query1, query2 = batch[-2:]
    batch = tuple(t.to('cuda') for t in batch[:-2])
    input_ids, token_type_ids, attention_mask, labels = batch

    # print(len(masked_input_ids))
    outputs = model(input_ids=input_ids.long(),
                    token_type_ids=token_type_ids.long(),
                    attention_mask=attention_mask,
                    query1=query1,
                    query2=query2,
                    mask_rate=0.1,
                    labels=labels)

    loss, logits = outputs
    print(loss.shape, logits.shape)
    break

