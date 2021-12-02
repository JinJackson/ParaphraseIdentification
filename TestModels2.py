from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from model.VAEMatchModel import VaeBertMatchModel, VaeMultiTaskMatchModel
from all_dataset import TrainData, Multi_task_dataset
from parser1 import args
import torch

# from attention import attention

device = 'cuda' if torch.cuda.is_available() else 'cpu'

text1 = '英雄联盟什么英雄最好？'
text2 = '英雄联盟最好英雄是什么'

model = VaeMultiTaskMatchModel.from_pretrained('bert-base-chinese')
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

max_length = 128
learning_rate = 1e-5
adam_epsilon = 1e-8

train_data = Multi_task_dataset(data_file='data/LCQMC/tagging/test_tag.txt',
                                max_length=max_length,
                                tokenizer=tokenizer)


train_dataloader = DataLoader(dataset=train_data,
                              batch_size=4,
                              shuffle=True)

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)

for batch in train_dataloader:
    batch = tuple(t.to(device) for t in batch)
    input_ids, token_type_ids, attention_mask, labels_main, labels_vice1, labels_vice2 = batch

    # print(len(masked_input_ids))
    outputs = model(input_ids=input_ids.long(),
                    token_type_ids=token_type_ids.long(),
                    attention_mask=attention_mask,
                    labels_main=labels_main,
                    labels_vice1=labels_vice1,
                    labels_vice2=labels_vice2)

    loss, logits = outputs
    # print(loss.shape, logits.shape)
    break

