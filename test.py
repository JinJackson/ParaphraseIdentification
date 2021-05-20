from model.MatchModel import BertMatchModel

from dataset.MRPCdataset import TrainData
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertMatchModel.from_pretrained('bert-base-uncased')
train_dataset = TrainData(data_file='data/MRPC/clean/train_clean.txt', max_length=100, tokenizer=tokenizer, model_type='bert')
TrainDataLoader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
for batch in TrainDataLoader:
    input_ids, token_type_ids, attention_mask, label = batch
    outputs = model(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask, label=label)
    print(outputs)
    break