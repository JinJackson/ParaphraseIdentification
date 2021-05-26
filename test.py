from model.MatchModel import BertMatchModel
from tqdm import tqdm
from dataset.dataset import TrainData
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertMatchModel.from_pretrained('bert-base-uncased')
train_dataset = TrainData(data_file='data/MRPC/clean/test_clean.txt', max_length=100, tokenizer=tokenizer, model_type='bert')
TrainDataLoader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss = []
all_labels = None
all_logits = None
model.to(device)
model.eval()

count = 0
with torch.no_grad():
    for batch in tqdm(TrainDataLoader, desc='Iteration'):
        batch = [t.to(device) for t in batch]
        input_ids, token_type_ids, attention_mask, labels = batch
        outputs = model(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask, labels=labels)
        test_loss, logits = outputs
        #print(logits)
        #print(logits.item())
        if logits.item() > 0:
            count += 1
        loss.append(test_loss.item())

        if all_labels is None:
            all_labels = labels.detach().cpu().numpy()
            all_logits = logits.detach().cpu().numpy()
        else:
            all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()), axis=0)
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
all_predict = (all_logits > 0) + 0
results = (all_predict == all_labels)
acc = results.sum() / len(all_predict)