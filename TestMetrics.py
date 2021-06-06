from model.MatchModel import BertMatchModel
from tqdm import tqdm
from all_dataset import TrainData
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import torch
from parser1 import args
from utils.classification_metrics import accuracy, f1_score
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertMatchModel.from_pretrained('bert-base-chinese')
train_dataset = TrainData(data_file='data/LCQMC/clean/test_clean2.txt', max_length=args.max_length, tokenizer=tokenizer, model_type='bert')
TrainDataLoader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss = []
all_labels = None
all_logits = None
model.to(device)
model.eval()

count = 0
with torch.no_grad():
    for batch in tqdm(TrainDataLoader, desc='Iteration'):
        batch = [t.to(device) for t in batch[:-2]]
        input_ids, token_type_ids, attention_mask, labels = batch
        outputs = model(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask, labels=labels)
        test_loss, logits = outputs

        loss.append(test_loss.item())

        if all_labels is None:
            all_labels = labels.detach().cpu().numpy()
            all_logits = logits.detach().cpu().numpy()
        else:
            all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()), axis=0)
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

    acc = accuracy(all_logits, all_labels)
    f1 = f1_score(all_logits, all_labels)
    print(acc, f1)