from model.MatchModel import BertMatchModel, AlbertMatchModel, RobertaMatchModel
from transformers import BertTokenizer
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from all_dataset import TrainData
import torch

# params
test_file = '../data/LCQMC/clean/test_clean.txt'
max_length = 128
model_type = 'bert-base-chinese'


# load models
model = BertMatchModel.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model_type = 'bert'
# loss_function
loss_func = BCEWithLogitsLoss()

# load data
dataset = TrainData(data_file=test_file,
                    max_length=max_length,
                    tokenizer=tokenizer,
                    model_type=model_type)

dataloader = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle=False)

wrong_case = []
for a_data in dataloader:
    query1, query2 = a_data[-2], a_data[-1]
    a_data = [t for t in a_data[:-2]]
    if 'roberta' in model_type:
        input_ids, attention_mask, label = a_data
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        loss, logits = model(input_ids=input_ids, attention_mask=attention_mask,
                             labels=label)
    else:
        input_ids, token_type_ids, attention_mask, label = a_data
        input_ids = input_ids.long()
        token_type_ids = token_type_ids.long()
        attention_mask = attention_mask.long()
        loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                             labels=label)
    if (logits.item() * label.item()) < 0:
        #print(query1, query2, label.item())
        wrong_case.append([query1, query2, label.item(), logits.item()])