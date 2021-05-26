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
model = BertMatchModel.from_pretrained('')
tokenizer = BertTokenizer.from_pretrained('')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

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

for a_data in dataloader:
    a_data = [t.to(device) for t in a_data]
    