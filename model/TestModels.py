from model.MatchModel import ErnieMatchModel
from transformers import AutoTokenizer

model = ErnieMatchModel.from_pretrained("nghuyong/ernie-1.0")
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")

# print(model)