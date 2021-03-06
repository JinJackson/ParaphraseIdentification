from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np


def readDataFromFile(data_file):
    datas = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            pair = line.strip().split('\t')
            datas.append(pair)
    return datas


class TrainData(Dataset):
    def __init__(self, data_file, max_length, tokenizer, model_type='bert'):
        self.datas = readDataFromFile(data_file)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __getitem__(self, item):
        data = self.datas[item]
        query1, query2, label = data[0], data[1], int(data[2])
        if 'roberta' in self.model_type:
            query1 = query1.replace('[SEP]', '</s>')
            query2 = query2.replace('[SEP]', '</s>')
        tokenzied_dict = self.tokenizer.encode_plus(text=query1,
                                                    text_pair=query2,
                                                    max_length=self.max_length,
                                                    truncation=True,
                                                    padding='max_length')
        input_ids  = np.array(tokenzied_dict['input_ids'])
        attention_mask = np.array(tokenzied_dict['attention_mask'])
        
        if 'roberta' in self.model_type:
            return input_ids, attention_mask, np.array([label]), query1, query2
        else:
            token_type_ids = np.array(tokenzied_dict['token_type_ids'])
            return input_ids, token_type_ids, attention_mask, np.array([label]), query1, query2
        
        

    def __len__(self):
        return len(self.datas)


class Multi_task_dataset(Dataset):
    def __init__(self, data_file, max_length, tokenizer, model_type):
        self.datas = readDataFromFile(data_file)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.qtype_dict = {'??????': 0, '??????': 1, '??????': 2, '??????': 3, '??????': 4, '??????': 5, '??????': 6, '??????': 7}
    def __getitem__(self, item):
        data = self.datas[item]
        info1, info2, label_main = data[0], data[1], int(data[2])
        query1, qtype1 = info1.split('[SEP]')
        query2, qtype2 = info2.split('[SEP]')
        label_vice1 = self.qtype_dict[qtype1[:2]]
        label_vice2 = self.qtype_dict[qtype2[:2]]
        if 'roberta' in self.model_type:
            query1 = query1.replace('[SEP]', '</s>')
            query2 = query2.replace('[SEP]', '</s>')
        tokenzied_dict = self.tokenizer.encode_plus(text=query1,
                                                    text_pair=query2,
                                                    max_length=self.max_length,
                                                    truncation=True,
                                                    padding='max_length')
        input_ids  = np.array(tokenzied_dict['input_ids'])
        attention_mask = np.array(tokenzied_dict['attention_mask'])
        
        if 'roberta' in self.model_type:
            return input_ids, attention_mask, np.array([label_main]), np.array([label_vice1]), np.array([label_vice2])
        else:
            token_type_ids = np.array(tokenzied_dict['token_type_ids'])
            return input_ids, token_type_ids, attention_mask, np.array([label_main]), np.array([label_vice1]), np.array([label_vice2])

        
    
    def __len__(self):
        return len(self.datas)



class Multi_task_dataset_eng(Dataset):
    def __init__(self, data_file, max_length, tokenizer, model_type):
        self.datas = readDataFromFile(data_file)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.qtype_dict = {'thing': 0, 'human': 1, 'approach': 2, 'choice': 3, 'time': 4, 'location': 5, 'reason': 6, 'normal': 7}
    def __getitem__(self, item):
        data = self.datas[item]
        info1, info2, label_main = data[0], data[1], int(data[2])
        query1, qtype1 = info1.split('[SEP]')
        query2, qtype2 = info2.split('[SEP]')
        label_vice1 = self.qtype_dict[qtype1]
        label_vice2 = self.qtype_dict[qtype2]
        if 'roberta' in self.model_type:
            query1 = query1.replace('[SEP]', '</s>')
            query2 = query2.replace('[SEP]', '</s>')
        tokenzied_dict = self.tokenizer.encode_plus(text=query1,
                                                    text_pair=query2,
                                                    max_length=self.max_length,
                                                    truncation=True,
                                                    padding='max_length')
        input_ids  = np.array(tokenzied_dict['input_ids'])
        attention_mask = np.array(tokenzied_dict['attention_mask'])
        
        if 'roberta' in self.model_type:
            return input_ids, attention_mask, np.array([label_main]), np.array([label_vice1]), np.array([label_vice2])
        else:
            token_type_ids = np.array(tokenzied_dict['token_type_ids'])
            return input_ids, token_type_ids, attention_mask, np.array([label_main]), np.array([label_vice1]), np.array([label_vice2])

        
    
    def __len__(self):
        return len(self.datas)


# if __name__ == '__main__':
#     from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
#     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#     # model = RobertaModel.from_pretrained('bert-base-uncased')
#     train_dataset = TrainData(data_file='./data/quora/tagging/train_tag.txt', max_length=100, tokenizer=tokenizer, model_type='roberta')
#     TrainDataLoader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)
#     for batch in TrainDataLoader:
#         input_ids, attention_mask, label, q1, q2 = batch
#         print(q1, q2)
#         print(input_ids)
#         break
        # outputs = model(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask.long(), return_dict=True)
        # print(type(outputs))
        # res = outputs.pooler_output
        # res2 = outputs[1]
        # print(res.shape)
        # print(res == res2)
        # break

