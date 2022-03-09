from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.classification_metrics import accuracy, f1_score
from tqdm import tqdm
import torch

from transformers import BertTokenizer
from model.VAEMatchModel import VaeBertMatchModel
from model.MatchModel import BertMatchModel
import sys
import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_data', default='LCQMC')
    parser.add_argument('--model_type', default='vae') #vae baseline multi-task
    parser.add_argument('--model_path', default='bert-base-chinese')
    parser.add_argument('--test_set', default='all')

    args_test = parser.parse_args()
    return args_test

args_test = get_args()


key_dict = {'事物': 'what', '人物': 'who', '做法': 'how', '选择': 'which', '时间': 'when', '地点': 'where', '原因': 'why',
            '其他': 'others', '未知': 'UNK'}
q_type = ['事物', '人物', '做法', '选择', '时间', '地点', '原因',
          '其他', '未知']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def read_data_5lines(data_file):
    all_data = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            data = line.strip().split('\t')
            qtype1, query1, qtype2, query2, label = data
            all_data.append([query1+'[SEP]'+qtype1, query2+'[SEP]'+qtype2, label])
    return all_data


def read_data_3lines(data_file):
    all_data = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            data = line.strip().split('\t')
            qtype1, query1, qtype2, query2, label = data
            all_data.append([query1, query2, label])
    return all_data


class vae_Dataset(Dataset):
    def __init__(self, data_file, max_length, tokenizer):
        self.datas = read_data_5lines(data_file)
        self.max_length = max_length
        self.tokenizer = tokenizer
    
    def __getitem__(self, item):
        data = self.datas[item]
        query1, query2, label = data[0], data[1], int(data[2])
        tokenzied_dict = self.tokenizer.encode_plus(text=query1,
                                                    text_pair=query2,
                                                    max_length=self.max_length,
                                                    truncation=True,
                                                    padding='max_length')
        input_ids  = np.array(tokenzied_dict['input_ids'])
        attention_mask = np.array(tokenzied_dict['attention_mask'])
        token_type_ids = np.array(tokenzied_dict['token_type_ids'])
        return input_ids, token_type_ids, attention_mask, np.array([label]), query1, query2
    
    def __len__(self):
        return len(self.datas)


class norm_Dataset(Dataset):
    def __init__(self, data_file, max_length, tokenizer):
        self.datas = read_data_3lines(data_file)
        self.max_length = max_length
        self.tokenizer = tokenizer
    
    def __getitem__(self, item):
        data = self.datas[item]
        query1, query2, label = data[0], data[1], int(data[2])
        tokenzied_dict = self.tokenizer.encode_plus(text=query1,
                                                    text_pair=query2,
                                                    max_length=self.max_length,
                                                    truncation=True,
                                                    padding='max_length')
        input_ids  = np.array(tokenzied_dict['input_ids'])
        attention_mask = np.array(tokenzied_dict['attention_mask'])
        token_type_ids = np.array(tokenzied_dict['token_type_ids'])
        return input_ids, token_type_ids, attention_mask, np.array([label]), query1, query2
    
    def __len__(self):
        return len(self.datas)


def load_all_datasets(tokenizer, test_type):
    q_type = ['事物', '人物', '做法', '选择', '时间', '地点', '原因',
          '其他', '未知']

    key_dict = {'事物': 'what', '人物': 'who', '做法': 'how', '选择': 'which', '时间': 'when', '地点': 'where', '原因': 'why',
            '其他': 'others', '未知': 'UNK'}

    max_length = 128

    all_dataset = dict()
    dataset_docker = None
    print('test_type', test_type)

    all_test_type = ['vae', 'baseline']
    if test_type == 'vae':
        dataset_docker = vae_Dataset
    elif test_type == 'baseline':
        dataset_docker = norm_Dataset
    elif test_type not in all_test_type:
        print('test_type illegal')
        sys.exit(0)

    for type in q_type:
        data_file = './data/LCQMC/split/' + key_dict[type] + '.txt'
        all_dataset[type] = dataset_docker(data_file=data_file,
                                            max_length=max_length, 
                                            tokenizer=tokenizer)
    return all_dataset
    

def test_for_dataset(dataset, model, model_type):
    data_loader = DataLoader(dataset=dataset, 
                            batch_size=16, 
                            shuffle=True)
    all_labels = None
    all_logits = None
    loss = []
    for batch in tqdm(data_loader, desc='testing'):
        input_ids, token_type_ids, attention_mask, labels, query1, query2 = batch
        batch = [t.to(device) for t in batch[:-2]]
        input_ids, token_type_ids, attention_mask, labels = batch

        if model_type == 'vae':
            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            attention_mask=attention_mask.long(),
                            query1=query1,
                            query2=query2,
                            mask_rate=None,
                            labels=labels)

            eval_loss, logits = outputs[:2]
        
        elif model_type == 'baseline':
            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            attention_mask=attention_mask.long(),
                            labels=labels)

            eval_loss, logits = outputs[:2]

        loss.append(eval_loss.item())

        if all_labels is None:
            all_labels = labels.detach().cpu().numpy()
            all_logits = logits.detach().cpu().numpy()
        else:
            all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()), axis=0)
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

    acc = accuracy(all_logits, all_labels)
    f1 = f1_score(all_logits, all_labels)
    return np.array(loss).mean(), acc, f1


def Test():
    model_type = args_test.model_type
    model_class = None
    model_path = args_test.model_path
    test_set = args_test.test_set

    # model_type to get Model class file
    if model_type == 'vae':
        model_class = VaeBertMatchModel
    elif model_type == 'baseline':
        model_class = BertMatchModel
    else:
        print('please choose model_type from:1.baseline 2.vae 3.multi-task')
        sys.exit(0)
    
    # load model and tokenizer from args.model_path
    tokenzier = BertTokenizer.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path).to(device)
    all_dataset = load_all_datasets(tokenizer=tokenzier, 
                                    test_type=model_type)
    
    # choose test sub set: "all" to test all sub set
    if test_set == 'all':
        for key in all_dataset:
            # print(key) 
            key_dataset = all_dataset[key]

            loss, acc, f1 = test_for_dataset(dataset=key_dataset,
                                            model=model,
                                            model_type=model_type)
            print("dataset_%s, acc:%.2f, f1:%.2f, loss:%.2f." % (key, acc, f1, loss))
    elif test_set not in all_dataset.keys():
        print('test_set type not exists')
        sys.exit(0)
    else:
        for key in all_dataset:
            if key == test_set:
                key_dataset = all_dataset[key]

                loss, acc, f1 = test_for_dataset(dataset=key_dataset,
                                            model=model)
                print(key, loss, acc, f1)
                break

            
if __name__ == "__main__":
    Test()