from torch.utils.data import Dataset, DataLoader
import numpy as np

key_dict = {'事物': 'what', '人物': 'who', '做法': 'how', '选择': 'which', '时间': 'when', '地点': 'where', '原因': 'why',
            '其他': 'others', '未知': 'UNK'}
q_type = ['事物', '人物', '做法', '选择', '时间', '地点', '原因',
          '其他', '未知']


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


def load_all_datasets(model, tokenizer, test_type):
    q_type = ['事物', '人物', '做法', '选择', '时间', '地点', '原因',
          '其他', '未知']

    key_dict = {'事物': 'what', '人物': 'who', '做法': 'how', '选择': 'which', '时间': 'when', '地点': 'where', '原因': 'why',
            '其他': 'others', '未知': 'UNK'}

    max_length = 128

    all_dataset = dict()

    if test_type == 'vae':
        dataset_docker = vae_Dataset
    for type in q_type:
        data_file = './data/LCQMC/split/' + key_dict[type] + '.txt'
        all_dataset[type] = dataset_docker(data_file=data_file, max_length=max_length, tokenizer=tokenizer)
    return all_dataset
    

if __name__ == "__main__":
    model = None
    from transformers import BertTokenizer
    tokenzier = BertTokenizer.from_pretrained('bert-base-chinese')
    all_dataset = load_all_datasets(model=model, tokenizer=tokenzier, test_type='vae')
    for key in all_dataset:
        print(key) 
        key_dataset = all_dataset[key]
        data_loader = DataLoader(dataset=key_dataset, batch_size=2, shuffle=True)
        for batch in data_loader:
            input_ids, token_type_ids, attention_mask, label, query1, query2 = batch
            print(query1, query2, input_ids)
            break
        break