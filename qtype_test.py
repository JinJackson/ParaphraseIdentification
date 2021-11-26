from torch.utils.data import Dataset, DataLoader

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


data_file = 'data/LCQMC/split/what.txt'
all_data = read_data_5lines(data_file)
print(all_data)

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
        attention_mask = np.array(tokenzied_dict['attention_mask'])S
        token_type_ids = np.array(tokenzied_dict['token_type_ids'])
        return input_ids, token_type_ids, attention_mask, np.array([label]), query1, query2



# def vae_Dataset():
#     data_collections = dict()
#
#     for key in q_type:
#         data_file = "../data/LCQMC/split/" + key_dict[key] + ".txt"
#         with open(data_file, 'r', encoding='utf-8') as reader:
#             lines = reader.readlines()
#             for line in lines:
#                 if data_collections.get(key, -1) == -1:
#                     data_collections[key] = []
#                 else:
#                     data_collections[key].append()
