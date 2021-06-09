data_file = '../data/LCQMC/clean/train_clean.txt'
translate_file = '../data/LCQMC/translation/train_en.txt'

from tqdm import tqdm

def readDataFromFile(data_file):
    datas = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            pair = line.strip().split('\t')
            datas.append(pair)
    return datas


def readTranslateData(data_file):
    datas = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            a_data = line.strip().split('||')
            # split bugs
            if len(a_data) == 2:
                line = line.replace('||', '|')
                a_data = line.strip().split('|')
            # delete ' '
            if len(a_data) == 4:
                for i in range(len(a_data)):
                    if not a_data[i] or a_data[i] == ' ':
                        a_data.pop(i)
                        break
            if len(a_data) != 3:
                # datas.append(a_data)
                datas.append(['UNK', 'UNK', 'UNK', 'UNK'])
            else:
                datas.append(a_data)
    return datas


# def markQtype(sentence)
all_datas = readDataFromFile(data_file)
all_trans = readTranslateData(translate_file)

print(len(all_datas), len((all_trans)))

# tagging data
all_taggings = []
for origin_data, trans_data in tqdm(zip(all_datas, all_trans), total=len(all_datas)):
    origin_query1, origin_query2, label = origin_data
    if len(trans_data) == 4:
        all_taggings.append([origin_query1, origin_query2, label, 'UNK', 'UNK'])
    origin_query1, origin_query2, label = origin_data
