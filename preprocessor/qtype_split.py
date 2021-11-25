data_file = r'D:\Dev\pythonspace\pycharmWorkspace\NLPtasks\ParaphraseIdentification\data\LCQMC\tagging\test_tag.txt'


def read_data(data_file):
    all_data = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            query1, query2, label = line.strip().split('\t')
            query1, q_type1 = query1.strip().split('[SEP]')
            query2, q_type2 = query2.strip().split('[SEP]')
            all_data.append([(q_type1, query1), (q_type2, query2), label])
    return all_data


def classify(data_file):
    all_data = read_data(data_file)

    counter = dict()
    for info in all_data:
        info1, info2, label = info
        qtype1, query1 = info1
        qtype2, query2 = info2

        if len(qtype1) > 2:
            continue
        else:
            if counter.get(qtype1, -1) == -1:
                counter[qtype1] = []
            else:
                counter[qtype1].append((qtype1, query1, qtype2, query2, label))
    return counter


counter = classify(data_file)
key_dict = {'事物': 'what', '人物': 'who', '做法': 'how', '选择': 'which', '时间': 'when', '地点': 'where', '原因': 'why',
            '其他': 'others', '未知': 'UNK'}

for key in key_dict:
    written_file = '../data/LCQMC/split/' + key_dict[key] + '.txt'
    with open(written_file, 'w', encoding='utf-8') as writer:
        for data in counter[key]:
            writer.write('\t'.join(data) + '\n')


