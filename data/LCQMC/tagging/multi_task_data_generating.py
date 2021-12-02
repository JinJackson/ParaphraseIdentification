
data_file = ''
q_type_dict = {'事物': 0, '人物': 1, '做法': 2, '选择': 3, '时间': 4, '地点': 5, '原因': 6,
          '其他': 7, '未知': 8}

all_datas = []
with open(data_file, 'r', encoding='utf-8') as reader:
    for lines in reader.readlines():
        for line in lines:
            data = line.strip().split('\t')
            info1, info2, label = data
            query1, qtype1 = info1.split('[SEP]')
            query2, qtype2 = info2.split('[SEP]')

