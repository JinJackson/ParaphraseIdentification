import json

file_name = '../data/LCQMC/origin/test.json'
datas = []
with open(file_name, 'r', encoding='utf-8') as reader:
    lines = reader.readlines()
    for line in lines:
        a_data = json.loads(line)
        query1 = a_data['sentence1'].strip()
        query2 = a_data['sentence2'].strip()
        label = a_data['gold_label'].strip()
        datas.append([query1, query2, str(label)])

written_file = '../data/LCQMC/clean/test_clean.txt'
with open(written_file, 'w', encoding='utf-8') as writer:
    for a_data in datas:
        query1, query2, label = a_data
        writer.write(query1 + '\t' + query2 + '\t' + label + '\n')