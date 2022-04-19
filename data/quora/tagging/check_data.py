data_file = 'test_tag.txt'
types = ['reason', 'thing', 'normal', 'approach', 'time', 'choice', 'human', 'location']
with open(data_file, 'r', encoding='utf-8') as reader:
    lines = reader.readlines()
    print(len(lines))
    for line in lines:
        data = line.strip().split('\t')
        info1, info2, label = data
        query1, qtype1 = info1.split('[SEP]')
        query2, qtype2 = info2.split('[SEP]')
        if qtype1 not in types or qtype2 not in types:
            print(line)