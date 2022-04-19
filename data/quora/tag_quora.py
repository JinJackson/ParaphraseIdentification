data_file = './clean/test_clean.txt'
written_file = './tagging/test_tag.txt'
import re
from collections import OrderedDict
all_pattern = OrderedDict()

all_pattern['reason'] = ['why .* ', 'what .* reason']
all_pattern['location'] = ['where', 'what .* place']
all_pattern['approach'] = ['how', 'what .* way']
all_pattern['human'] = ['who', 'what .* person', 'what .* people']
all_pattern['choice'] = ['which']
all_pattern['time'] = ['when', 'what .* time', 'what .* date', 'what date']
all_pattern['thing'] = ['what']
# all_pattern['normal'] = ['is there .*', 'can i .*', 'can you .*']
# all_pattern['unk'] = []

all_data = []
badcase = []
count = 0
with open(data_file, 'r', encoding='utf-8') as reader:
    lines = reader.readlines()
    print(len(lines))
    for line in lines:
        a_data = line.strip().split('\t')
        query1 = a_data[0]
        query2 = a_data[1]
        label = a_data[2]
        # print(query1, query2, label)
        flag1 = False
        flag2 = False
        qtype1 = None
        qtype2 = None
        flag = False
        for key in all_pattern:
            if flag == True:
                break
            for p in all_pattern[key]:
                pattern = re.compile(p, re.I)
                if not flag1:
                    match_1 = pattern.search(query1)
                    if match_1:
                        flag1 = True
                        qtype1 = key
                if not flag2:
                    match_2 = pattern.search(query2)
                    if match_2:
                        flag2 = True
                        qtype2 = key

                if flag1 and flag2:
                    all_data.append([query1, qtype1, query2, qtype2, label])
                    flag = True
                    break
                

        if flag == False:
            count += 1
            if qtype1 == None:
                qtype1 = 'normal'
            if qtype2 == None:
                qtype2 = 'normal'
            all_data.append([query1, qtype1, query2, qtype2, label])
            badcase.append([query1, qtype1, query2, qtype2, label])


with open(written_file, 'w', encoding='utf-8') as writer:
    for a_data in all_data:
        query1, qtype1, query2, qtype2, label = a_data
        sent1 = query1 + '[SEP]' + qtype1
        sent2 = query2 + '[SEP]' + qtype2
        writer.write('\t'.join([sent1, sent2, label]) + '\n')
