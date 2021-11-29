data_file = '/Volumes/FileSystem/Devs/Codes/ParaphraseIdentification/data/LCQMC/tagging/test_tag.txt'

counter = dict()
with open(data_file, 'r', encoding='utf-8') as reader:
    lines = reader.readlines()
    for line in lines:
        data = line.strip().split('\t')
        info1, info2, label = data
        query1, qtype1 = info1.split('[SEP]')
        query2, qtype2 = info2.split('[SEP]')
        counter[qtype1] = counter.get(qtype1, 0) + 1
        counter[qtype2] = counter.get(qtype2, 0) + 1


print(counter)

count = 0
for key in counter:
    if len(key) >5:
        count += counter[key]
print(count)
        