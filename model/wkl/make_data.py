#制作问题类型分类模型的数据
#通过src和answer预测question对应的问题类型(8类)
#comm_ques_words = ['what','who','how','which','when','where','why','others']
from collections import Counter
def make(data_type):
    """
    :param data_type:
    :return:每条数据对应的问题类型
    """
    comm_ques_words = ['what', 'who', 'how', 'which', 'when', 'where', 'why', 'others']
    tgt_path = "/home/klwu/Code/Python/PGN/data/tgt-"+data_type+".txt"
    result_file = open(data_type+"_label.txt","w",encoding="utf-8")
    ques_words_dict = {}
    for i,word in enumerate(comm_ques_words):
        ques_words_dict[word] = i
    with open(tgt_path, 'r', encoding="utf-8") as f:
        # print(len(f.readlines()))
        for line in f.readlines():
            line = line.strip()
            flag = True
            for key in comm_ques_words:
                if key in line:
                    result_file.write(str(ques_words_dict[key]))
                    result_file.write('\n')
                    flag = False
                    break
            if flag:
                result_file.write(str(ques_words_dict["others"]))
                result_file.write('\n')
    result_file.close()


def count_ques(data_type):
    """
    统计train/dev/test中问题词的分布情况
    :param data_type:
    :return:统计字典
    """
    comm_ques_words = ['what', 'who', 'how', 'which', 'when', 'where', 'why']
    tgt_path = "/home/klwu/Code/Python/PGN/data/tgt-"+data_type+".txt"
    counter = Counter()
    with open(tgt_path,'r',encoding="utf-8") as f:
        # print(len(f.readlines()))
        for line in f.readlines():
            line = line.strip()
            flag = True
            for key in comm_ques_words:
                if key in line:
                    counter[key] += 1
                    flag = False
                    break
            if flag:
                counter["others"] += 1
    return counter

def count(label,path):
    comm_ques_words = ['what','who','how','which','when','where','why','others']
    ques_dict = {}
    result = []
    for i,word in enumerate(comm_ques_words):
        ques_dict[i] = word
    label_file = open(label,'r',encoding="utf-8")
    pred_file = open(path,'r',encoding="utf-8")
    correct = 0
    result_file = open("test_qtype_bert.txt","w",encoding="utf-8")
    for l,p in zip(label_file.readlines(),pred_file.readlines()):
        pred_list = [float(x) for x in p.strip().split("\t")]
        pred = pred_list.index(max(pred_list))
        result_file.write(ques_dict[pred])
        result_file.write("\n")
        result.append(ques_dict[pred])
        # print(pred)
        # print(ques_dict[pred])
        # print(l.strip())
        # exit(0)
        if pred == int(l.strip()):
            correct += 1
    return correct,result



if __name__ == '__main__':
    correct,result = count("test_label.txt","/home/klwu/Code/Python/PGN/bert/output_150/test_results_7000_test.tsv")
    print(correct)
    print(correct/11877)
    print(len(result))
    for k,v in Counter(result).items():
        print(k+":"+str(v/len(result)))
    # for dtype in ["train","dev","test"]:
    #     counter = count_ques(dtype)
    #     print(counter)
    #     print(sum(counter.values()))
        # make(dtype)





















