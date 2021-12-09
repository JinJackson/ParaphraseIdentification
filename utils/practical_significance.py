import sys
import numpy as np
from numpy import mean, std
from numpy.core.fromnumeric import std


def read_data_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as reader:
        data_file = []
        try:
            lines = reader.readlines()
            data_file = [float(line.strip()) for line in lines]
        except:
            print('Data format error, please check')
        if len(data_file) == 0:
            print('Empty file, exit')
            sys.exit(0)
    return data_file


def two_side_data_reader(file1_name, file2_name):
    data_file1 = read_data_from_file(file1_name)
    data_file2 = read_data_from_file(file2_name)

    return data_file1, data_file2


def cal_cohen_d(data1, data2):
    def cohen_d(x,y):
        return (mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)

    std1 = np.std(data1)
    std2 = np.std(data2)

    print(mean1, mean2, std1, std2)
    
    return cohen_d(data1, data2)


if __name__ == '__main__':
    file_1 = sys.argv[1]
    file_2 = sys.argv[2]

    data1, data2 = two_side_data_reader(file1, file_2)

    res = cal_cohen_d(data1, data2)

    print(res)