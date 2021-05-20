import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--train_file', default='data/MRPC/clean/train_clean.txt')
parser.add_argument('--dev_file', default=None)
parser.add_argument('--test_file', default='data/MRPC/clean/test_clean.txt')
parser.add_argument('--model_type', default='bert-base-uncased')