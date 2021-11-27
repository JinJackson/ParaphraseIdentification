import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--test_data', default='LCQMC')
parser.add_argument('--model_type', default='vae') #vae baseline multi-task
parser.add_argument('--model_path', default='bert-base-chinese')
parser.add_argument('--test_set', default='all')

args_test = parser.parse_args()