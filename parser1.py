import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--train_file', default='data/LCQMC/clean/train_clean.txt')
parser.add_argument('--dev_file', default=None)
parser.add_argument('--test_file', default='data/LCQMC/clean/test_clean.txt')
parser.add_argument('--model_type', default='roberta-base')
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--save_dir', default='result/roberta/checkpoints')
parser.add_argument('--do_train', default=True)
parser.add_argument('--do_lower_case', default=True)
# TODO  常改参数
parser.add_argument('--learning_rate', default='2e-5', type=float)
parser.add_argument('--adam_epsilon', default=1e-8, type=float)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--max_length', default=200, type=int)
parser.add_argument('--warmup_steps', default=0.1, type=float)

parser.add_argument('--fp16', action='store_true')
parser.add_argument('--fptype', default='O2')

args = parser.parse_args()

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

args.device = device
