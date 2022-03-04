from all_dataset import Multi_task_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
from utils.logger import getLogger
from utils.classification_metrics import accuracy, f1_score
from model.MatchModel import BertMatchModel
from model.VAEMatchModel import VaeBertMatchModel, VaeMultiTaskMatchModel
from transformers import BertTokenizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_file", default=None, type=str, required=True, 
                        help="test file path")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="which model to initalize")
    # parser.add_argument("--model_path", default=None, type=str, required=True, 
    #                     help="ckpt for model to test")
    parser.add_argument("--save_dir", default="bert-base-uncased", type=str, required=True, 
                        help="ckpt for model to test")
    parser.add_argument("--max_length", default=128, type=int, required=True, 
                        help="max_length to be padding or truncation")
    parser.add_argument("--batch_size", default=32, type=int, required=True, 
                        help="batch size to test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    return args
        

def test(model, tokenizer, test_file):
    test_data = Multi_task_dataset(data_file=test_file,
                                    max_length=args.max_length,
                                    tokenizer=tokenizer)

    test_dataLoader = DataLoader(dataset=test_data,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    loss = []

    all_labels = None
    all_logits = None

    model.eval()

    for batch in tqdm(test_dataLoader, desc="Evaluating", ncols=50):
        with torch.no_grad():

            batch = [t.to(args.device) for t in batch]

            input_ids, token_type_ids, attention_mask, labels_main, labels_vice1, labels_vice2 = batch
            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            attention_mask=attention_mask.long(),
                            labels_main=labels_main,
                            labels_vice1=labels_vice1,
                            labels_vice2=labels_vice2)

            eval_loss, logits = outputs[:2]

            loss.append(eval_loss.item())

            if all_labels is None:
                all_labels = labels_main.detach().cpu().numpy()
                all_logits = logits.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, labels_main.detach().cpu().numpy()), axis=0)
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

    acc = accuracy(all_logits, all_labels)
    f1 = f1_score(all_logits, all_labels)
    return np.array(loss).mean(), acc, f1


if __name__ == "__main__":
    args = get_args()
    model_type = args.model_type

    assert model_type in ['cvae', 'baseline', 'vae2task']
    assert args.test_file in ['LCQMC', 'BQ']

    model_class = None

    if model_type == 'cvae':
        model_class = VaeBertMatchModel
    elif model_type == 'vae2task':
        model_class = VaeMultiTaskMatchModel
    elif model_type == 'baseline':
        model_class = BertMatchModel
    

    dev_file = 'data/' + args.test_file + '/clean/dev_clean.txt'
    test_file = 'data/' + args.test_file + '/clean/test_clean.txt'
    model = model_class.from_pretrained(args.save_dir)
    tokenizer = BertTokenizer.from_pretrained(args.save_dir)


    dev_loss, dev_acc, dev_f1 = test(model, tokenizer, dev_file)
    test_loss, test_acc, test_f1 = test(model, tokenizer, test_file)

    print('dev_loss:', dev_loss, ', dev_acc:', dev_acc, ', dev_f1:', dev_f1)
    print('test_loss:', test_loss, ', test_acc:', test_acc, ', test_f1:', test_f1)
    
    
    
