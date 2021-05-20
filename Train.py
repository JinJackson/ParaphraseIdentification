from parser1 import args
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from model.MatchModel import BertMatchModel, RobertaMatachModel, AlbertMatchModel
import os, random
import glob
import torch

import numpy as np
from tqdm import tqdm

from dataset.MRPCdataset import TrainData
from utils.logger import getLogger
from utils.metrics import acc

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


if args.seed > -1:
  seed_torch(args.seed)

logger = None

def train(model, tokenizer, checkpoint):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None

    train_data = TrainData(data_file=args.train_file,
                           max_length=200,
                           tokenizer=tokenizer,
                           model_type=args.model_type)

    train_dataLoader = DataLoader(dataset=train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    t_total = len(train_dataLoader) * args.epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_eplison)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fptype)

    # 读取断点 optimizer、scheduler
    checkpoint_dir = args.save_dir + "/checkpoint-" + str(checkpoint)
    if os.path.isfile(os.path.join(checkpoint_dir, "optimizer.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))
        if args.fp16:
            amp.load_state_dict(torch.load(os.path.join(checkpoint_dir, "amp.pt")))

    # 开始训练
    logger.debug("***** Running training *****")
    logger.debug("  Num examples = %d", len(train_dataLoader))
    logger.debug("  Num Epochs = %d", args.epochs)
    logger.debug("  Batch size = %d", args.batch_size)

    # 没有历史断点，则从0开始
    if checkpoint < 0:
        checkpoint = 0
    else:
        checkpoint += 1

    logger.debug("  Start Batch = %d", checkpoint)
    for epoch in range(checkpoint, args.epochs):
        model.train()
        epoch_loss = []

        for batch in tqdm(train_dataLoader, desc="Iteration"):
            model.zero_grad()
            # 设置tensor gpu运行
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, token_type_ids, attention_mask, labels = batch

            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            labels=labels)

            loss = outputs[0]

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()  # 计算出梯度

            outputs_adv = model(input_ids=input_ids.long(),
                                token_type_ids=token_type_ids.long(),
                                labels=labels)
            loss_adv = outputs_adv[0]
            # print(loss.item(), loss_adv.item())
            if args.fp16:
                with amp.scale_loss(loss_adv, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_adv.backward()

            epoch_loss.append(loss.item() + loss_adv.item())

            optimizer.step()
            scheduler.step()

            # 保存模型
        output_dir = args.save_dir + "/checkpoint-" + str(epoch)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.debug("Saving model checkpoint to %s", output_dir)
        if args.fp16:
            torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.debug("Saving optimizer and scheduler states to %s", output_dir)


def test(model, tokenizer, test_file, checkpoint, output_dir=None):
    test_data = TrainData(data_file=test_file,
                          max_length=200,
                          tokenizer=tokenizer)

    test_dataLoader = DataLoader(dataset=test_data,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    logger.debug("***** Running test {} *****".format(checkpoint))
    logger.debug("  Num examples = %d", len(test_dataLoader))
    logger.debug("  Batch size = %d", args.batch_size)

    loss = []

    all_labels = None
    all_logits = None

    model.eval()

    for batch in tqdm(test_dataLoader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, token_type_ids, attention_mask, labels = batch

        with torch.no_grad():
            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            labels=labels)

            eval_loss, logits = outputs[:2]

            loss.append(eval_loss.item())

            if all_labels is None:
                all_labels = labels.detach().cpu().numpy()
                all_logits = logits.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()), axis=0)
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

    all_predict = (all_logits > 0) + 0
    results = (all_predict == all_labels)
    acc = results.sum() / len(all_predict)
    return acc

    return np.array(loss).mean(), res_acc
