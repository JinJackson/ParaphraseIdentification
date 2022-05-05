from parser1 import args
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, AlbertTokenizer, AutoTokenizer
from model.MatchModel import BertMatchModel, RobertaMatchModel, AlbertMatchModel
from model.VAEMatchModel import VaeMultiTaskMatchModelClean, VaeMultiTaskMatchRobertaModelClean
import os, random
import glob
import torch

import numpy as np
from tqdm import tqdm

from all_dataset import Multi_task_dataset
from utils.logger import getLogger
from utils.classification_metrics import accuracy, f1_score


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

def train(model, tokenizer, checkpoint, round):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None

    train_data = Multi_task_dataset(data_file=args.train_file,
                                    max_length=args.max_length,
                                    tokenizer=tokenizer,
                                    model_type=args.model_type)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    t_total = len(train_dataloader) * args.epochs
    warmup_steps = int(args.warmup_steps * t_total)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fptype)

    # 读取断点 optimizer、scheduler
    checkpoint_dir = args.save_dir + "/checkpoint-" + str(checkpoint) + '-' + str(round)
    if os.path.isfile(os.path.join(checkpoint_dir, "optimizer.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))
        if args.fp16:
            amp.load_state_dict(torch.load(os.path.join(checkpoint_dir, "amp.pt")))

    # 开始训练
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  learning_rate = %s", str(args.learning_rate))
    logger.info("  Total steps = %d", t_total)
    logger.info("  warmup steps = %d", warmup_steps)
    logger.info("  Model_type = %s", args.model_type)
    logger.info("  Decoder_type = %s", args.decoder_type)
    logger.info("  vice_loss_weight = %s", str(args.vice_weight))


    # 没有历史断点，则从0开始
    if checkpoint < 0:
        checkpoint = 0
        round = 0
    else:
        checkpoint += 1
        round += 1

    max_test_acc = 0
    max_test_f1 = 0

    logger.debug("  Start Batch = %d", checkpoint)
    for epoch in range(checkpoint, args.epochs):
        model.train()
        epoch_loss = []

        step = 0
        for batch in tqdm(train_dataloader, desc="Iteration", ncols=50):
            model.zero_grad()
            # 设置tensor gpu运行
            batch = tuple(t.to(args.device) for t in batch)

            
            if 'roberta' in args.model_type:
                input_ids, attention_mask, labels_main, labels_vice1, labels_vice2 = batch
                outputs = model(input_ids=input_ids.long(),
                                attention_mask=attention_mask.long(),
                                labels_main=labels_main,
                                labels_vice1=labels_vice1,
                                labels_vice2=labels_vice2,
                                model_type='roberta')
            else:
                input_ids, token_type_ids, attention_mask, labels_main, labels_vice1, labels_vice2 = batch
                outputs = model(input_ids=input_ids.long(),
                                token_type_ids=token_type_ids.long(),
                                attention_mask=attention_mask.long(),
                                labels_main=labels_main,
                                labels_vice1=labels_vice1,
                                labels_vice2=labels_vice2)

            loss = outputs[0]

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()  # 计算出梯度

            epoch_loss.append(loss.item())

            optimizer.step()
            scheduler.step()
            step += 1
            if step % 500 == 0:
                logger.debug("loss:"+str(np.array(epoch_loss).mean()))
                logger.debug('learning_rate:' + str(optimizer.state_dict()['param_groups'][0]['lr']))
            if step % args.saving_steps == 0:
                round += 1
                dev_loss, dev_acc, dev_f1 = test(model=model, tokenizer=tokenizer, test_file=args.dev_file,
                                                 checkpoint=epoch, round=round)
                logger.info(
                    '【DEV】Train Epoch %d, round %d: train_loss=%.4f, acc=%.4f, f1=%.4f' % (
                    epoch, round, dev_loss, dev_acc, dev_f1))

                test_loss, test_acc, test_f1 = test(model=model, tokenizer=tokenizer, test_file=args.test_file,
                                                    checkpoint=epoch, round=round)
                logger.info(
                    '【TEST】Train Epoch %d, round %d: train_loss=%.4f, acc=%.4f, f1=%.4f' % (
                    epoch, round, test_loss, test_acc, test_f1))
                output_dir = args.save_dir + "/checkpoint-" + str(epoch) + '-' + str(round)
                if test_acc > max_test_acc or test_f1 > max_test_f1:
                    max_test_acc = max(test_acc, max_test_acc)
                    max_test_f1 = max(test_f1, max_test_f1)

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
                model.train()

            # 保存模型
        output_dir = args.save_dir + "/checkpoint-" + str(epoch) + '-' + str(round)
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

        dev_loss, dev_acc, dev_f1 = test(model=model, tokenizer=tokenizer, test_file=args.dev_file, checkpoint=epoch, round=round)
        test_loss, test_acc, test_f1 = test(model=model, tokenizer=tokenizer, test_file=args.test_file, checkpoint=epoch, round=round)
        #print(test_loss, test_acc)
        logger.info(
            '【DEV】Train Epoch %d, round %d: train_loss=%.4f, acc=%.4f, f1=%.4f' % (
            epoch, round, dev_loss, dev_acc, dev_f1))
        logger.info(
            '【TEST】Train Epoch %d, round %d: train_loss=%.4f, acc=%.4f, f1=%.4f' % (
            epoch, round, test_loss, test_acc, test_f1))
        if test_acc > max_test_acc or test_f1 > max_test_f1:
            max_test_acc = max(test_acc, max_test_acc)
            max_test_f1 = max(test_f1, max_test_f1)
    logger.info('【BEST TEST ACC】: %.4f,   【BEST TEST F1】: %.4f' % (max_test_acc, max_test_f1))

def test(model, tokenizer, test_file, checkpoint, round, output_dir=None):
    print(type(model))
    test_data = Multi_task_dataset(data_file=test_file,
                                    max_length=args.max_length,
                                    tokenizer=tokenizer,
                                    model_type=args.model_type)

    test_dataLoader = DataLoader(dataset=test_data,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    logger.debug("***** Running test {} for round {}*****".format(checkpoint, round))
    logger.debug("  Num examples = %d", len(test_dataLoader))
    logger.debug("  Batch size = %d", args.batch_size)

    loss = []

    all_labels = None
    all_logits = None

    model.eval()

    for batch in tqdm(test_dataLoader, desc="Evaluating", ncols=50):
        with torch.no_grad():

            batch = [t.to(args.device) for t in batch]

            
            if 'roberta' in args.model_type:
                input_ids, attention_mask, labels_main, labels_vice1, labels_vice2 = batch
                outputs = model(input_ids=input_ids.long(),
                                attention_mask=attention_mask.long(),
                                labels_main=labels_main,
                                labels_vice1=labels_vice1,
                                labels_vice2=labels_vice2,
                                model_type='roberta')
            else:
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

    # 创建存储目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger = getLogger(__name__, os.path.join(args.save_dir, 'log.txt'))

    if 'roberta' in args.model_type:
        MatchModel = VaeMultiTaskMatchRobertaModelClean
        Tokenizer = RobertaTokenizer
    elif 'bert' in args.model_type or 'ernie' in args.model_type:
        MatchModel = VaeMultiTaskMatchModelClean
        Tokenizer = BertTokenizer
    # elif 'ernie' in args.model_type:
    #     MatchModel = VaeBertMatchModel
    #     Tokenizer = AutoTokenizer

    if args.do_train:
        # train： 接着未训练完checkpoint继续训练
        checkpoint = -1
        round = -1
        for checkpoint_dir_name in glob.glob(args.save_dir + "/*"):
            try:
                checkpoint = max(checkpoint, int(checkpoint_dir_name.split('/')[-1].split('-')[1]))
                round = max(round, int(checkpoint_dir_name.split('/')[-1].split('-')[-1]))
            except Exception as e:
                pass
        checkpoint_dir = args.save_dir + "/checkpoint-" + str(checkpoint) + '-' + str(round)
        if checkpoint > -1:
            logger.debug(f" Load Model from {checkpoint_dir}")

        tokenizer = Tokenizer.from_pretrained(args.model_type if checkpoint == -1 else checkpoint_dir,
                                              do_lower_case=args.do_lower_case)
        model = MatchModel.from_pretrained(args.model_type if checkpoint == -1 else checkpoint_dir)
        model.to(args.device)
        # 训练
        train(model, tokenizer, checkpoint, round)

    else:
        # eval：指定模型
        checkpoint = args.checkpoint
        round = args.round
        checkpoint_dir = args.save_dir + "/checkpoint-" + str(checkpoint) + '-' + str(round)
        tokenizer = Tokenizer.from_pretrained(checkpoint_dir,
                                              do_lower_case=args.do_lower_case)
        model = MatchModel.from_pretrained(checkpoint_dir)
        model.to(args.device)
        # 评估
        test_loss, test_acc, test_f1 = test(model, tokenizer, test_file=args.test_file, checkpoint=checkpoint, round=round)
        logger.debug('Evaluate Epoch %d: test_loss=%.4f, test_acc=%.4f, test_f1=%.4f' % (
        checkpoint, test_loss, test_acc, test_f1))