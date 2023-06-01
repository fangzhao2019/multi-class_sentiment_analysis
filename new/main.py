import os
import torch
import random
import argparse
import numpy as np
from transformers import BertTokenizer
from utils.data import build_data
from models.MESGM import MESGM
from metric import Metric
from trainer import Trainer


def str2bool(v):
    return v.lower() in ('true')


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--dataset_name', type=str, default="Ren_CECps", choices=['NLPCC2013', 'NLPCC2014', 'Ren_CECps'])
    data_arg.add_argument('--max_clause_num', type=int, default=15)  # Ren_CECps:  15    NLPCC2013:  15  NLPCC2014:  15
    data_arg.add_argument('--max_sent_len', type=int, default=510)  # Ren_CECps: 510    NLPCC2013: 185  NLPCC2014: 165
    data_arg.add_argument('--max_clause_len', type=int, default=180)  # Ren_CECps: 180    NLPCC2013:  95  NLPCC2014:  85
    data_arg.add_argument('--remain_neutral', type=str2bool, default=True)  # 是否保留中性情感标签
    data_arg.add_argument('--standardized_label', type=str2bool, default=True)  # 是否将情感标签转化为0,1离散值
    data_arg.add_argument('--checkpoint_path', type=str, default="./data/generated_data/checkpoint/")
    data_arg.add_argument('--generated_data_directory', type=str, default="./data/generated_data/")
    data_arg.add_argument('--best_param_directory', type=str, default="./data/generated_data/model_param/")
    data_arg.add_argument('--bert_directory', type=str, default="./bert-base-chinese")

    learn_arg = parser.add_argument_group('Learning')
    learn_arg.add_argument('--model_name', type=str, default="MESGM")
    learn_arg.add_argument('--fix_bert_embeddings', type=str2bool, default=True)  # 固定bert参数
    learn_arg.add_argument('--hidden_size', type=int, default=256)
    learn_arg.add_argument('--label_embedding_dim', type=int, default=300)
    learn_arg.add_argument('--num_of_classes', type=int, default=8)
    learn_arg.add_argument('--num_labelDecoder_layers', type=int, default=1)
    learn_arg.add_argument('--batch_size', type=int, default=4)
    learn_arg.add_argument('--max_epoch', type=int, default=50)
    learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learn_arg.add_argument('--decoder_lr', type=float, default=2e-5)
    learn_arg.add_argument('--encoder_lr', type=float, default=1e-5)
    learn_arg.add_argument('--lr_decay', type=float, default=0.01)
    learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learn_arg.add_argument('--max_grad_norm', type=float, default=0)
    learn_arg.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])

    evaluation_arg = parser.add_argument_group('Evaluation')
    evaluation_arg.add_argument('--test_batch_size', type=int, default=50)

    misc_arg = parser.add_argument_group('MISC')
    misc_arg.add_argument('--resume', type=str2bool, default=True)  # train from the checkpoint
    misc_arg.add_argument('--save_txt', type=str2bool, default=True)
    misc_arg.add_argument('--save_mp', type=str2bool, default=True)
    misc_arg.add_argument('--refresh', type=str2bool, default=False)  # refresh the data
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--random_seed', type=int, default=1)
    misc_arg.add_argument('--gpu_id', type=str, default="0")

    args, unparsed = get_args()
    if not os.path.exists(args.generated_data_directory):
        os.makedirs(args.generated_data_directory)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.best_param_directory):
        os.makedirs(args.best_param_directory)
    if args.dataset_name == 'Ren_CECps':
        args.max_sent_len = 510
        args.max_clause_len = 180
    elif args.dataset_name == 'NLPCC2013':
        args.max_sent_len = 185
        args.max_clause_len = 95
    elif args.dataset_name == 'NLPCC2014':
        args.max_sent_len = 165
        args.max_clause_len = 85

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    import spacy  # 必须放在os_environ后面，不然默认使用全部gpu

    nlp = spacy.load('zh_core_web_sm')
    tokenizer = BertTokenizer.from_pretrained(args.bert_directory, do_lower_case=False)

    set_seed(args.random_seed)
    data = build_data(args, tokenizer, nlp)

    loss_weight = data.cal_loss_weight(args.remain_neutral)
    args.num_of_classes = data.train_dataset[0]['labels'].shape[-1]
    model = MESGM(args, loss_weight)

    if args.use_gpu and torch.cuda.is_available():
        print("GPU is True, the number of GPU is %d" % torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    else:
        print("GPU is False")
    model.to(device)

    # for n, p in model.named_parameters():
    #     print(n, p.size(), p.requires_grad)

    metric = Metric(args)
    trainer = Trainer(args, data, model, metric, device)
    trainer.train_model()
