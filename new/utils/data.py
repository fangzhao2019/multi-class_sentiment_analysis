import os
import sys
import copy
import pickle
import numpy as np
from tqdm import tqdm
from xml.dom.minidom import parse


class Data:
    def __init__(self):
        self.train_dataset = []
        self.valid_dataset = []
        self.test_dataset = []

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Train  Instance Number: %s \t Total Sentence Number: %s" % (len(self.train_dataset), sum([len(d['word_recovery']) for d in self.train_dataset])))
        print("     Valid  Instance Number: %s \t Total Sentence Number: %s" % (len(self.valid_dataset), sum([len(d['word_recovery']) for d in self.valid_dataset])))
        print("     Test   Instance Number: %s \t Total Sentence Number: %s" % (len(self.test_dataset), sum([len(d['word_recovery']) for d in self.test_dataset])))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def cal_loss_weight(self, remain_neutral):
        label_set = np.concatenate([d['labels'] for d in self.train_dataset]) > 0
        print('     Label Num Count:', '  '.join(['%d' % lw for lw in label_set.sum(0)]))
        T = label_set.sum(0) / label_set.sum()
        loss_weight = 1. / np.log(1.2 + T)
        #  将中性情感的损失权重除以10
        if remain_neutral:
            loss_weight[-1] = loss_weight[-1] / 10.
        print("     Loss Weight:", '  '.join(['%.3f' % lw for lw in loss_weight]))
        return loss_weight

    def generate_instance(self, args, tokenizer, nlp):
        if 'NLPCC' in args.dataset_name:
            train_file = './data/{}/training_data.xml'.format(args.dataset_name)
            test_file = './data/{}/testing_data.xml'.format(args.dataset_name)
            training_data = data_read_nlpcc(train_file, tokenizer, nlp, args)
            testing_data = data_read_nlpcc(test_file, tokenizer, nlp, args)
            self.train_dataset = training_data[:int(len(training_data) * 0.8)]
            self.valid_dataset = training_data[int(len(training_data) * 0.8):]
            self.test_dataset = testing_data
        elif 'Ren_CECps' in args.dataset_name:
            all_data_file = './data/{}'.format(args.dataset_name)
            all_data = data_read_ren_cecps(all_data_file, tokenizer, nlp, args)
            self.train_dataset = all_data[: int(len(all_data) * 0.8)]
            self.valid_dataset = all_data[int(len(all_data) * 0.8): int(len(all_data) * 0.9)]
            self.test_dataset = all_data[int(len(all_data) * 0.9):]


def build_data(args, tokenizer, nlp):
    print('Dataset_name: {}'.format(args.dataset_name))
    file = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    if os.path.exists(file) and not args.refresh:
        data = load_data_setting(args)
    else:
        data = Data()
        data.generate_instance(args, tokenizer, nlp)
        save_data_setting(data, args)
    return data


def save_data_setting(data, args):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(args.generated_data_directory):
        os.makedirs(args.generated_data_directory)
    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting is saved to file: ", saved_path)


def load_data_setting(args):
    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting is loaded from file: ", saved_path)
    data.show_data_summary()
    return data


def sent_process(sentences, tokenizer, nlp):
    # 原句（连续）
    sent = '[CLS]' + '[SEP]'.join(sentences) + '[SEP]'
    # 原句由BERT分割的字符token
    token_cut = ['[CLS]']
    # 原句子分割后对应被切割词的原始位置
    word_recovery = []
    # 词的邻接矩阵
    adj_matrix = []

    num = 1
    for sent_num, sentence in enumerate(sentences):
        if len(sentence) == 0:  # 确保子句长度不为0
            continue
        words = nlp(sentence)
        word_rec = []

        matrix = np.zeros((len(words), len(words))).astype('float32')
        for word in words:
            matrix[word.i, word.i] = 1
            matrix[word.head.i, word.i] = 1
            matrix[word.i, word.head.i] = 1

            word_rec.append(num)
            tokens = tokenizer.tokenize(word.text)
            num += len(tokens)
            for token in tokens:
                token_cut.append(token)

        num += 1
        token_cut.append('[SEP]')
        word_recovery.append(word_rec)
        adj_matrix.append(matrix)

    # 原句由BERT分割的字符token_id
    token_cut_id = tokenizer.convert_tokens_to_ids(token_cut)
    token_cut = ' '.join(token_cut)

    return {'sent': sent, 'num_of_clauses': len(word_recovery), 'token_cut': token_cut, 'token_cut_id': token_cut_id,
            'word_recovery': word_recovery, 'adj_matrix': adj_matrix}


def data_read_nlpcc(filepath, tokenizer, nlp, args):
    label_set = ['like', 'happiness', 'sadness', 'fear', 'surprise', 'disgust', 'anger', 'none']
    if '2013' in filepath and 'testing' in filepath:
        label_set = ['喜好', '高兴', '悲伤', '恐惧', '惊讶', '厌恶', '愤怒', '无']

    dataset = []
    dom = parse(filepath)
    raw_weibo_set = dom.getElementsByTagName("weibo")
    for raw_weibo in tqdm(raw_weibo_set):
        sentences = []
        labels = []
        raw_sentences = raw_weibo.getElementsByTagName('sentence')
        for raw_sentence in raw_sentences:
            if not raw_sentence.firstChild:
                continue
            sentences.append(raw_sentence.firstChild.data)
            emotion_type = []
            if raw_sentence.hasAttribute('emotion_tag'):
                emotion_tag = raw_sentence.getAttribute('emotion_tag')
            else:  # elif raw_sentence.hasAttribute('opinionated'):
                emotion_tag = raw_sentence.getAttribute('opinionated')
            if emotion_tag == 'N':
                emotion_type.append('none')
            elif emotion_tag == 'Y':
                if raw_sentence.hasAttribute('emotion-1-type'):
                    emotion_type.append(raw_sentence.getAttribute('emotion-1-type'))
                if raw_sentence.hasAttribute('emotion-2-type'):
                    emotion_type.append(raw_sentence.getAttribute('emotion-2-type'))
            label = np.array([1 if e in emotion_type else 0 for e in label_set]).astype(np.float32)
            if label[: -1].sum() == 0.:
                label[-1] = 1.
            else:
                label[-1] = 0.
            labels.append(label)
        data = sent_process(sentences, tokenizer, nlp)

        # 句子长度筛选
        if len(data['token_cut_id']) > args.max_sent_len:
            continue
        # 子句数量筛选
        if data['num_of_clauses'] > args.max_clause_num or data['num_of_clauses'] == 0:
            continue
        # 子句长度筛选
        if max([len(c) for c in data['word_recovery']]) > args.max_clause_len:
            continue

        data['labels'] = np.array(labels)
        dataset.append(data)
    return dataset


def data_read_ren_cecps(path, tokenizer, nlp, args):
    # standardized_label: 是否将标签标准化为0和1的离散值
    # cut_paragraph: 是否根据段落切割成多条数据
    dataset = []

    for file in tqdm(os.listdir(path)):
        filepath = '{}/{}'.format(path, file)
        dom = parse(filepath)
        raw_paragraphs = dom.getElementsByTagName("paragraph")
        for raw_paragraph in raw_paragraphs:
            sentences = []
            labels = []

            raw_sentences = raw_paragraph.getElementsByTagName("sentence")
            for raw_sentence in raw_sentences:
                sentences.append(raw_sentence.getAttribute('S'))
                label = np.array([float(raw_sentence.getElementsByTagName('Joy')[0].firstChild.data),
                                 float(raw_sentence.getElementsByTagName('Hate')[0].firstChild.data),
                                 float(raw_sentence.getElementsByTagName('Love')[0].firstChild.data),
                                 float(raw_sentence.getElementsByTagName('Sorrow')[0].firstChild.data),
                                 float(raw_sentence.getElementsByTagName('Anxiety')[0].firstChild.data),
                                 float(raw_sentence.getElementsByTagName('Surprise')[0].firstChild.data),
                                 float(raw_sentence.getElementsByTagName('Anger')[0].firstChild.data),
                                 float(raw_sentence.getElementsByTagName('Expect')[0].firstChild.data)])
                if args.standardized_label:
                    label = (label > 0).astype(np.float32)
                else:
                    label = label
                    # 直接归一化
                    # label = label / sum(label)
                    # 使用mask_softmax归一化
                    mask = label > 0
                    label = np.exp(label * mask) / sum(np.exp(label * mask))
                    label = label * mask
                    label = label / (label.sum(-1) + 1e-13)

                if args.remain_neutral:
                    label = np.append(label, [0.])
                    if label.sum() == 0.:
                        label[-1] = 1.
                labels.append(label)
            data = sent_process(sentences, tokenizer, nlp)

            # 句子长度筛选
            if len(data['token_cut_id']) > args.max_sent_len:
                continue
            # 子句数量筛选
            if data['num_of_clauses'] > args.max_clause_num or data['num_of_clauses'] == 0:
                continue
            # 子句长度筛选
            if max([len(c) for c in data['word_recovery']]) > args.max_clause_len:
                continue

            # 剔除包含中性情感的句子
            if (not args.remain_neutral) and (0 in np.array(labels).sum(-1)):
                continue

            data['labels'] = np.array(labels)
            dataset.append(data)
    return dataset


