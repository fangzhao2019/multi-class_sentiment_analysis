import sys
import numpy as np
from utils.alphabet import Alphabet
from utils.functions import *

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"

class Data:
    """
    所有数据预处理程序都封装在Data类里面
    Data类中封装了Alphabet类，Alphabet类主要功能是word转成id，将id转成词
    Alphabet类的构建是通过build_alphabet函数构建的
    """
    #
    def __init__(self):
        self.MAX_SENTENCE_NUMBER= 8#最大句子数量
        self.MAX_SENTENCE_LENGTH = 64#句子最大长度
        self.number_normalized = True#是否将数字归一化
        self.norm_word_emb = True#是否将词向量归一化
        self.word_alphabet = Alphabet('word')#word的词表与id
        self.label_alphabet = Alphabet('label', True) #not end "</unk>"

        self.train_texts = []
        self.test_texts = []

        self.train_Ids = []
        self.test_Ids = []

        self.word_emb_dim = 256
        self.pretrain_word_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.label_alphabet_size = 0
        ### hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 16  # 1
 
        self.HP_gpu = False  # true
        self.HP_lr = 0.01
        self.HP_lr_decay = 0.05
        self.weight_decay = 0.00000005
        self.use_clip = False #是否控制梯度
        self.HP_clip = 30.0 #最大梯度
        self.HP_momentum = 0#控制优化器的一个超参
        self.random_seed = 100

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("                          Use GPU: %s" % (self.HP_gpu))
        print("              MAX SENTENCE NUMBER: %s" % (self.MAX_SENTENCE_NUMBER))
        print("              MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("              Number   normalized: %s" % (self.number_normalized))
        print("              Word  alphabet size: %s" % (self.word_alphabet_size))
        print("              Label alphabet size: %s" % (self.label_alphabet_size -1 ))
        print("              Word embedding size: %s" % (self.word_emb_dim))
        print("              Norm     word   emb: %s" % (self.norm_word_emb))
        print("            Train instance number: %s" % (len(self.train_texts)))
        print("            Test  instance number: %s" % (len(self.test_texts)))

        print("--*--整体参数设定区域--*--")
        print("     Hyperpara        random seed: %s" % (self.random_seed))
        print("     Hyperpara          iteration: %s" % (self.HP_iteration))
        print("     Hyperpara         batch size: %s" % (self.HP_batch_size))
        print("     Hyperpara                 lr: %s" % (self.HP_lr))
        print("     Hyperpara           lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyperpara       weight_decay: %s" % (self.weight_decay))
        if self.use_clip:
            print("     Hyperpara        HP_clip: %s" % (self.HP_clip))
        print("     Hyperpara           momentum: %s" % (self.HP_momentum))
        print("DATA SUMMARY END.")
        sys.stdout.flush()#强制刷新缓冲区

# 构建词典
    def build_alphabet(self, input_file):
        in_lines = open(input_file, 'r').readlines()
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:#有数据才处理
                pairs = line.strip().split()
                words = [w for w in pairs[:-1] if len(w)>0]#获取汉字
                for word in words:
                    if self.number_normalized:  # True
                        word = normalize_word(word)  # 把字符中的数字变成0
                    self.word_alphabet.add(word)
                label = pairs[-1]
                self.label_alphabet.add(label)

        self.word_alphabet_size = self.word_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.label_alphabet.close()

    def build_word_pretrain_emb(self, emb_path):
        # 载入预训练词向量
        print("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet,
                                                                                   self.word_emb_dim,
                                                                                   self.norm_word_emb)

    def generate_instance(self, input_file, name):
        # 产生训练开发训练数据
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(input_file, self.word_alphabet,
                                                                 self.label_alphabet, self.number_normalized,
                                                                 self.MAX_SENTENCE_NUMBER, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(input_file, self.word_alphabet,
                                                                 self.label_alphabet, self.number_normalized,
                                                                 self.MAX_SENTENCE_NUMBER, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))
