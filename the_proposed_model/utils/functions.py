import sys
import numpy as np
from utils.alphabet import Alphabet
import gensim

NULLKEY = "-null-"

#将数字统一
def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

#加载与训练词向量
def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=256, norm=True):
    if embedding_path != None:
        embedd_model = gensim.models.Word2Vec.load(embedding_path)
    scale = np.sqrt(3.0 / embedd_dim)#决定生成的随机向量的范围
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_model:#相当于判断字典的key是否包含word
            if norm:
                pretrain_emb[index] = norm2one(embedd_model[word])
            else:
                pretrain_emb[index] = embedd_model[word]
            perfect_match += 1
        elif word.lower() in embedd_model:
            if norm:
                pretrain_emb[index] = norm2one(embedd_model[word.lower()])
            else:
                pretrain_emb[index] = embedd_model[word.lower()]
            case_match += 1
        else:
            #pretrain_emb[index] = np.random.uniform(-scale, scale, embedd_dim)
            pretrain_emb[index] = np.zeros(embedd_dim)
            not_match += 1
    #pretrained_size = len(embedd_model)
    #print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
    #pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / word_alphabet.size()))
    return pretrain_emb, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square

def read_instance(input_file, word_alphabet, label_alphabet, number_normalized, max_sent_number, max_sent_length):
    in_lines = open(input_file, 'r').readlines()
    instence_texts = []#最终数据与标签
    instence_Ids = []
    clauseSet = []#存储每条数据及其标签的临时列表
    labelSet = []
    clauseSet_ids = []
    labelSet_ids = []
    for line in in_lines:
        if len(line) > 2:
            clauses=[]#存储当前数据的每条子句的列表
            clauses_id=[]
            pairs = line.strip().split()
            words = [w for w in pairs[:-1] if len(w)>0]#获取汉字
            label = pairs[-1]
            for word in words:
                if number_normalized:
                    word = normalize_word(word)
                    clauses.append(word)
                    clauses_id.append(word_alphabet.get_index(word))

            clauseSet.append(clauses)
            labelSet.append(label)
            clauseSet_ids.append(clauses_id)
            labelSet_ids.append(label_alphabet.get_index(label))
        else:
            #每一个句子结束后将这个句子的文本和id都存储起来
            max_length=max([len(clause) for clause in clauseSet])
            if (max_sent_length < 0) or (len(clauseSet) < max_sent_number) or (max_length<max_sent_length):
                instence_texts.append([clauseSet, labelSet])
                instence_Ids.append([clauseSet_ids, labelSet_ids])
            clauseSet = []
            labelSet = []
            clauseSet_ids = []
            labelSet_ids = []
    return instence_texts, instence_Ids