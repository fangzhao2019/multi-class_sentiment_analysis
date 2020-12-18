# coding:utf-8
from __future__ import division
import numpy as np
import jieba
import jieba.posseg as pseg
jieba.load_userdict('data/userdict.txt')
import os
import time
import gensim

#导入标记好的初始评论数据
def load_initComment(filename):
    f=open(filename,encoding='utf-8')
    dataSet=[row.strip() for row in f.readlines()]
    
    commentSet=[]
    labels=[]
    for i in range(1,len(dataSet)):
        data=dataSet[i].split('\t')
        comment=[w for w in jieba.cut(data[0])]
        label=int(data[1])
        commentSet.append(comment)
        labels.append(label)
    return np.array(commentSet),np.array(labels)

#为输入文本构建词向量
def create_vector(sen,padding_size,vec_size):
    matrix=[]
    for i in range(padding_size):
        try:
            matrix.append(model[sen[i]].tolist())
        except:
                # 这里有两种except情况，
                # 1. 这个单词找不到
                # 2. sen没那么长
                # 不管哪种情况，我们直接贴上全是0的vec
            matrix.append([0] * vec_size)
    return matrix

def transform_to_matrix(x,model, padding_size=128, vec_size=128):
    res = []
    for sen in x:
        matrix =create_vector(sen,padding_size,vec_size)  
        res.append(matrix)
    return res

time1=time.time()

trainComment,trainLabel=load_initComment('data/reviewTrain.txt')
print('导入训练数据%d条'%len(trainLabel))
#devComment,devLabel=load_initComment('data/reviewDev.txt')
#print('导入验证数据%d条'%len(devLabel))
testComment,testLabel=load_initComment('data/reviewTest.txt')
print('导入测试数据%d条'%len(testLabel))
print('载入词向量模型')
model=gensim.models.Word2Vec.load('data/word2vec/carCommentData.model')
print('生成训练数据矩阵')
trainMat=transform_to_matrix(trainComment,model)
#print('生成验证数据矩阵')
#devMat=transform_to_matrix(devComment,model)
print('生成测试数据矩阵')
testMat=transform_to_matrix(testComment,model)

print(u'数据矩阵构建完毕，正在保存。。。')
np.save('mat/trainMat.npy',trainMat)
#np.save('mat/devMat.npy',devMat)
np.save('mat/testMat.npy',testMat)
np.save('mat/trainLabel.npy',trainLabel)
#np.save('mat/devLabel.npy',devLabel)
np.save('mat/testLabel.npy',testLabel)

time2=time.time()
print('耗时%d秒'%(time2-time1))
