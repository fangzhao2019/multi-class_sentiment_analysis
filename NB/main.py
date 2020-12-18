# coding:utf-8
from __future__ import division
import numpy as np
import time
from sklearn import svm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

def evaluate(clf,testMat,testLabel,trainLabel):
    labelSet=list(set(trainLabel))
    results_count=np.zeros((len(labelSet),len(labelSet)))
    for i in range(len(testMat)):
        testData=testMat[i]
        label=testLabel[i]
        predict=clf.predict([testData])[0]
        index1=labelSet.index(label)
        index2=labelSet.index(predict)
        results_count[index1][index2]+=1
        
    fmeasure={}
    total_TP= 0
    for idx in range(len(labelSet)):
        metric={}
        TP=results_count[idx,idx]
        total_TP += TP
        precision= TP/float(np.sum(results_count,axis=0)[idx]+0.5)
        recall= TP/float(np.sum(results_count,axis=1)[idx]+0.5)
        f_score=2*precision*recall/float(recall+precision)
        metric['p']=precision
        metric['r']=recall
        metric['f']=f_score
        fmeasure[labelSet[idx]]=metric
    accuracy=total_TP/np.sum(results_count)
    fmeasure['acc']=accuracy
    return fmeasure

time1=time.time()
trainMat=np.load('trainMat.npy')
testMat=np.load('testMat.npy')
trainLabel=np.load('trainLabel.npy')
testLabel=np.load('testLabel.npy')


print('正在训练模型')
#clf=BernoulliNB()
clf=MultinomialNB()
clf=clf.fit(trainMat,trainLabel)

print('训练数据在训练集上的结果：')
fmeasure=evaluate(clf,trainMat,trainLabel,trainLabel)
accuracy1=fmeasure['acc']
print("acc: %.4f" %accuracy1)
for k in fmeasure.keys():
    if k=='acc':continue
    print('label %s    p: %.4f, r: %.4f, f: %.4f'%(k, fmeasure[k]['p'], fmeasure[k]['r'], fmeasure[k]['f']))

print('训练数据在测试集上的结果：')
fmeasure=evaluate(clf,testMat,testLabel,trainLabel)
accuracy3=fmeasure['acc']
print("acc: %.4f" %accuracy3)
for k in fmeasure.keys():
    if k=='acc':continue
    print('label %s    p: %.4f, r: %.4f, f: %.4f'%(k, fmeasure[k]['p'], fmeasure[k]['r'], fmeasure[k]['f']))

time2=time.time()
print('耗时%d秒'%(time2-time1))





