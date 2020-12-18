# coding:utf-8
from __future__ import division
import numpy as np
import jieba
import jieba.posseg as pseg
jieba.load_userdict('/home/som/Documents/lee/multi-class_sentiment_analysis/svm/userdict.txt')
import os
import time

#导入标记好的初始评论数据
def load_initComment(filename):
    f=open(filename,encoding='utf-8')
    dataSet=[row.strip() for row in f.readlines()]
    
    commentSet=[]
    labels=[]
    for i in range(1,len(dataSet)):
        data=dataSet[i].split('\t')
        comment=data[0]
        label=int(data[1])
        commentSet.append(comment)
        labels.append(label)
    return np.array(commentSet),np.array(labels)

#对评论进行分词，筛选掉停用词，存储为（word,pos）对的集合
def commentPOS(commentSet,stopWords,method='jieba'):
    commentVecSet=[]
    if method=='jieba':
        for i in range(len(commentSet)):
            commentVec=[(w.word,w.flag) for w in pseg.cut(commentSet[i]) if w.word not in stopWords]
            commentVecSet.append(commentVec)
    return np.array(commentVecSet)

#将分词后的评论向量根据词性进行筛选
def commentVecSet_Filter(commentVecSet,POS):
    commentPOSVec=[]
    for comment in commentVecSet:
        if POS=='all':
            POSVec=[com[0] for com in comment]
        else:
            POSVec=[com[0] for com in comment if com[1] in POS]
        commentPOSVec.append(POSVec)
    return commentPOSVec

#若参数为vocab，统计每个元素出现的频率，存储为一个词典
#若参数为gram，统计每个2-POS-gram出现的频率，存储为一个词典
def createVocabList(commentPOSVec,types='vocab'):
    vocabList={}
    for comment in commentPOSVec:
        if types=='gram':
            commentVec=[(comment[i],comment[i+1]) for i in range(len(comment)-1)]
        if types=='vocab':
            commentVec=comment
        for word in commentVec:
            if not word in vocabList.keys():
                vocabList[word]=0
            vocabList[word]+=1
    return vocabList

#若参数为vocab，根据出现频率对词汇进行筛选
#若参数为gram，根据出现频率对词对进行筛选
def vocabFilter(vocabList,min_freq,max_freq,types='vocab'):
    myVocab=[]
    sortResult=sorted(vocabList.items(),key=lambda e:e[1],reverse=True)
    for word in sortResult:
        if word[1]>min_freq and word[1]<max_freq:
            if types=='vocab':
                myVocab.append(word[0])
            if types=='gram':
                myVocab.append(word[0])
    return myVocab

def load_patternSet(filename):
    f=open(filename,encoding='utf-8')
    dataSet=[row.strip() for row in f.readlines()]
    patternSet=[]
    for i in range(1,len(dataSet)):
        pattern=dataSet[i].split(',')
        patternSet.append(pattern)
    return patternSet

#判断一个模式是否为另一个模式的子模式
def pattern_contain(pattern,comment):
    all_index=[]
    #首先计算出待测模式中的每个词在评论中的位置
    #print(pattern)
    for pat in pattern:
        all_index.append([i for i in range(len(comment)) if comment[i]==pat])
    position=-1
    for i in range(len(all_index)):
        if len(all_index[i])==0:return 0
        count=0
        for j in range(len(all_index[i])):
            if all_index[i][j]>position:
                position=all_index[i][j]
                break
            else:
                count+=1
        if count==len(all_index[i]):
            return 0
    return 1

#为输入文本构建分类属性向量，包含为1，否则为0
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

def setOfGrams2Vec(gramList, inputSet):
    returnVec=[0]*len(gramList)
    for pattern in gramList:
        if pattern_contain(pattern,inputSet):
            returnVec[gramList.index(pattern)]=1
    return returnVec

def create_dataMat(commentPOSVec,myVocab,myPattern):
    dataMat=[]
    for i in range(len(commentPOSVec)):
        data=setOfWords2Vec(myVocab, commentPOSVec[i])
        data.extend(setOfGrams2Vec(myPattern,commentPOSVec[i]))
        dataMat.append(data)
    return np.array(dataMat)

def evaluate(clf,testComment,testLabel,trainLabel):
    labelSet=list(set(trainLabel))
    results_count=np.zeros((len(labelSet),len(labelSet)))
    for i in range(len(testComment)):
        comment=testComment[i]
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
############################################
#训练数据
stopWords=u'，。的 ；了、：吧'
trainComment,trainLabel=load_initComment('reviewTrain.txt')
trainCommentVec=commentPOS(trainComment,stopWords,method='jieba')
trainCommentPOSVec=commentVecSet_Filter(trainCommentVec,['n','a','d','v'])
print('载入训练集成功,包含数据%d个'%len(trainLabel))
testComment,testLabel=load_initComment('reviewTest.txt')
testCommentVec=commentPOS(testComment,stopWords,method='jieba')
testCommentPOSVec=commentVecSet_Filter(testCommentVec,['n','a','d','v'])
print('载入测试集成功,包含数据%d个'%len(testLabel))
#devComment,devLabel=load_initComment('reviewDev.txt')
#devCommentVec=commentPOS(devComment,stopWords,method='jieba')
#devCommentPOSVec=commentVecSet_Filter(devCommentVec,['n','a','d','v'])
#print('载入验证集成功,包含数据%d个'%len(devLabel))
#根据训练集进行特征选择，vocab+pattern
vocabList=createVocabList(trainCommentPOSVec,'vocab')
myVocab=vocabFilter(vocabList,10,0.8*len(trainComment),'vocab')
print('共有单词%d个'%(len(myVocab)))
myPattern=load_patternSet(u'patternSet.txt')
print('共有模式%d个'%(len(myPattern)))

#构建数据矩阵
trainMat=create_dataMat(trainCommentPOSVec,myVocab,myPattern)
print('训练集数据矩阵构建完毕')
testMat=create_dataMat(testCommentPOSVec,myVocab,myPattern)
print('测试集数据矩阵构建完毕')  
#devMat=create_dataMat(devCommentPOSVec,myVocab,myPattern)
#print('验证集数据矩阵构建完毕')

np.save('trainMat.npy',trainMat)
np.save('testMat.npy',testMat)
#np.save('devMat.npy',devMat)
np.save('trainLabel.npy',trainLabel)
np.save('testLabel.npy',testLabel)
#np.save('devLabel.npy',devLabel)

            
time2=time.time()
print('耗时%d秒'%(time2-time1))





