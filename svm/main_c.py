# coding:utf-8
from __future__ import division
import numpy as np
import time
from sklearn import svm
from sklearn.externals import joblib
import matplotlib.pyplot as plt

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

def drawFigure(x,y1,y2,y3):
    plt.figure()
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.title('the accuracy of C selection')
    plt.savefig('/home/amax/Documents/robot/Lee/zhao/implicit-opinion/svm/result/accuracy_svm_C.jpg')
    plt.show()

def saveToTxt(x,train,dev,test):
    f=open('/home/amax/Documents/robot/Lee/zhao/implicit-opinion/svm/result/accuracy_svm_C.txt','w',encoding='utf-8')
    for i in range(len(x)):
        f.write('%.4f %.4f %.4f %.4f'%(x[i],train[i],dev[i],test[i]))
        f.write('\n')
    f.close()

time1=time.time()
trainMat=np.load('trainMat.npy')
testMat=np.load('testMat.npy')
devMat=np.load('testMat.npy')
trainLabel=np.load('trainLabel.npy')
testLabel=np.load('testLabel.npy')
devLabel=np.load('testLabel.npy')

#训练模型
best_dev_accuracy=[0,'tt',0,-1]
#kernelSet=['linear','rbf']
x=[]
train=[]
dev=[]
test=[]

print('正在训练模型')
epoch=0
for c in range(1,200):
    c=c/10.
    for gamma in range(1,2):
        c=9.7
        kernel='rbf'
        gamma=gamma/10.
        print('当前迭代次数epoch=%d'%epoch)
        print('当前超参数设置C=%.3f,kernal=%s,gamma=%.3f'%(c,kernel,gamma))
        clf=svm.SVC(C=c,kernel=kernel,gamma=gamma)
        clf=clf.fit(trainMat,trainLabel)

        print('训练数据在训练集上的结果：')
        fmeasure=evaluate(clf,trainMat,trainLabel,trainLabel)
        accuracy1=fmeasure['acc']
        print("acc: %.4f" %accuracy1)
        for k in fmeasure.keys():
            if k=='acc':continue
            print('label %s    p: %.3f, r: %.3f, f: %.3f'%(k, fmeasure[k]['p'], fmeasure[k]['r'], fmeasure[k]['f']))

        print('训练数据在验证集上的结果：')
        fmeasure=evaluate(clf,devMat,devLabel,trainLabel)
        accuracy2=fmeasure['acc']
        print("acc: %.4f" %accuracy2)
        for k in fmeasure.keys():
            if k=='acc':continue
            print('label %s    p: %.3f, r: %.3f, f: %.3f'%(k, fmeasure[k]['p'], fmeasure[k]['r'], fmeasure[k]['f']))

        if accuracy2>best_dev_accuracy[3]:
            best_dev_accuracy[0]=c
            best_dev_accuracy[1]=kernel
            best_dev_accuracy[2]=gamma
            best_dev_accuracy[3]=accuracy2
            joblib.dump(clf,'svm.model')
        epoch+=1

        print('训练数据在测试集上的结果：')
        fmeasure=evaluate(clf,testMat,testLabel,trainLabel)
        accuracy3=fmeasure['acc']
        print("acc: %.4f" %accuracy3)
        for k in fmeasure.keys():
            if k=='acc':continue
            print('label %s    p: %.3f, r: %.3f, f: %.3f'%(k, fmeasure[k]['p'], fmeasure[k]['r'], fmeasure[k]['f']))

        print(best_dev_accuracy)
        print('\n')
        x.append(c)
        train.append(accuracy1)
        dev.append(accuracy2)
        test.append(accuracy3)

print('验证集上最佳结果:')
print(best_dev_accuracy)
clf=joblib.load('svm.model')
fmeasure=evaluate(clf,testMat,testLabel,trainLabel)
accuracy=fmeasure['acc']
print('测试集上最终结果：')
print("acc: %.4f" %accuracy)
for k in fmeasure.keys():
    if k=='acc':continue
    print('label %s    p: %.4f, r: %.4f, f: %.4f'%(k, fmeasure[k]['p'], fmeasure[k]['r'], fmeasure[k]['f']))

drawFigure(x,train,dev,test)
saveToTxt(x,train,dev,test)
time2=time.time()
print('耗时%d秒'%(time2-time1))





