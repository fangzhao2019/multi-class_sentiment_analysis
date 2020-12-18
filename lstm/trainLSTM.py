import numpy as np
import keras
from keras.layers import Flatten
from keras.models import load_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, Bidirectional
from keras.models import Sequential

import matplotlib.pyplot as plt
batch_size = 128
epochs = 50

def trans(testLabel,labelSet,num_classes):
    newTestLabel=np.zeros((len(testLabel),num_classes)).astype('float32')
    for i in range(len(testLabel)):
        label=testLabel[i]
        index=labelSet.index(label)
        newTestLabel[i][index]=1.0 
    return newTestLabel

def evaluate(testLabel,predictLabel,labelSet):

    results_count=np.zeros((len(labelSet),len(labelSet)))
    for i in range(len(testLabel)):
        index1=testLabel[i].argmax()
        index2=predictLabel[i].argmax()
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
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('the accuracy of CNN iteration')
    plt.savefig('/home/som/Documents/lee/multi-class_sentiment_analysis/lstm/result/accuracy_cnn.jpg')
    plt.show()

def saveToTxt(x,train,dev,test):
    f=open('/home/som/Documents/lee/multi-class_sentiment_analysis/lstm/result/accuracy_cnn.txt','w',encoding='utf-8')
    for i in range(len(x)):
        f.write('%.4f %.4f %.4f %.4f'%(x[i],train[i],dev[i],test[i]))
        f.write('\n')
    f.close()

trainMat=np.load('mat/trainMat.npy')
trainLabel=np.load('mat/trainLabel.npy')
devMat=np.load('mat/testMat.npy')
devLabel=np.load('mat/testLabel.npy')
testMat=np.load('mat/testMat.npy')
testLabel=np.load('mat/testLabel.npy')

trainMat=np.sum(trainMat,axis=1).reshape(trainMat.shape[0],1,trainMat.shape[1])
devMat=np.sum(devMat,axis=1).reshape(devMat.shape[0],1,devMat.shape[1])
testMat=np.sum(testMat,axis=1).reshape(testMat.shape[0],1,testMat.shape[1])

trainMat = trainMat.astype('float32')
devMat = devMat.astype('float32')
testMat = testMat.astype('float32')
print('x_train shape:', trainMat.shape)
print(trainMat.shape[0], 'train samples')
print(devMat.shape[0], 'dev samples')
print(testMat.shape[0], 'test samples')

labelSet=list(set(trainLabel))
num_classes=len(labelSet)
print('共有%d个类别'%num_classes)

# convert class vectors to binary class matrices
trainLabel = trans(trainLabel,labelSet,num_classes)
devLabel = trans(devLabel,labelSet,num_classes)
testLabel = trans(testLabel,labelSet,num_classes)

model = Sequential()
model.add(Bidirectional(LSTM(32, recurrent_dropout=0.1)))
model.add(Dropout(0.25))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

best_dev_accuracy=-1
x=[]
train=[]
dev=[]
test=[]
print('正在训练模型')

for i in range(epochs):
    print('第%d次迭代'%i)
    model.fit(trainMat, trainLabel,
            batch_size=batch_size,
            epochs=1,
            verbose=1,
            validation_data=(devMat, devLabel))
    score1=model.evaluate(trainMat,trainLabel)[1]
    score2=model.evaluate(devMat,devLabel)[1]
    score3=model.evaluate(testMat,testLabel)[1]
    if score2>best_dev_accuracy:
        best_dev_accuracy=score2
        model.save('lstm.model')
    print(best_dev_accuracy)
    print('\n')
    x.append(i+1)
    train.append(score1)
    dev.append(score2)
    test.append(score3)

model=load_model('lstm.model')
result=model.predict(testMat,verbose=0)
fmeasure=evaluate(testLabel,result,labelSet)
accuracy=fmeasure['acc']
print('\n')
print("acc: %.4f" %(accuracy))
for k in fmeasure.keys():
    if k=='acc':continue
    print('label %s    p: %.4f, r: %.4f, f: %.4f'%(k, fmeasure[k]['p'], fmeasure[k]['r'], fmeasure[k]['f']))

drawFigure(x,train,dev,test)
saveToTxt(x,train,dev,test)
