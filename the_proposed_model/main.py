import argparse
from utils.data import Data
import pickle
from utils.batchify_with_label import batchify_with_label
from utils.metric import get_ner_fmeasure
import time
import sys
import torch.optim as optim
from model.CNN_BiLSTM import CNN_BiLSTM
import random
import numpy as np
import torch
import gc
import os
import copy
import matplotlib.pyplot as plt
#数据
def data_initialization(data, train_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(test_file)
    data.fix_alphabet()
    return data

#更新学习率
def lr_decay(optimizer, epoch, decay_rate, init_lr):
    # 用于衰减学习率
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

#更新学习率
def lr_decay1(optimizer, epoch, decay_rate, init_lr):
    # 用于衰减学习率
    lr=init_lr
    if epoch>25:lr=0.001
    if epoch>50:lr=0.0001
    if epoch>75:lr=0.00001
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def calculateLoss(tag_seq, batch_label,loss_func, gpu):
    batch_size=len(tag_seq)
    clause_number=len(tag_seq[0])
    label_number=len(tag_seq[0][0])
    target= torch.zeros((batch_size,clause_number,label_number)).float()
    for idx in range(batch_size):
        for idy in range(clause_number):
            index=batch_label[idx,idy]
            target[idx,idy,index]= 1
    if gpu:
    	target=target.cuda()
    loss=loss_func(tag_seq, target)
    return loss

def MSELOSS(weight,predict,fact):
    number=1.
    for num in predict.size():
        number*=num

    loss=weight[0]*sum((predict[:,0]-fact[:,0])**2)
    for i in range(1,len(weight)):
        loss+=weight[i]*sum((predict[:,i]-fact[:,i])**2)
    return loss/number

def calculateLoss2(weight, tag_seq, batch_label, gpu):
    batch_size=len(tag_seq)
    clause_number=len(tag_seq[0])
    label_number=len(tag_seq[0][0])
    target= torch.zeros((batch_size,clause_number,label_number)).float()
    for idx in range(batch_size):
        for idy in range(clause_number):
            index=batch_label[idx,idy]
            target[idx,idy,index]= 1

    target=target.view(batch_size*clause_number,-1)
    tag_seq=tag_seq.view(batch_size*clause_number,-1)
    if gpu:
    	target=target.cuda()

    loss=MSELOSS(weight,tag_seq, target)

    return loss

def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    #print(pred_variable)
    #print(gold_variable)
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    pred = np.argmax(pred, axis=2)
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label

def drawFigure(x,y1,y2,y3,filename):
    plt.figure()
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('the accuracy of iteration')
    plt.savefig(filename)
    plt.show()

def saveToTxt(x,train,dev,test,filename):
    f=open(filename,'w',encoding='utf-8')
    for i in range(len(x)):
        f.write('%.4f %.4f %.4f %.4f'%(x[i],train[i],dev[i],test[i]))
        f.write('\n')
    f.close()

def evaluate(data, model, name, padding_label):
    ## 评价函数
    if name == "train":
        instances = data.train_Ids
    elif name == 'test':
        instances = data.test_Ids
    else:
        print("Error: wrong evaluate name,", name)
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_clause, batch_clauselen, batch_clauserecover, batch_label, mask = batchify_with_label(instance,
                                                                                                    data.HP_gpu,
                                                                                                    padding_label,
                                                                                                    data.MAX_SENTENCE_LENGTH)
        #print(batch_clauserecover)
        tag_seq = model.forward(batch_clause, batch_clauselen)
        if data.HP_gpu:
        	tag_seq=torch.from_numpy(np.argmax(tag_seq.cpu().data.numpy(), axis=2)).cuda()
        else:
        	tag_seq=torch.from_numpy(np.argmax(tag_seq.data.numpy(), axis=2))
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_clauserecover)

        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    fmeasure = get_ner_fmeasure(gold_results, pred_results,data.label_alphabet.instances)
    return speed, fmeasure

#根据标签的比例为损失函数设定权重
def calLabelWeight(data):
	labels=data.label_alphabet.instances
	label_count=[0]*len(labels)
	for i in range(len(data.train_texts)):
		train_label=data.train_texts[i][1]
		for label in train_label:
			index=labels.index(label)
			label_count[index]+=1
	label_weight1=[sum(label_count)/float(c) for c in label_count]
	label_weight2=[c/sum(label_weight1) for c in label_weight1]
	return torch.FloatTensor(label_weight2)

def train(data, model, save_model_dir, padding_label, seg=True):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("opimizer is Adam")
    optimizer = optim.Adam(parameters, lr=data.HP_lr, weight_decay=data.weight_decay)

    label_weight=calLabelWeight(data)
    if data.HP_gpu:
    	label_weight=label_weight.cuda()
    print(label_weight)

    #loss_func=torch.nn.CrossEntropyLoss()#weight=label_weight)
    loss_func=torch.nn.MSELoss()
    loss_list=[]
    acc_list=[]
    x=[]
    train1=[]

    best_test = -1
    for idx in range(data.HP_iteration):
        x.append(idx)
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, data.HP_iteration))
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_loss = 0  # 每500条数据汇总的损失
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)#随机排序
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1

        # 对输入做batch处理
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
                continue
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            model.zero_grad()
            # tensor化
            batch_clause, batch_clauselen, batch_clauserecover, batch_label, mask = batchify_with_label(instance, data.HP_gpu,
                                                                                padding_label, data.MAX_SENTENCE_LENGTH)
            instance_count += 1
            # 对将一个batch经过tensor化后的tensor输入到模型中
            tag_seq= model.forward(batch_clause, batch_clauselen)
            #print(tag_seq.size())
            loss= calculateLoss2(label_weight, tag_seq, batch_label,data.HP_gpu)
            #loss= calculateLoss( tag_seq, batch_label, loss_func, data.HP_gpu)
            # 结果一些辅助信息
            right, whole = predict_check(tag_seq, batch_label, mask)
            #print(right,whole)
            right_token += right
            whole_token += whole
            sample_loss += loss.item()
            total_loss += loss.item()
            # loss的反传及模型参数的优化
            loss.backward()
            optimizer.step()
            model.zero_grad()
            # 辅助信息
            if end % (batch_size*16) == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                    end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
                sys.stdout.flush()
                sample_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        acc1 = (right_token + 0.) / whole_token
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
        end, temp_cost, sample_loss, right_token, whole_token, acc1))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
        idx, epoch_cost, train_num / epoch_cost, total_loss))
        loss_list.append(total_loss)

        train1.append(acc1)
        # 在test集上评价
        speed, fmeasure = evaluate(data, model, "test", padding_label)
        test_finish = time.time()
        test_cost = test_finish - epoch_finish
        current_score_test = fmeasure['acc']
        acc_list.append(current_score_test)

        print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, current_score_test))
        for k in fmeasure.keys():
            if k=='acc':continue
            print('label %s    p: %.4f, r: %.4f, f: %.4f'%(k, fmeasure[k]['p'], fmeasure[k]['r'], fmeasure[k]['f']))

        if current_score_test > best_test:
            print("Exceed previous best acc score:", best_test)
            # model_name = save_model_dir + '_' + str(idx) + ".model"
            # torch.save(model.state_dict(), model_name)
            # print(model_name)
            best_test = current_score_test
        gc.collect()  # 删除输出，清理内存

    print(loss_list)
    print(acc_list)
    print(best_test)

    drawFigure(x,train1,acc_list,acc_list,'/home/som/Documents/lee/multi-class_sentiment_analysis/the_proposed_model/result/accuracy.jpg')
    saveToTxt(x,train1,acc_list,acc_list,'/home/som/Documents/lee/multi-class_sentiment_analysis/the_proposed_model/result/accuracy.txt')

    drawFigure(x,loss_list,loss_list,loss_list,'/home/som/Documents/lee/multi-class_sentiment_analysis/the_proposed_model/result/loss.jpg')
    saveToTxt(x,loss_list,loss_list,loss_list,'/home/som/Documents/lee/multi-class_sentiment_analysis/the_proposed_model/result/loss.txt')



if __name__=='__main__':
    seed_num = 100
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    torch.cuda.manual_seed(seed_num)

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    ## carComment
    train_file = "/home/som/Documents/lee/multi-class_sentiment_analysis/the_proposed_model/data/carComment.train"
    test_file = "/home/som/Documents/lee/multi-class_sentiment_analysis/the_proposed_model/data/carComment.test"

    word_emb_file='/home/som/Documents/lee/multi-class_sentiment_analysis/the_proposed_model/data/word2vec/carCommentData.model'
    print(train_file)
    data=Data()
    data.HP_gpu = True  # 是否使用GPU
    data.random_seed = seed_num
    data_initialization(data, train_file, test_file)
    data.build_word_pretrain_emb(word_emb_file)

    print('finish loading')
    data.generate_instance(train_file, 'train')
    print("train_file done")
    data.generate_instance(test_file, 'test')
    print("test_file done")
    print('random seed: ' + str(seed_num))
    # 模型的声明
    model = CNN_BiLSTM(data)
    print("打印模型可优化的参数名称")
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    save_model_dir = "data/model_para/OntoNotes_lstm_gat_crf_epoch"
    o_label2index = data.label_alphabet.instance2index['5']
    print(data.label_alphabet.instance2index)
    #print(o_label2index)
    #print(len(data.train_texts))
    #print(data.word_alphabet.instances[0])
    train(data, model, save_model_dir, o_label2index)