import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class CNN_BiLSTM(nn.Module):
    def __init__(self,data):
        super(CNN_BiLSTM,self).__init__()
        print("build CNN_BiLSTM model...")
        data.show_data_summary()
        self.batch_size=data.HP_batch_size
        self.sentenceNumber= data.MAX_SENTENCE_NUMBER
        self.sentenceLenth= data.MAX_SENTENCE_LENGTH
        self.embedding_dim = data.word_emb_dim
        self.dropout= 0.25
        # 声明embedding层
        self.word_embeddings= nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.word_alphabet.size(),
                                                       self.embedding_dim)))
        self.dropembed = nn.Dropout(self.dropout)
        # 声明CNN
        self.in_channels = 1
        self.out_channels= 16#也就是卷积核的数量
        self.window_size1= 1
        self.window_size2= 2
        self.window_size3= 3
        self.window_size4= 4
        self.window_size5= 5
        self.conv_dim1= self.sentenceLenth-self.window_size1+1
        self.conv_dim2= self.sentenceLenth-self.window_size2+1
        self.conv_dim3= self.sentenceLenth-self.window_size3+1
        self.conv_dim4= self.sentenceLenth-self.window_size4+1
        self.conv_dim5= self.sentenceLenth-self.window_size5+1
        self.conv_dim = self.conv_dim1+ self.conv_dim2 +self.conv_dim3 + self.conv_dim4+self.conv_dim5
        # 声明LSTM
        self.bilstm_flag=True
        self.hidden_dim = 300
        self.lstm_layer = 1
        if self.bilstm_flag:
            lstm_hidden= self.hidden_dim//2
        else:lstm_hidden= self.hidden_dim

        self.droplstm = nn.Dropout(self.dropout)
        self.drop = nn.Dropout(self.dropout)
        #全连接层
        self.connection = 30

        self.conv1=nn.Conv2d(self.in_channels, self.out_channels,(self.window_size1,self.embedding_dim))
        self.conv2=nn.Conv2d(self.in_channels, self.out_channels,(self.window_size2,self.embedding_dim))
        self.conv3=nn.Conv2d(self.in_channels, self.out_channels,(self.window_size3,self.embedding_dim))
        self.conv4=nn.Conv2d(self.in_channels, self.out_channels,(self.window_size4,self.embedding_dim))
        self.conv5=nn.Conv2d(self.in_channels, self.out_channels,(self.window_size5,self.embedding_dim))
        self.batchNorm1=nn.BatchNorm2d(self.out_channels)
        self.batchNorm2=nn.BatchNorm2d(self.out_channels)
        self.batchNorm3=nn.BatchNorm2d(self.out_channels)
        self.batchNorm4=nn.BatchNorm2d(self.out_channels)
        self.batchNorm5=nn.BatchNorm2d(self.out_channels)

        self.lstm=nn.LSTM(self.conv_dim,lstm_hidden,
                          num_layers=self.lstm_layer,
                          batch_first=True, bidirectional=self.bilstm_flag)
        #self.fc1=nn.Linear(self.conv_dim,self.connection)
        self.fc1=nn.Linear(self.hidden_dim,self.connection)
        self.fc2=nn.Linear(self.connection,len(data.label_alphabet.instances))

        self.gpu=data.HP_gpu
        if self.gpu:
            self.word_embeddings = self.word_embeddings.cuda()
            self.dropembed=self.dropembed.cuda()
            self.droplstm=self.droplstm.cuda()
            self.drop=self.drop.cuda()
            self.conv1=self.conv1.cuda()
            self.conv2=self.conv2.cuda()
            self.conv3=self.conv3.cuda()
            self.conv4=self.conv4.cuda()
            self.conv5=self.conv5.cuda()
            self.batchNorm1=self.batchNorm1.cuda()
            self.batchNorm2=self.batchNorm2.cuda()
            self.batchNorm3=self.batchNorm3.cuda()
            self.batchNorm4=self.batchNorm4.cuda()
            self.batchNorm5=self.batchNorm5.cuda()
            self.lstm=self.lstm.cuda()
            self.fc1=self.fc1.cuda()
            self.fc2=self.fc2.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        """
        可以用来随机初始化word embedding
        """
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
            #pretrain_emb[index, :] = np.zeros((1, embedding_dim))
        return pretrain_emb

    def forward(self, x, batch_clauselen):
        self.batch_size= len(x)
        self.sentenceNumber=max(batch_clauselen)
        x=self.word_embeddings(x)
        x=self.dropembed(x)

        x=x.view(self.batch_size*self.sentenceNumber,1,self.sentenceLenth,self.embedding_dim)#

        x1=F.relu(self.conv1(x))
        x1=self.batchNorm1(x1)
        x1=x1.view(self.batch_size*self.sentenceNumber,self.out_channels,self.conv_dim1)
        x1=F.max_pool2d(x1,(self.out_channels,1))
        x1=x1.view(self.batch_size, self.sentenceNumber, self.conv_dim1)

        x2=F.relu(self.conv2(x))
        x2=self.batchNorm2(x2)
        x2=x2.view(self.batch_size*self.sentenceNumber,self.out_channels,self.conv_dim2)
        x2=F.max_pool2d(x2,(self.out_channels,1))
        x2=x2.view(self.batch_size, self.sentenceNumber, self.conv_dim2)

        x3=F.relu(self.conv3(x))
        x3=self.batchNorm3(x3)
        x3=x3.view(self.batch_size*self.sentenceNumber,self.out_channels,self.conv_dim3)
        x3=F.max_pool2d(x3,(self.out_channels,1))
        x3=x3.view(self.batch_size, self.sentenceNumber, self.conv_dim3)

        x4=F.relu(self.conv4(x))
        x4=self.batchNorm4(x4)
        x4=x4.view(self.batch_size*self.sentenceNumber,self.out_channels,self.conv_dim4)
        x4=F.max_pool2d(x4,(self.out_channels,1))
        x4=x4.view(self.batch_size, self.sentenceNumber, self.conv_dim4)

        x5=F.relu(self.conv5(x))
        x5=self.batchNorm5(x5)
        x5=x5.view(self.batch_size*self.sentenceNumber,self.out_channels,self.conv_dim5)
        x5=F.max_pool2d(x5,(self.out_channels,1))
        x5=x5.view(self.batch_size, self.sentenceNumber, self.conv_dim5)

        x=torch.cat((x1,x2,x3,x4,x5),2)
        #print(x.size())
        x=pack_padded_sequence(x, batch_clauselen, batch_first=True)
        x, (h, c) = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        #h0=torch.randn(2,self.batch_size,150).cuda()
        #c0=torch.randn(2,self.batch_size,150).cuda()
        #x,hn=self.lstm(x, (h0,c0))

        x = self.droplstm(x)
        x = x.view(self.batch_size*self.sentenceNumber,-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.softmax(self.fc2(x),dim=1)
        x =x.view(self.batch_size, self.sentenceNumber,-1)
        return x