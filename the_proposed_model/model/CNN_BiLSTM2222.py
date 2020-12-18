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
        self.dropout= 0.3
        # 声明embedding层
        self.word_embeddings= nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        self.dropembed = nn.Dropout(self.dropout)
        # 声明CNN
        self.in_channels = 1
        self.out_channels1 = 32#也就是卷积核的数量
        self.out_channels2 = 64
        self.kernel_size = 3
        self.pool_size = 2

        self.conv_dim1=(self.sentenceLenth-2*self.kernel_size+2)/2
        self.conv_dim2=(self.embedding_dim-2*self.kernel_size+2)/2
        self.conv_dim=int(self.out_channels2*self.conv_dim1*self.conv_dim2)
        # 声明LSTM
        self.bilstm_flag=True
        self.hidden_dim = 300
        self.lstm_layer = 1
        if self.bilstm_flag:
            lstm_hidden= self.hidden_dim//2
        else:lstm_hidden= self.hidden_dim

        self.droplstm = nn.Dropout(self.dropout)
        self.drop = nn.Dropout(self.dropout)
        self.drop1 = nn.Dropout(self.dropout)
        self.drop2 = nn.Dropout(self.dropout)
        #全连接层
        self.connection = 128

        self.conv1=nn.Conv2d(self.in_channels, self.out_channels1,(self.kernel_size,self.kernel_size))
        self.conv2=nn.Conv2d(self.out_channels1, self.out_channels2,(self.kernel_size,self.kernel_size))
        self.lstm=nn.LSTM(self.conv_dim,lstm_hidden,
                          num_layers=self.lstm_layer,
                          batch_first=True, bidirectional=self.bilstm_flag)
        self.fc1=nn.Linear(self.hidden_dim,self.connection)
        self.fc2=nn.Linear(self.connection,len(data.label_alphabet.instances))

        self.gpu=data.HP_gpu
        if self.gpu:
            self.word_embeddings = self.word_embeddings.cuda()
            self.dropembed=self.dropembed.cuda()
            self.droplstm=self.droplstm.cuda()
            self.drop1=self.drop1.cuda()
            self.drop2=self.drop2.cuda()
            self.conv1=self.conv1.cuda()
            self.conv2=self.conv2.cuda()
            self.lstm=self.lstm.cuda()
            self.fc1=self.fc1.cuda()
            self.fc2=self.fc2.cuda()

    def forward(self, x, batch_clauselen):
        self.batch_size= len(x)
        self.sentenceNumber=max(batch_clauselen)
        x=self.word_embeddings(x)
        x=self.dropembed(x)

        x=x.view(self.batch_size*self.sentenceNumber,1,self.sentenceLenth,self.embedding_dim)

        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x,(self.pool_size,self.pool_size))
        x = self.drop1(x)
        x=x.view(self.batch_size, self.sentenceNumber, self.conv_dim)

        x=pack_padded_sequence(x, batch_clauselen, batch_first=True)
        x, (h, c) = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.droplstm(x)

        x=x.view(self.batch_size*self.sentenceNumber, -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.softmax(self.fc2(x),dim=1)
        x =x.view(self.batch_size, self.sentenceNumber,-1)
        return x