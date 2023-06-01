import torch
import torch.nn as nn
from models.seq_encoder import SeqEncoder
from models.graph_convolution import GraphConvolution
from models import label_decoder_standard, label_decoder_soft
from transformers.modeling_bert import BertAttention, BertLayerNorm, BertIntermediate, BertOutput
import torch.nn.functional as F


class MESGM(nn.Module):
    def __init__(self, args, loss_weight):
        super(MESGM, self).__init__()
        self.args = args

        # 句子编码层
        self.seq_encoder = SeqEncoder(args)
        self.config = self.seq_encoder.config

        # GCN层
        self.gc1 = GraphConvolution(args.hidden_size, args.hidden_size)
        self.gc2 = GraphConvolution(args.hidden_size, args.hidden_size)

        # 池化层后的线性变换层
        self.linear_projection = nn.Linear(4 * args.hidden_size, args.hidden_size)
        self.pooling_dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # 子句信息编码层（自注意力 or 双向LSTM）（是否需要？？？）
        self.config.hidden_size = args.hidden_size
        self.config.num_attention_heads = 8
        self.attention = BertAttention(self.config)
        self.intermediate = BertIntermediate(self.config)
        self.output = BertOutput(self.config)

        # 标签解码层（LSTM)
        if args.standardized_label:
            self.label_decoder = label_decoder_standard.LabelDecoder(self.config, args)
        else:
            self.label_decoder = label_decoder_soft.LabelDecoder(self.config, args)

        # 损失函数
        weight = torch.from_numpy(loss_weight)
        if args.standardized_label:
            self.BCE_loss_func = nn.BCEWithLogitsLoss(weight=weight)
        else:
            self.KL_loss_func = nn.KLDivLoss(reduction='mean')
            # self.BCE_loss_func = nn.BCEWithLogitsLoss()
            # self.BCE_loss_func = nn.BCELoss()

    def forward(self, source_ids, source_mask, clause_num_mask, word_recovery, word_recovery_mask, adj_matrix, target_labels, is_training=True):
        # print(source_ids.size())
        # print(source_mask.size())
        # print(clause_num_mask.size())
        # print(word_recovery.size())
        # print(word_recovery_mask.size())
        # print(target_labels.size())

        state = self.seq_encoder(source_ids, source_mask)
        # print(state['encoder_hs'].sum(-1))
        state = self.clause_segmentation(state, word_recovery, word_recovery_mask)
        # print(state['encoder_clause_hs'].sum(-1))
        state = self.gcn_encoder(state, adj_matrix, word_recovery_mask)
        # print(state['encoder_GCN_hs'].sum(-1))
        state = self.pooling(state, word_recovery_mask)
        # print(state['clause_vector'].sum(-1))
        state = self.self_attention(state, clause_num_mask)
        # print(state['clause_vector'].sum(-1))

        if is_training is True:
            pred_scores = self.label_decoder(state, clause_num_mask, target_labels)
            loss = self.criterion(pred_scores, target_labels, clause_num_mask)
            return loss
        else:
            pred_scores = self.label_decoder(state, clause_num_mask, None)
            return pred_scores

    # 基于word_recovery将sentence切割为多条子句
    @staticmethod
    def clause_segmentation(state, word_recovery, word_recovery_mask):
        bz, _, hs = state['encoder_hs'].size()
        _, mcn, mcl = word_recovery.size()
        new_word_recovery = word_recovery.view(-1, mcn * mcl).unsqueeze(-1).repeat(1, 1, hs)
        encoder_clause_hs = state['encoder_hs'].gather(1, new_word_recovery).view(bz, mcn, mcl, hs)
        state['encoder_clause_hs'] = encoder_clause_hs * word_recovery_mask.unsqueeze(-1).repeat(1, 1, 1, hs)
        return state

    def gcn_encoder(self, state, adj_matrix, word_recovery_mask):
        hs = state['encoder_clause_hs'].size()[-1]
        encoder_GCN_hs = F.relu(self.gc1(state['encoder_clause_hs'], adj_matrix))
        encoder_GCN_hs = F.relu(self.gc2(encoder_GCN_hs, adj_matrix))
        encoder_GCN_hs = torch.cat((state['encoder_clause_hs'], encoder_GCN_hs), -1)
        state['encoder_GCN_hs'] = encoder_GCN_hs * word_recovery_mask.unsqueeze(-1).repeat(1, 1, 1, 2 * hs)
        return state

    def pooling(self, state, word_recovery_mask):
        clause_lens = torch.sum(word_recovery_mask, dim=-1, keepdim=True) + 1e-45
        max_pooling = state['encoder_GCN_hs'].max(-2)[0]
        avg_pooling = state['encoder_GCN_hs'].sum(-2) / clause_lens
        clause_cat_vector = torch.cat((max_pooling, avg_pooling), -1)
        state['clause_vector'] = F.relu(self.pooling_dropout(self.linear_projection(clause_cat_vector)))
        return state

    def self_attention(self, state, clause_num_mask):
        extended_attention_mask = (1.0 - clause_num_mask[:, None, None, :]) * -10000.0
        self_attention_outputs = self.attention(state['clause_vector'], attention_mask=extended_attention_mask)[0]
        intermediate_output = self.intermediate(self_attention_outputs)
        state['clause_vector'] = self.output(intermediate_output, self_attention_outputs)
        return state

    def criterion(self, pred_scores, target_labels, clause_num_mask):
        bz, mcn = clause_num_mask.size()
        BCE_loss_count = 0.
        KL_loss_count = 0.
        # print(pred_scores)
        # print(target_labels)

        for i in range(bz):
            for j in range(mcn):
                if clause_num_mask[i, j] == 1:
                    BCE_loss_count += self.BCE_loss_func(pred_scores[i][j], (target_labels[i][j] > 0).float())
                    if not self.args.standardized_label:
                        pred_logsoftmax_score = torch.log_softmax(pred_scores[i][j], -1)
                        KL_loss_count += self.KL_loss_func(pred_logsoftmax_score, target_labels[i][j])
        if self.args.standardized_label:
            return BCE_loss_count / (clause_num_mask.sum())
        else:
            return KL_loss_count / (clause_num_mask.sum())
            # return (0.3 * BCE_loss_count + 0.7 * KL_loss_count) / (bz * mcn)


if __name__ == '__main__':
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--hidden_size', type=int, default=256)
    # parser.add_argument('--bert_directory', type=str, default="bert-base-chinese")
    # args, unparsed = parser.parse_known_args()
    #
    # model = CMSA(args)
    # for n, p in model.named_parameters():
    #     print(n, p.size())
    # input_ids = torch.ones((4, 60), requires_grad=False).long()
    # input_mask = torch.ones((4, 60), requires_grad=False, dtype=torch.float32)
    # output = model(input_ids, input_mask)
    # print(output.size())
