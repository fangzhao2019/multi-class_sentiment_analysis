import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, RobertaModel, BertConfig


class SeqEncoder(nn.Module):
    def __init__(self, args):
        super(SeqEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_directory)
        if args.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
        self.config = self.bert.config
        self.linear_projection_layer = nn.Linear(self.config.hidden_size, args.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, source_ids, source_mask):
        last_hidden_state, _ = self.bert(source_ids, attention_mask=source_mask)
        encoder_hidden_states = self.linear_projection_layer(last_hidden_state)
        encoder_hidden_states = F.relu(self.dropout(encoder_hidden_states))
        return {'encoder_hs': encoder_hidden_states, 'source_mask': source_mask}
