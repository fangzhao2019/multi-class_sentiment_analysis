import torch.nn as nn
import torch
from transformers import BertModel


class CMSA(nn.Module):
    def __init__(self, args):
        super(CMSA, self).__init__()
        self.args = args

        # bert嵌入层
        self.bert = BertModel.from_pretrained(args.bert_directory)
        if args.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
        self.config = self.bert.config

        # 线性编码层（降维）
        self.feedforward_encoder_layer = nn.Linear(self.config.hidden_size, args.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, source_token_ids, source_mask, target_label_ids):
        encoder_hs = self.feedforward_encoder(source_token_ids, source_mask)

    def feedforward_encoder(self, source_token_ids, source_mask):
        last_hidden_state, _ = self.bert(source_token_ids, attention_mask=source_mask)
        encoder_hs = self.feedforward_encoder_layer(last_hidden_state)
        encoder_hs = self.dropout(self.activation(encoder_hs))
        return encoder_hs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--bert_directory', type=str, default="bert-base-chinese")
    args, unparsed = parser.parse_known_args()

    model = CMSA(args)
    for n, p in model.named_parameters():
        print(n, p.size())
    input_ids = torch.ones((4, 60), requires_grad=False).long()
    input_mask = torch.ones((4, 60), requires_grad=False, dtype=torch.float32)
    output = model(input_ids, input_mask)
    print(output.size())

