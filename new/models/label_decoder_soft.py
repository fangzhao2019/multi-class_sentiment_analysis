import torch
import torch.nn as nn
from transformers.modeling_bert import BertAttention, BertLayerNorm
import torch.nn.functional as F


class LabelDecoder(nn.Module):
    def __init__(self, config, args):
        super(LabelDecoder, self).__init__()
        self.args = args
        self.config = config
        self.label_embedding = nn.Embedding(args.num_of_classes, args.label_embedding_dim)
        config.output_attentions = True
        config.num_attention_heads = 1
        self.attention = BertAttention(config)
        self.input_projection_layer = nn.Linear(args.label_embedding_dim + config.hidden_size * 2, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=args.num_labelDecoder_layers, batch_first=True)
        self.lstm_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, args.num_of_classes)

    def forward(self, state, clause_num_mask, target_labels):
        # clause_vector    [bz, mcn, hs]
        # target_labels    [bz, mcn, nc]

        bz, mcn, hs = state['clause_vector'].size()
        nc = self.args.num_of_classes
        nt = self.args.num_labelDecoder_layers
        label_embed = self.label_embedding.weight.unsqueeze(0).repeat(bz, 1, 1)  # [bz, nc, hs]

        state['decoder_hidden'] = state['clause_vector'].new_zeros(bz, hs)  # [bz, hs]
        state['decoder_hidden_all'] = state['clause_vector'].new_zeros(nt, bz, hs)  # [nt, bz, hs]
        state['decoder_context'] = state['clause_vector'].new_zeros(bz, hs)  # [bz, hs]
        state['decoder_context_all'] = state['clause_vector'].new_zeros(nt, bz, hs)  # [nt, bz, hs]

        dec_labels = clause_num_mask.new_zeros(bz, nc)  # start of triple
        dec_mask = dec_labels > 0
        if self.args.remain_neutral:
            dec_labels[:, -1] = 1

        step_pred_labels = state['clause_vector'].new_zeros(bz, mcn, nc)
        if target_labels is not None:
            for t in range(mcn):
                if t > 0:
                    dec_labels = target_labels[:, t-1]

                state = self._decoder_step(state, dec_labels, clause_num_mask, label_embed, t)
                pred_scores = self.output_layer(state["decoder_hidden"])
                step_pred_labels[:, t, :] = pred_scores
        else:
            for t in range(mcn):
                if t > 0:
                    dec_labels = pred_scores
                state = self._decoder_step(state, dec_labels, clause_num_mask, label_embed, t)
                pred_scores = self.output_layer(state["decoder_hidden"])
                pred_scores = pred_scores.softmax(-1)
                step_pred_labels[:, t, :] = pred_scores
        return step_pred_labels

    def _decoder_step(self, state, last_predictions, clause_num_mask, label_embed, t):
        bz, mcn, hs = state['clause_vector'].size()
        # label_weight = masked_softmax(last_predictions, dec_mask).unsqueeze(1)
        embedded_input = torch.bmm(last_predictions.unsqueeze(1), label_embed).view(bz, -1)

        s_prev = state["decoder_hidden"].view(bz, 1, hs)
        encoder_extended_attention_mask = (1.0 - clause_num_mask[:, None, None, :]) * -10000.0
        cross_attention_outputs = self.attention(hidden_states=s_prev, encoder_hidden_states=state["clause_vector"], encoder_attention_mask=encoder_extended_attention_mask)
        attentive_read = cross_attention_outputs[0].view(bz, hs)
        decoder_input = torch.cat((embedded_input, attentive_read, state["clause_vector"][:, t, :]), -1)
        projected_decoder_input = self.input_projection_layer(decoder_input)

        _, (state["decoder_hidden_all"], state["decoder_context_all"]) = self.lstm(projected_decoder_input.unsqueeze(1), (state["decoder_hidden_all"], state["decoder_context_all"]))
        state["decoder_hidden"] = self.lstm_dropout(state["decoder_hidden_all"][-1])
        state["decoder_context"] = state["decoder_context_all"][-1]
        return state


def masked_softmax(vector, mask):
    mask = mask.float()
    while mask.dim() < vector.dim():
        mask = mask.unsqueeze(1)
    result = torch.nn.functional.softmax(vector * mask, dim=-1)
    result = result * mask
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result





