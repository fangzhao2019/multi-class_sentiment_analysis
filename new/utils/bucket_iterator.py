# -*- coding: utf-8 -*-
import math
import random
import torch
import numpy as np


class CustomTextDataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class BucketIterator(object):
    def __init__(self, data, batch_size, shuffle=False, sort=False):
        self.shuffle = shuffle
        self.sort = sort
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x['token_cut_id']))
            sorted_data = sorted(sorted_data, key=lambda x: x['num_of_clauses'])
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.padding_data(sorted_data[i * batch_size: (i+1) * batch_size]))
        return batches

    def padding_data(self, batch_data):
        bz = len(batch_data)
        num_of_classes = batch_data[0]['labels'].shape[-1]

        sent_lens = [len(d['token_cut_id']) for d in batch_data]
        max_sent_len = max(sent_lens)

        max_clause_num = max([d['num_of_clauses'] for d in batch_data])

        clause_lens = [list(map(len, d['word_recovery'])) for d in batch_data]
        max_clause_len = max([max(c) for c in clause_lens])

        batch_source_ids = torch.zeros((bz, max_sent_len), requires_grad=False).long()
        batch_source_mask = torch.zeros((bz, max_sent_len), requires_grad=False, dtype=torch.float32)
        batch_clause_num_mask = torch.zeros((bz, max_clause_num), requires_grad=False, dtype=torch.float32)
        batch_word_recovery = torch.zeros((bz, max_clause_num, max_clause_len), requires_grad=False).long()
        batch_word_recovery_mask = torch.zeros((bz, max_clause_num, max_clause_len), requires_grad=False, dtype=torch.float32)
        batch_adj_matrix = torch.zeros((bz, max_clause_num, max_clause_len, max_clause_len), requires_grad=False, dtype=torch.float32)
        batch_labels = torch.zeros((bz, max_clause_num, num_of_classes), requires_grad=False, dtype=torch.float32)
        for idx, (item, sent_len, ce_lens) in enumerate(zip(batch_data, sent_lens, clause_lens)):
            token_cut_id, num_of_clauses, word_recovery, adj_matrix, labels = item['token_cut_id'], \
                         item['num_of_clauses'], item['word_recovery'], item['adj_matrix'], item['labels']
            batch_source_ids[idx, :sent_len] = torch.LongTensor(token_cut_id)
            batch_source_mask[idx, :sent_len] = torch.FloatTensor([1] * sent_len)
            for idy, (word_rec, matrix, label, ce_len) in enumerate(zip(word_recovery, adj_matrix, labels, ce_lens)):
                batch_clause_num_mask[idx, idy] = 1.
                batch_word_recovery[idx, idy, :ce_len] = torch.LongTensor(word_rec)
                batch_word_recovery_mask[idx, idy, :ce_len] = torch.FloatTensor([1] * ce_len)
                batch_adj_matrix[idx, idy, :ce_len, :ce_len] = torch.LongTensor(matrix)
                batch_labels[idx, idy, :] = torch.FloatTensor(label)
        return {'source_ids': batch_source_ids, 'source_mask': batch_source_mask, 'clause_num_mask': batch_clause_num_mask,
                'word_recovery': batch_word_recovery, 'word_recovery_mask': batch_word_recovery_mask,
                'adj_matrix': batch_adj_matrix, 'target_labels': batch_labels}

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
