import torch


def batchify_with_label(input_batch_list, gpu, padding_label, MAX_SENTENCE_LENTH):
    batch_size = len(input_batch_list)
    clauses = [sent[0] for sent in input_batch_list]
    labels = [sent[1] for sent in input_batch_list]#[[labelsid],[labelsid]...[labelsid]]
    clause_number = list(map(len, clauses))  # 得到batch中每条数据的子句数
    max_clause_number = max(clause_number)  # batch中包含最多子句的数量
    clause_tensor = torch.zeros((batch_size, max_clause_number, MAX_SENTENCE_LENTH), requires_grad=False).long()
    label_seq_tensor = torch.ones((batch_size, max_clause_number), requires_grad=False).long()
    #######默认是1
    label_seq_tensor = padding_label * label_seq_tensor###
    mask = torch.zeros((batch_size, max_clause_number), requires_grad=False).byte()#标记是否有数据，有为1，无为0
    for idx, (seq, label, seqlen) in enumerate(zip(clauses, labels, clause_number)):
        for idy in range(seqlen):
            word_lenth=len(seq[idy])
            clause_tensor[idx, idy, :word_lenth] = torch.LongTensor(seq[idy])
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)

    clause_number = torch.LongTensor(clause_number)
    clause_number, word_perm_idx = clause_number.sort(0, descending=True)#按照句子数量进行排序
    clause_tensor = clause_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    #在原始序列中每个句子的长度排名
    _, clause_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        clause_tensor = clause_tensor.cuda()
        clause_number = clause_number.cuda()
        clause_recover = clause_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
    return clause_tensor, clause_number, clause_recover, label_seq_tensor, mask