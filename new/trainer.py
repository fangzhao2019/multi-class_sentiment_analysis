import gc
import os
import time
import torch
import numpy as np
import random
from tqdm import tqdm
from torch import nn, optim
from transformers import AdamW
from utils.average_meter import AverageMeter
from utils.bucket_iterator import CustomTextDataset, BucketIterator


class Trainer(nn.Module):
    def __init__(self, args, data, model, metric, device):
        super().__init__()
        self.args = args
        self.data = data
        self.model = model
        self.metric = metric
        self.device = device

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['encoder', 'decoder']
        grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and component[0] not in n],
                'weight_decay': args.weight_decay,
                'lr': args.decoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and component[0] not in n],
                'weight_decay': 0.0,
                'lr': args.decoder_lr
            }
        ]
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(grouped_params)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(grouped_params)
        else:
            raise Exception("Invalid optimizer.")
        if args.use_gpu:
            self.cuda()

    def train_model(self):
        best_f1 = [0, 0]
        best_result_epoch = -1
        start_epoch = -1
        start_batch = -1

        count = 2
        batch_size = self.args.batch_size
        path_checkpoint = '%s%s_id_%d.pth.tar' % (self.args.checkpoint_path, self.args.dataset_name, count)
        if self.args.resume and os.path.exists(path_checkpoint):
            print('resume from checkpoint!')
            checkpoint = torch.load(path_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'batch_id' in checkpoint:
                start_batch = checkpoint['batch_id']
            print('start_epoch:', start_epoch, 'start_batch:', start_batch)
        new_start_epoch = 0 if start_epoch == -1 else start_epoch

        # np.save('%s_emotion_vector.npy' % self.args.dataset_name, self.model.label_decoder.label_embedding.weight.cpu().detach().numpy())
        # aaa

        epoch_start_time = time.time()
        batch_start_time = time.time()
        for epoch in range(new_start_epoch, self.args.max_epoch):
            # Train
            self.model.train()
            self.model.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)
            print("=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()

            train_data = CustomTextDataset(self.data.train_dataset)
            # train_data = CustomTextDataset(self.data.test_dataset)
            train_dataloader = BucketIterator(data=train_data, batch_size=batch_size, shuffle=True, sort=True)

            for batch_id, batch_data in tqdm(enumerate(train_dataloader)):
                # break
                if epoch == new_start_epoch:
                    if batch_id < start_batch + 1:
                        continue

                source_ids, source_mask, clause_num_mask, word_recovery, word_recovery_mask, adj_matrix, target_labels = map(lambda x: batch_data[x].to(self.device), batch_data)
                try:
                    loss = self.model(source_ids, source_mask, clause_num_mask, word_recovery, word_recovery_mask, adj_matrix, target_labels)
                    loss = loss.mean()

                    avg_loss.update(loss.item(), 1)
                    loss.backward()
                except RuntimeError as exception:
                    if 'out of memory' in str(exception):
                        tqdm.write('WARNING: OUT OF MEMORY  at  %d. INPUT SIZE is %s' % (batch_id, str(word_recovery.size())))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception
                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if batch_id % 200 == 0 and batch_id != 0:
                    batch_time = time.time()
                    tqdm.write("     Instance: %d; loss: %.6f, speed: %d s" % (batch_id * batch_size, avg_loss.avg, batch_time - batch_start_time))
                    batch_start_time = batch_time
                if (batch_id * batch_size) % (625 * batch_size) == 0 and batch_id != 0 and self.args.save_mp is True:
                    # count += 1
                    tqdm.write('--- save checkpoint ---,speed time %d second' % (time.time() - epoch_start_time))
                    checkpoint_dict = {'epoch': epoch,
                                       'batch_id': batch_id,
                                       'model_state_dict': self.model.state_dict(),
                                       }
                    torch.save(checkpoint_dict, '%s%s_id_%d.pth.tar' % (self.args.checkpoint_path, self.args.dataset_name, count))
            gc.collect()
            torch.cuda.empty_cache()
            print('the average loss of epoch %d is %.6f, training time %d second\n' % (epoch, avg_loss.avg, time.time() - epoch_start_time))

            # Validation
            print("=== Epoch %d Validation ===" % epoch)
            valid_result = self.eval_model(self.data.valid_dataset)
            valid_f1 = valid_result['macro/f1']
            # Test
            print("=== Epoch %d Test ===" % epoch, flush=True)
            test_result = self.eval_model(self.data.test_dataset)
            test_f1 = test_result['macro/f1']

            if self.args.save_txt is True:
                self.save_to_txt(int(epoch), (valid_result['macro/f1'], valid_result['micro/f1'], valid_result['hamming_loss'], valid_result['mAP']
                                              , test_result['macro/f1'], test_result['micro/f1'], test_result['hamming_loss'], test_result['mAP'])
                                 , 'data/%s_%d_test_result.txt' % (self.args.dataset_name, count))
            if valid_f1 > best_f1[0] and self.args.save_mp is True:
                print("Achieving Best Result on Valid Set.", flush=True)
                best_f1[0] = valid_f1
                best_f1[1] = test_f1
                best_result_epoch = epoch
                # checkpoint_dict = {'epoch': epoch,
                #                    'model_state_dict': self.model.state_dict()
                #                    }
                # best_model_path = '%s%s_epoch_%d_%.4f.pth.tar' % (self.args.best_param_directory, self.args.dataset_name
                #                                                   , epoch, valid_f1)
                # torch.save(checkpoint_dict, best_model_path)

            gc.collect()
            torch.cuda.empty_cache()
            epoch_time = time.time()
            print('this epoch speed time %d second\n\n' % (epoch_time - epoch_start_time))
            epoch_start_time = epoch_time

        # checkpoint = torch.load(best_model_path)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # Test
        # print("=== Test ===", flush=True)
        # test_result = self.eval_model(self.data.test_dataset)
        print("Best result: epoch %d   valid set %.4f   test set %.4f \n" % (best_result_epoch, best_f1[0], best_f1[1]), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    def eval_model(self, eval_dataset):
        batch_size = self.args.test_batch_size
        self.model.eval()
        gold = []
        pred = []
        with torch.no_grad():
            eval_data = CustomTextDataset(eval_dataset)
            eval_dataloader = BucketIterator(data=eval_data, batch_size=batch_size, shuffle=False, sort=False)
            for batch_id, batch_data in tqdm(enumerate(eval_dataloader)):
                source_ids, source_mask, clause_num_mask, word_recovery, word_recovery_mask, adj_matrix, target_labels = map(lambda x: batch_data[x].to(self.device), batch_data)
                pred_scores = self.model(source_ids, source_mask, clause_num_mask, word_recovery, word_recovery_mask, adj_matrix, target_labels, False)
                gold.extend(self.formulate(target_labels, clause_num_mask))
                pred.extend(self.formulate(pred_scores, clause_num_mask))
                assert len(gold) == len(pred)
        result = self.metric.metric_all(np.array(gold), np.array(pred))
        return result

    @staticmethod
    def save_to_txt(x, y, filename):
        f = open(filename, 'a+', encoding='utf-8')
        f.write('%d' % x)
        for j in range(len(y)):
            f.write('   %.4f' % y[j])
        f.write('\n')
        f.close()

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer

    @staticmethod
    def formulate(labels, clause_num_mask):
        labels = labels.cpu().detach().numpy()
        tag_set = []
        bz, mcn, _ = labels.shape
        for i in range(bz):
            for j in range(mcn):
                if clause_num_mask[i, j] == 1:
                    tag_set.append(labels[i, j])
        return tag_set

