import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, hamming_loss


class Metric:
    def __init__(self, args):
        self.args = args

    def metric_all(self, gold, pred_scores):
        # cosine_sim = 0.
        # for i in range(len(gold)):
        #     cosine_sim += self.get_cos_similar(gold[i], pred_scores[i])
        # cosine_sim = cosine_sim / len(gold)
        # print('The Cosine Similarity is %.4f' % cosine_sim)
        # all_metrics = {}
        # all_metrics['macro/f1'] = cosine_sim

        if not self.args.standardized_label:
            gold = np.array(gold > 0, dtype=float)
            pred_scores = self.sigmoid(pred_scores)
        all_metrics = self.calculate_metrics(gold, pred_scores, threshold=0.5)
        std_metrics = self.calculate_metrics(gold[:, :-1], pred_scores[:, :-1], threshold=0.5)
        print('The Detailed Macro Precision is ', '  '.join(['%.3f' % m for m in all_metrics['detailed_mac_p']]))
        print('The Detailed Macro Recall    is ', '  '.join(['%.3f' % m for m in all_metrics['detailed_mac_r']]))
        print('The Detailed Macro F1-Score  is ', '  '.join(['%.3f' % m for m in all_metrics['detailed_mac_f1']]))
        print('The Detailed Average Precision is', '  '.join(['%.3f' % m for m in all_metrics['detailed_ap']]))
        print('All Macro F1-score:  %.3f.  All Micro F1-score:  %.3f. All Hamming Loss:  %.3f.  All Average Precision:  %.3f' %
              (all_metrics['macro/f1'], all_metrics['micro/f1'], all_metrics['hamming_loss'], all_metrics['mAP']))
        print('STD Macro F1-score:  %.3f.  STD Micro F1-score:  %.3f. STD Hamming Loss:  %.3f.  STD Average Precision:  %.3f' %
              (std_metrics['macro/f1'], std_metrics['micro/f1'], std_metrics['hamming_loss'], std_metrics['mAP']))
        return all_metrics

    @staticmethod
    def get_cos_similar(v1, v2):
        num = float(np.dot(v1, v2))  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return (num / denom) if denom != 0 else 0

    def calculate_metrics(self, target, pred_scores, threshold=0.5):
        pred = np.array(pred_scores > threshold, dtype=float)
        hl = hamming_loss(target, pred)
        detailed_ap, mAP = self._compute_AP(target, pred_scores)
        detailed_mac_p = []
        detailed_mac_r = []
        detailed_mac_f1 = []
        for i in range(pred.shape[-1]):
            detailed_mac_p.append(precision_score(y_true=target[:, i], y_pred=pred[:, i], zero_division=0))
            detailed_mac_r.append(recall_score(y_true=target[:, i], y_pred=pred[:, i], zero_division=0))
            detailed_mac_f1.append(f1_score(y_true=target[:, i], y_pred=pred[:, i], zero_division=0))

        return {'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
                'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
                'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
                'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
                'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
                'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
                'hamming_loss': hl,
                'mAP': mAP,
                'detailed_mac_p': detailed_mac_p,
                'detailed_mac_r': detailed_mac_r,
                'detailed_mac_f1': detailed_mac_f1,
                'detailed_ap': detailed_ap
                }

    @staticmethod
    def _compute_AP(gt_labels, pd_probs):
        gt_instances = np.sum(gt_labels, axis=0)
        pd_instances = np.sum(pd_probs, axis=0)
        computed_ap = average_precision_score(gt_labels, pd_probs, average=None)
        actual_ap = []
        num_classes = np.shape(gt_labels)[-1]
        for k in range(num_classes):
            if ((gt_instances[k] != 0) or (pd_instances[k] != 0)) and not np.isnan(computed_ap[k]):
                actual_ap.append(computed_ap[k])
            else:
                actual_ap.append("n/a")
        mAP = np.mean([i for i in actual_ap if i != 'n/a'])
        return actual_ap, mAP

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
