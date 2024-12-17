"""
# Segmentation Prediction Metrics
"""
import math
import os
from collections import defaultdict

import numpy as np
import sklearn.metrics
import torch


def compute_segmentation_metrics(c: np.ndarray):
    """Computation of metrics between predicted and ground truth depths
    """

    iou = np.diag(c) / (c.sum(1) + c.sum(0) - np.diag(c))

    mean_iou = np.nanmean(iou[1:])

    mean_pixel_acc = np.nanmean((np.diag(c) / c.sum(1))[1:])
    pixel_acc = np.diag(c)[1:].sum() / c[1:,1:].sum()

    return {
        "acc/pixel_acc": pixel_acc,
        "acc/mean_pixel_acc": mean_pixel_acc,
        "acc/iou": mean_iou,
    }


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        self._mask_onehot = torch.tensor([1] + [-1] * (num_classes-1)) * math.inf

    def update_confusion_matrix(self, gt: torch.Tensor, pred: torch.Tensor):

        pred = pred.clone()
        pred[:, :, 0] = -math.inf
        pred = torch.where(gt.unsqueeze(2) > 0, pred, self._mask_onehot.to(pred))
        pred_class = pred.argmax(-1)

        pred_class = pred_class.view(-1).cpu().numpy()
        gt = gt.view(-1).cpu().numpy()
        self.confusion_matrix += sklearn.metrics.confusion_matrix(gt, pred_class, labels=range(self.num_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


class Evaluator(object):
    def __init__(self, num_classes: int):
        # Error and Accuracy metric trackers
        self.metrics = defaultdict(AverageMeter)
        self.best_tracked = {}

        self.confusion_matrix = ConfusionMatrix(num_classes)

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        self.confusion_matrix.reset()
        for key in self.metrics.keys():
            self.metrics[key].reset()

    @torch.no_grad()
    def compute_eval_metrics(self, gt_sem, pred_sem, mask=None, track=True):
        """
        Computes metrics used to evaluate the model
        """
        N = gt_sem.shape[0]

        self.confusion_matrix.update_confusion_matrix(gt_sem, pred_sem)
        assert track == True

    def update_best(self, sem_errors):
        for key, val in sem_errors.items():
            k0, ky = key.split("/")
            if key not in self.best_tracked:
                self.best_tracked[key] = val
            elif k0 == "acc":
                if val > self.best_tracked[key]:
                    self.best_tracked[key] = val
            elif k0 == "err":
                if val < self.best_tracked[key]:
                    self.best_tracked[key] = val
            else:
                assert 0
        return self.best_tracked

    def get_results(self, update_best: bool):
        errors = compute_segmentation_metrics(self.confusion_matrix.confusion_matrix)

        for key, val in errors.items():
            self.metrics[key].update(val, 1)

        if not update_best:
            return errors

        best_errors = self.update_best(errors)
        return errors, best_errors

    def print(self, dir=None):
        avg_metrics = {key: self.metrics[key].avg for key in self.metrics.keys()}

        print("\n| " + ("{:>20} | " * len(avg_metrics)).format(*avg_metrics.keys()))
        print(("| " + "{:20.5f} | " * len(avg_metrics)).format(*avg_metrics.values()))

        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'w') as f:
                print("\n| " + ("{:>20} | " * len(avg_metrics)).format(*avg_metrics.keys()), file=f)
                print(("| " + "{:20.5f} | " * len(avg_metrics)).format(*avg_metrics.values()), file=f)
