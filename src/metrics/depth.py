"""
# Depth Prediction Metrics
"""
import os
from collections import defaultdict

import torch


MAX_DEPTH_METERS = 10

def compute_depth_metrics(gt: torch.Tensor, pred: torch.Tensor, mask=None):
    """Computation of metrics between predicted and ground truth depths
    """
    gt_depth = gt * MAX_DEPTH_METERS
    pred_depth = pred * MAX_DEPTH_METERS

    gt_depth = gt_depth.clip(0.01, MAX_DEPTH_METERS)
    pred_depth = pred_depth.clip(0.01, MAX_DEPTH_METERS)

    ###########STEP 1: compute delta#######################
    thresh = torch.max((gt_depth / pred_depth), (pred_depth / gt_depth))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    ##########STEP 2:compute mean error###################
    error = gt_depth - pred_depth
    mae = error.abs().mean()
    mre = error.abs().div(gt_depth).mean()


    return {
        "err/mae": mae,
        "err/mre": mre,
        "acc/a1": a1,
        "acc/a2": a2,
        "acc/a3": a3,
    }


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
    def __init__(self):
        # Error and Accuracy metric trackers
        self.metrics = defaultdict(AverageMeter)
        self.best_tracked = {}

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        for key in self.metrics.keys():
            self.metrics[key].reset()

    @torch.no_grad()
    def compute_eval_metrics(self, gt_depth, pred_depth, mask=None, track=True):
        """
        Computes metrics used to evaluate the model
        """
        N = gt_depth.shape[0]

        depth_errors = compute_depth_metrics(gt_depth, pred_depth, mask)

        if track:
            for key, val in depth_errors.items():
                self.metrics[key].update(val, N)
        else:
            return depth_errors

    def update_best(self, depth_errors):
        for key, val in depth_errors.items():
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

    def get_results(self):
        depth_errors = {key: val.avg for key, val in self.metrics.items()}
        best_errors = self.update_best(depth_errors)
        return depth_errors, best_errors

    def print(self, dir=None):
        avg_metrics = {key: self.metrics[key].avg for key in self.metrics.keys()}

        print("\n| " + ("{:>13} | " * len(avg_metrics)).format(*avg_metrics.keys()))
        print(("| " + "{:13.5f} | " * len(avg_metrics)).format(*avg_metrics.values()))

        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'w') as f:
                print("\n| " + ("{:>13} | " * len(avg_metrics)).format(*avg_metrics.keys()), file=f)
                print(("| " + "{:13.5f} | " * len(avg_metrics)).format(*avg_metrics.values()), file=f)
