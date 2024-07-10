import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:  Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.05, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        if targets.ndim == 1:
            targets = torch.zeros(log_probs.size()).scatter_(
                1,
                targets.unsqueeze(1).cpu(), 1)

            if torch.cuda.is_available(): targets = targets.cuda()
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        elif targets.ndim == 2:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class CDTLoss(nn.Module):

    def __init__(self, Delta_list, gamma=0.5, weight=None, reduction=None, device = None):
        super(CDTLoss, self).__init__()
        self.gamma = gamma
        self.Delta_list = torch.pow(torch.FloatTensor(Delta_list), self.gamma)
        self.Delta_list = self.Delta_list.shape[0] * self.Delta_list / torch.sum(self.Delta_list)
        self.Delta_list = self.Delta_list.to(device)
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        if self.reduction == "sum":
            output = x * self.Delta_list
            return F.cross_entropy(output, target, weight=self.weight, reduction='sum')
        else:
            output = x * self.Delta_list
            return F.cross_entropy(output, target, weight=self.weight)


class LDTLoss(nn.Module):

    def __init__(self, Delta_list, gamma = 0.5, weight=None, reduction = None, device = None):
        super(LDTLoss, self).__init__()
        self.gamma = gamma
        self.Delta_list = torch.pow(torch.FloatTensor(Delta_list), self.gamma)
        self.Delta_list = self.Delta_list.shape[0] * self.Delta_list / torch.sum(self.Delta_list)
        self.Delta_list = self.Delta_list.to(device)
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        if self.reduction == "sum":
            ldt_output = (x.T*self.Delta_list[target]).T
            return F.cross_entropy(ldt_output, target, weight=self.weight, reduction = 'sum')
        else:
            ldt_output = (x.T*self.Delta_list[target]).T
            return F.cross_entropy(ldt_output, target, weight=self.weight)


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self,
                 s,
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0,
                 eps = 0):
        super().__init__()
        self.eps = eps
        self.s = s
        self.m1 = m1  # 1
        self.m2 = m2  # 0.5
        self.m3 = m3  # 0
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    def forward(self, logits, labels):  # this logit is cosine value of the angle
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0 and self.m2 > 0:
            with torch.no_grad():
                target_logit.arccos_()  # angle for target class  \theta_y
                logits.arccos_()        # angle for all classes   \theta_j
                final_target_logit = target_logit + self.m2  # \theta_y + m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit  # update \theta_y in \theta vector
                logits.cos_()
            logits = logits * self.s  # s*cos(\theta_j)  (j=y: target class different)

        elif self.m1 == 1.0 and self.m3 == 0.0 and self.m2 == 0.0:
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        if self.eps > 0:
            criterion = CrossEntropyLabelSmooth(logits.shape[-1], epsilon=self.eps)
        elif self.eps == -1:
            K = logits.shape[-1]
            eps = np.exp(-1 / (K - 1) * self.s) / (np.exp(1 * self.s) + (K - 1) * np.exp(-1 / (K - 1) * self.s)) * K
            criterion = CrossEntropyLabelSmooth(logits.shape[-1], epsilon=eps)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        # logits still need to go through SoftMax for cross entropy loss
        return criterion(logits, labels)
