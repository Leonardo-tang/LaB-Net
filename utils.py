def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr


import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEdgeLoss(nn.Module):
    """
    简单的边缘损失 - 使用拉普拉斯核检测边缘
    """

    def __init__(self, weight=0.05):
        super(SimpleEdgeLoss, self).__init__()
        self.weight = weight

        # 拉普拉斯核（边缘检测）
        self.laplace_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)

    def compute_edges(self, x):
        """计算边缘图"""
        # 使用反射填充来处理边界
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')

        # 应用拉普拉斯卷积
        edges = F.conv2d(x_padded, self.laplace_kernel.to(x.device))

        # 取绝对值，计算边缘强度
        edges = torch.abs(edges)

        # 归一化到0-1范围
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-6)

        return edges

    def forward(self, pred, target):
        """
        计算边缘损失
        pred: 预测图 [B, 1, H, W] (logits)
        target: 真实图 [B, 1, H, W] (0-1)
        """
        # 将logits转为概率图
        pred_prob = torch.sigmoid(pred)

        # 计算预测图的边缘
        pred_edges = self.compute_edges(pred_prob)

        # 计算真实图的边缘
        target_edges = self.compute_edges(target)

        # 计算L1损失（对边缘更敏感）
        edge_loss = F.l1_loss(pred_edges, target_edges)

        return edge_loss * self.weight


class EdgeAwareCombinedLoss(nn.Module):
    """
    组合损失函数：BCE + IoU + 边缘损失
    """

    def __init__(self, edge_weight=0.05, iou_weight=0.5, weights=[0.4, 0.3, 0.2, 0.1]):
        super(EdgeAwareCombinedLoss, self).__init__()
        self.weights = weights
        self.edge_weight = edge_weight
        self.iou_weight = iou_weight

        # 基础损失函数
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.edge_loss = SimpleEdgeLoss(weight=edge_weight)

    def iou_loss(self, pred, target):
        """IoU损失"""
        pred = torch.sigmoid(pred)
        smooth = 1e-6

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + smooth) / (union + smooth)

        return 1 - iou.mean()

    def forward(self, preds, target):
        s4, s3, s1, s2 = preds

        # 计算各尺度损失
        loss1 = self.weights[0] * (
                self.bce_loss(s1, target) +
                self.iou_weight * self.iou_loss(s1, target) +
                self.edge_weight * self.edge_loss(s1, target)
        )

        loss2 = self.weights[1] * (
                self.bce_loss(s2, target) +
                self.iou_weight * self.iou_loss(s2, target) +
                self.edge_weight * self.edge_loss(s2, target)
        )

        loss3 = self.weights[2] * (
                self.bce_loss(s3, target) +
                self.iou_weight * self.iou_loss(s3, target) +
                self.edge_weight * self.edge_loss(s3, target)
        )

        loss4 = self.weights[3] * (
                self.bce_loss(s4, target) +
                self.iou_weight * self.iou_loss(s4, target) +
                self.edge_weight * self.edge_loss(s4, target)
        )

        total_loss = loss1 + loss2 + loss3 + loss4

        return total_loss, loss1, loss2, loss3, loss4