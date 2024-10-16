import torch
import numpy as np
from torchvision.utils import save_image
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score
from scipy.spatial.distance import directed_hausdorff
from medpy import metric
from torch import nn as nn
from torch.nn import *


device = torch.device('cuda:0')


def make_miou(ture, pred):
    n_batch=ture.shape[0]
    iou=0
    acc=0
    f1=0
    for i in range(n_batch):
        image=torch.unsqueeze(pred[i][1],dim=0)
        image=torch.ceil((image.float()+255)/255)-1.0

        image=torch.flatten(image)
        image=image.cpu().detach().numpy()
        label=ture[i]
        label = torch.ceil((label.float()+255)/255)-1.0
        label=torch.flatten(label)
        label=label.cpu().numpy()
        a=confusion_matrix(label,image)
        f1 += f1_score(y_true=label,y_pred=image)
        iou+=np.diagonal(a)/(a.sum(1)+a.sum(0)-np.diagonal(a))
        acc+=np.diagonal(a).sum()/a.sum()
    return acc/n_batch, iou.mean()/n_batch,f1/n_batch


def unet_dice(y_pred_orig, y_true, class_weights):

    smooth = 1.
    mdsc  = 0.0
    mdsc0 = 0.0
    n_classes = y_pred_orig.shape[1] # for PyTorch data format
    # print("-----------go 18-------------")

    # convert probability to one-hot code
    y_pred=y_pred_orig   # x
    ###########################################
    max_idx = torch.argmax(y_pred, dim=1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        intersection = (pred_flat.float() * true_flat.float()).sum()
        w = class_weights[c]/class_weights.sum()
        mdsc0 += w*((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))

    mdsc=mdsc0

    return mdsc


def unet_loss(y_pred, y_true):
    smooth = 1.
    loss0 = 0.

    n_classes = y_pred.shape[1]
    # loss_fn = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none').to(device=device)

    for c in range(0, n_classes):  # pass 0 because 0 is background
        pred_flat = y_pred[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1).float()
        loss0 = loss0 + loss_fn(pred_flat, true_flat)

    nums = loss0.size(0)
    return loss0.sum() / nums
##########################################