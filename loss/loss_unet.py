import torch
import numpy as np
import numpy
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import *
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0，1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')

def make_miou(ture, pred):
    n_batch=ture.shape[0]
    iou=0
    acc=0
    f1=0
    for i in range(n_batch):
        image = torch.unsqueeze(pred[i][1], dim=0)
        image = torch.ceil((image.float()+255)/255)-1.0
        image = torch.flatten(image)
        image = image.cpu().detach().numpy()
        label = ture[i]
        label = torch.ceil((label.float()+255)/255)-1.0
        label = torch.flatten(label)
        label = label.cpu().numpy()
        a = confusion_matrix(label,image)
        f1 += f1_score(y_true=label, y_pred=image)
        iou += np.diagonal(a)/(a.sum(1)+a.sum(0)-np.diagonal(a))
        acc += np.diagonal(a).sum()/a.sum()
    return acc/n_batch, iou.mean()/n_batch, f1/n_batch


def unet_dice(y_pred_orig, y_true, class_weights, smooth = 1.0):

    smooth = 1.
    mdsc  = 0.0
    mdsc0 = 0.0
    n_classes = y_pred_orig.shape[1]  # for PyTorch data format
    # print("-----------go 18-------------")
    # print(n_classes)

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


# device = torch.device('cuda')
class EdgeEnhancementLoss(nn.Module):
    def __init__(self, weight=0.2):
        super(EdgeEnhancementLoss, self).__init__()
        self.weight = weight
        self.conv = torch.nn.Conv2d(1, 1, (3, 3), stride=1, padding=1, bias=False)
        self.conv.weight.data = torch.Tensor([[[[-1., 0., -1.],
                                              [0., 4., 0.],
                                              [-1., 0., -1.]]]])
        self.conv1 = torch.nn.Conv2d(2, 1, 3, stride=1, padding=1, bias=False)
        self.conv1.weight.data = torch.Tensor([[[[-1., 0., -1.],
                                               [0., 4., 0.],
                                               [-1., 0., -1.]],

                                              [[-1., 0., -1.],
                                               [0., 4., 0.],
                                               [-1., 0., -1.]]]])
        # self.conv = torch.nn.DataParallel(self.conv, device_ids=[0, 1])
        self.conv.to(device=device)
        # self.conv1 = torch.nn.DataParallel(self.conv1, device_ids=[0, 1])
        self.conv1.to(device=device)

    def forward(self, image, lable):

        g_image = self.conv1(image.float())
        g_lable = self.conv(lable.float())
        
        loss = F.mse_loss(g_image[0], g_lable[0], reduction='none')
        loss = self.weight * torch.mean(loss)

        return loss

def unet_loss(y_pred, y_true, y_orig):
    loss0 = 0.
    loss1 = 0
    weight = 0.2

    device = torch.device('cuda')
    n_classes = y_pred.shape[1]
    batch_num = y_pred.shape[0]
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    # loss_fn = torch.nn.DataParallel(loss_fn, device_ids=[0, 1])
    loss_fn = loss_fn.to(device=device)
    for i in range(batch_num):
        lable = y_orig[i]
        lable = torch.unsqueeze(lable, dim=0)
        image = torch.unsqueeze(y_pred[i], dim=0)
        loss1_fn1 = EdgeEnhancementLoss(weight=weight)
        loss1 += loss1_fn1(image, lable)
    
    for c in range(0, n_classes):  # pass 0 because 0 is background

        pred_flat = y_pred[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1).float()

        loss0 += loss_fn(pred_flat, true_flat.float())
  
    nums = loss0.size(0)
    
    return (loss0.sum())/nums*(1.0-weight)+loss1/(batch_num)
    # return (loss0.sum())/nums*0.3+loss1*0.7
##########################################