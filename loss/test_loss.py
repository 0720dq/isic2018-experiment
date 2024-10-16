import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import torch.nn.functional as F

device = torch.device('cuda:0')


def make_miou(ture, pred):
    n_batch = ture.shape[0]
    iou = 0
    acc = 0
    f1 = 0
    for i in range(n_batch):
        image = torch.unsqueeze(pred[i][0], dim=0)
        image = torch.ceil((image.float()+255)/255)-1.0

        image = torch.flatten(image)
        image = image.cpu().detach().numpy()
        label = ture[i]
        label = torch.ceil((label.float()+255)/255)-1.0
        label = torch.flatten(label)
        label = label.cpu().numpy()
        a = confusion_matrix(label, image)
        f1 += f1_score(y_true=label, y_pred=image)
        iou += np.diagonal(a)/(a.sum(1)+a.sum(0)-np.diagonal(a))
        acc += np.diagonal(a).sum()/a.sum()
    return acc/n_batch, iou.mean()/n_batch, f1/n_batch


def unet_dice(y_pred_orig, y_true):

    smooth = 1.
    mdsc0 = 0.0
    # print("-----------go 18-------------")

    # convert probability to one-hot code
    y_pred = y_pred_orig  # x
    ###########################################
    y_pred[y_pred > 0.5] = 1  # 将概率输出变为于标签相匹配的矩阵
    y_pred[y_pred <= 0.5] = 0

    pred_flat = y_pred.reshape(-1)
    true_flat = y_true.reshape(-1)
    intersection = (pred_flat.float() * true_flat.float()).sum()
    mdsc0 = ((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))

    mdsc=mdsc0

    return mdsc


# def unet_dice(y_pred_orig, y_true, class_weights):
#
#     smooth = 1.
#     mdsc = 0.0
#     mdsc0 = 0.0
#     n_classes = y_pred_orig.shape[1]  # for PyTorch data format
#     # print("-----------go 18-------------")
#
#     # convert probability to one-hot code
#     y_pred = y_pred_orig  # x
#     ###########################################
#     max_idx = torch.argmax(y_pred, dim=1, keepdim=True)
#     one_hot = torch.zeros_like(y_pred)
#     one_hot.scatter_(1, max_idx, 1)
#
#     for c in range(0, n_classes):
#         pred_flat = one_hot[:, c].reshape(-1)
#         true_flat = y_true[:, c].reshape(-1)
#         intersection = (pred_flat.float() * true_flat.float()).sum()
#         # w = class_weights[c]/class_weights.sum()
#         mdsc0 += ((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))
#
#     mdsc=mdsc0
#
#     return mdsc


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def main():
    ture = torch.rand(4, 1, 512, 512)
    pred = torch.rand(4, 1, 512, 512)
    y = make_miou(ture, pred)
    dice = unet_dice(pred, ture)

    print(y)
    print(dice)


if __name__ == '__main__':
    main()