from torch.utils.data import DataLoader
from local.file_dataset import *
from torch import nn as nn
import step1_config
from thop import profile
import numpy as np

##  ----------------------modify---------------------------------#
test_list = r'./csv/isic2018_test_list.csv'

num_channels = 3
num_workers = 2
weight_path = r'./weight2018/MaxAtten_TokMlp_2018.pth'
# weight_path = r'./weight2018/A100_40G/CaraNet_2018.pth'

device = torch.device('cuda:0')

from loss.loss_cara import *
# from model.CaraNet.CaraNet import caranet
# from model.CaraNet.CaraNet_MAatten import CaraNet_MAatten
# from model.CaraNet.CaraNet_MAatten_DLF import CaraNet_MAatten_DLF
# from model.CaraNet.CaraNet_Gridatten import CaraNet_Gridatten
# from model.CaraNet.CaraNet_Blockatten import CaraNet_Blockatten
# from model.experiment.Vit_Cnn import Vit_Cnn
# from model.experiment.Vit_Cnn_RA import Vit_Cnn_RA
# from model.experiment.VitBRA_Cnn import VitBRA_Cnn
# from model.experiment.lib.TransNeXt import transnext_tiny
from model.experiment.Vit_Cnn_PD import Vit_Cnn_PD

model = Vit_Cnn_PD()
model = model.to(device=device)
# model.load_state_dict(torch.load(weight_path))

input = torch.randn(1, 3, 512, 512)
input = input.to(device=device)
flops, params = profile(model, inputs=(input,))
print('flops:{}'.format(flops))
print('params:{}'.format(params))

# ##  ----------------end---modify---------------------------------#
model_path = step1_config.args.model_path
model_name = step1_config.args.model_name  # remember to include the project title (e.g., ALV)
checkpoint_name = step1_config.args.checkpoint_name

test_dataset = isic_dataset(test_list)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=2,
                         shuffle=True,
                         num_workers=num_workers,
                         pin_memory=True)

print('Testing model...')
class_weights = torch.Tensor([1, 2]).to(device, dtype=torch.float)

loss_epoch = 0.0
dice_epoch = 0.0
miou_epoch = 0.0
acc_epoch = 0.0
f1_epoch = 0.0

for i_batch, batched_sample in enumerate(test_loader):
    inputs, labels = batched_sample['image'].to(device, dtype=torch.float), \
                     batched_sample['label'].to(device, dtype=torch.float)
    original_lables = labels

    labels = torch.ceil((labels.float() + 255) / 255) - 1.0
    # labels[labels > 0] = 1
    bat_num = labels.shape[0]
    label = torch.reshape(labels, [bat_num, 512, 512])
    one_hot_label = nn.functional.one_hot(label.long(), num_classes=2)  # lastet dimension is channel
    one_hot_label = one_hot_label.permute(0, 3, 1, 2).contiguous()
    # print('one_hot_label=', one_hot_label)

    # lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1 = model(inputs.float())
    # outputs = lateral_map_5
    outputs = model(inputs.float())

    output = torch.ceil((outputs.float() + 255) / 255) - 1.0
    # labels[labels > 0] = 1
    bn = output.shape[0]
    output = torch.reshape(output, [bn, 512, 512])
    one_hot_output = nn.functional.one_hot(output.long(), num_classes=2)  # lastet dimension is channel
    one_hot_output = one_hot_output.permute(0, 3, 1, 2).contiguous()

    # ---- loss function ----
    # loss5 = structure_loss(lateral_map_5, original_lables)
    # loss3 = structure_loss(lateral_map_3, original_lables)
    # loss2 = structure_loss(lateral_map_2, original_lables)
    # loss1 = structure_loss(lateral_map_1, original_lables)
    # loss = loss5 + loss3 + loss2 + loss1

    loss = structure_loss(outputs, original_lables)
    # -------end moidfy
    dice = unet_dice(one_hot_output, one_hot_label, class_weights)
    acc, miou, f1 = make_miou(original_lables, outputs)

    loss_epoch += loss.item()
    dice_epoch += dice.item()
    miou_epoch += miou
    acc_epoch += acc
    f1_epoch += f1

test_loss = loss_epoch / len(test_loader)
test_dice = dice_epoch / len(test_loader)
test_miou = miou_epoch / len(test_loader)
test_acc = acc_epoch / len(test_loader)
test_f1 = f1_epoch / len(test_loader)

print('test_loss=', test_loss, 'test_dice=', test_dice, 'test_miou=', test_miou, 'test_acc=',
      test_acc, 'test_f1=', test_f1)
