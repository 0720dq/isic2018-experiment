# python -m visdom.server
import torch
from local.utils import *
from torch.utils.data import DataLoader
import torch.optim as optim
from local.file_dataset import *
import pandas as pd
import step1_config

##  ----------------------modify---------------------------------#
use_visdom = True

train_list = r'./csv/isic2018_train_list.csv'
valid_list = r'./csv/isic2018_valid_list.csv'
train_lr_loss_path = './csv/train_valid_data_isic2018/train/MaxAtten_TokMlp_CFP_Lsls_loss.csv'
valid_loss_path = './csv/train_valid_data_isic2018/valid/loss/MaxAtten_TokMlp_CFP_Lsls_loss.csv'
valid_metric_path = './csv/train_valid_data_isic2018/valid/metric/MaxAtten_TokMlp_CFP_Lsls_metric.csv'
# df = pd.DataFrame(columns=['test_dice', 'test_miou', 'test_acc', 'test_f1'])
# df.to_csv(valid_metric, index=False)

# num_classes = 2
num_channels = 3
num_epochs = 100
num_workers = 2
train_batch_size = 1
num_batches_to_print = 10
lr = 0.0001
test_epoch = 1
weight_path = r'./weight2018/MaxAtten_TokMlp_CFP_Lsls.pth'

device = torch.device('cuda:0')

from loss.loss_cara import *
from loss.SLSIoULoss import *
from model.experiment.MaxAtten_TokMlp_CFP_Lsls import MaxAtten_TokMlp_CFP_Lsls

model = MaxAtten_TokMlp_CFP_Lsls()
# model = torch.nn.DataParallel(model, device_ids=[0, 1])
model = model.to(device=device)
# ##  ----------------end---modify---------------------------------#

model_path = step1_config.args.model_path
model_name = step1_config.args.model_name  # remember to include the project title (e.g., ALV)
checkpoint_name = step1_config.args.checkpoint_name

if use_visdom:
    global plotter
    plotter = VisdomLinePlotter(env_name=model_name)

# mkdir 'models'
if not os.path.exists(model_path):
    os.mkdir(model_path)

train_dataset = isic_dataset(train_list)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          pin_memory=True)

valid_dataset = isic_dataset(valid_list)
valid_loader = DataLoader(dataset=valid_dataset,
                         batch_size=1,
                         shuffle=True,
                         num_workers=num_workers,
                         pin_memory=True)


# set model
opt = optim.Adam(model.parameters(), lr=lr, amsgrad=True)

train_loss, valid_loss = [], []
valid_dice, valid_miou, valid_acc, valid_f1 = [], [], [], []

# cudnn
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

print('Training model...')
class_weights = torch.Tensor([1, 2]).to(device, dtype=torch.float)
best_dice = 0.0
best_acc = 0.0
best_f1 = 0.0
best_miou = 0.0

for epoch in range(num_epochs):
    if (epoch + 1) % 10 == 0:
        lr = lr * 0.8
        opt = optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    print('-------------lr ={0}'.format(lr))

    # training
    model.train()
    running_train_loss = 0.0
    loss_epoch = 0.0

    for i_batch, batched_sample in enumerate(train_loader):
        # inputs, labels = batched_sample['image'], batched_sample['label']
        inputs, labels = batched_sample['image'].to(device, dtype=torch.float), \
                         batched_sample['label'].to(device, dtype=torch.float)
        original_lables = labels

        opt.zero_grad()

        masks, outputs = model(inputs.float())

        # ---- loss function ----
        down = nn.MaxPool2d(2, 2)
        loss_fun = SLSIoULoss()
        loss = 0
        loss = loss + loss_fun(outputs, original_lables, 5, 100)
        lable = original_lables
        for j in range(len(masks)):
            lable = down(lable)
            loss = loss + loss_fun(masks[j], lable, 5, 100)
        loss = loss / (len(masks) + 1)

        # -------end moidfy
        # torch.backends.cudnn.benchmark = False

        loss.backward()
        opt.step()
        running_train_loss += loss.item()
        loss_epoch += loss.item()

        if i_batch % num_batches_to_print == num_batches_to_print - 1:  # print every N mini-batches
            print('[Epoch: {0}/{1}, Batch: {2}/{3}] train_loss: {4:.7f}'.format(
                epoch + 1, num_epochs, i_batch + 1, len(train_loader), running_train_loss / num_batches_to_print))

            if use_visdom:
                plotter.plot('loss', 'train', 'Loss', epoch + (i_batch + 1) / len(train_loader),
                             running_train_loss / num_batches_to_print)

            running_train_loss = 0.0

    train_loss.append(loss_epoch / len(train_loader))

    # save all loss and dice data
    pd_dict = {'lr': lr, 'train_loss': train_loss}
    stat = pd.DataFrame(pd_dict)
    stat.to_csv(train_lr_loss_path)


    loss_epoch = 0.0
    dice_epoch = 0.0
    miou_epoch = 0.0
    acc_epoch = 0.0
    f1_epoch = 0.0
    # decay learning rate
    # scheduler.step()
    if (epoch + test_epoch) % 1 == 0:
        model.eval()
        with torch.no_grad():

            for i_batch, batched_sample in enumerate(valid_loader):
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

                masks, outputs = model(inputs.float())

                output = torch.ceil((outputs.float() + 255) / 255) - 1.0
                # labels[labels > 0] = 1
                bn = output.shape[0]
                output = torch.reshape(output, [bn, 512, 512])
                one_hot_output = nn.functional.one_hot(output.long(), num_classes=2)  # lastet dimension is channel
                one_hot_output = one_hot_output.permute(0, 3, 1, 2).contiguous()

                # ---------- modify
                # ---- loss function ----
                down = nn.MaxPool2d(2, 2)
                loss_fun = SLSIoULoss()
                loss = 0
                loss = loss + loss_fun(outputs, original_lables, 5, 100)
                lable = original_lables
                for j in range(len(masks)):
                    lable = down(lable)
                    loss = loss + loss_fun(masks[j], lable, 5, 100)
                loss = loss / (len(masks) + 1)

                running_train_loss += loss.item()
                loss_epoch += loss.item()

                dice = unet_dice(one_hot_output, one_hot_label, class_weights)
                acc, miou, f1 = make_miou(original_lables, outputs)

                dice_epoch += dice.item()
                miou_epoch += miou
                acc_epoch += acc
                f1_epoch += f1
                # print(miou)
                if use_visdom:
                    plotter.plot('loss', 'valid', 'Loss', epoch + (i_batch + 1) / len(valid_loader),
                                 running_train_loss / num_batches_to_print)
                running_train_loss = 0.0

            valid_loss.append(loss_epoch / len(valid_loader))
            valid_dice.append(dice_epoch / len(valid_loader))
            valid_miou.append(miou_epoch / len(valid_loader))
            valid_acc.append(acc_epoch / len(valid_loader))
            valid_f1.append(f1_epoch / len(valid_loader))

            if dice_epoch / len(valid_loader) > best_dice:
                best_dice = dice_epoch / len(valid_loader)
                best_epoch = epoch + 1
                torch.save(model.state_dict(), weight_path)

            if miou_epoch / len(valid_loader) > best_miou:
                best_miou = miou_epoch / len(valid_loader)

            if acc_epoch / len(valid_loader) > best_acc:
                best_acc = acc_epoch / len(valid_loader)

            if f1_epoch / len(valid_loader) > best_f1:
                best_f1 = f1_epoch / len(valid_loader)

            # valid_loss,valid_dice,valid_miou,valid_acc,valid_f1
            # save the checkpoint
            pd_dict = {'valid_loss': valid_loss}
            stat = pd.DataFrame(pd_dict)
            stat.to_csv(valid_loss_path)

            pd_dicts = {'test_dice': valid_dice, 'test_miou': valid_miou,
                        'acc': valid_acc, 'f1_score': valid_f1}
            stats = pd.DataFrame(pd_dicts)
            stats.to_csv(valid_metric_path)

print('best_dice=', best_dice, 'best_miou=', best_miou, 'best_acc=',
      best_acc, 'best_f1_score=', best_f1, 'best_epoch=', best_epoch)
