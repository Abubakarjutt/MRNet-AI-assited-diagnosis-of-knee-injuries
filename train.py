import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from dataloader import MRDataset
import model

from sklearn import metrics
import alexnet
import geomloss



def train_model(model, sagittal_train_loader, coronal_train_loader, axial_train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every=100):
    _ = model.train()

    if torch.cuda.is_available():
        model.cuda()

    y_preds = []
    y_trues = []
    losses = []

    for i, ((sagittal_image, label, weight), (coronal_image, _ ,_), (axial_image, _ ,_))  in enumerate(zip(sagittal_train_loader, coronal_train_loader, axial_train_loader)):
        optimizer.zero_grad()

        if torch.cuda.is_available():
              sagittal_image = sagittal_image.cuda()
              coronal_image = coronal_image.cuda()
              axial_image = axial_image.cuda()
              label = label.cuda()
              weight = weight.cuda()


        label = label
        weight = weight



        prediction = model.forward(sagittal_image / 255.0, coronal_image / 255.0, axial_image / 255.0)
        #loss = geomloss.SamplesLoss()(prediction, label)
        loss = nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
        #loss = nn.MultiLabelSoftMarginLoss(weight=weights)(prediction, label)
        #loss = F.smooth_l1_loss(prediction, label)




        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0][0]))
        y_trues.append(int(label[0][1]))
        y_trues.append(int(label[0][2]))


        y_preds.append(probas[0][0].item())
        y_preds.append(probas[0][1].item())
        y_preds.append(probas[0][2].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(sagittal_train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(sagittal_train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]| avg train loss {4} | train auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(sagittal_train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

        writer.add_scalar('Train/AUC_epoch', auc, epoch + i)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    return train_loss_epoch, train_auc_epoch, y_trues, y_preds


def evaluate_model(model, sagittal_validation_loader, coronal_validation_loader, axial_validation_loader, epoch, num_epochs, writer, current_lr, log_every=20):
    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    losses = []

    for i, ((sagittal_image, label, weight), (coronal_image, _ ,_), (axial_image, _ ,_)) in enumerate(zip(sagittal_validation_loader, coronal_validation_loader, axial_validation_loader)):


        if torch.cuda.is_available():
              sagittal_image = sagittal_image.cuda()
              coronal_image = coronal_image.cuda()
              axial_image = axial_image.cuda()
              label = label.cuda()
              weight = weight.cuda()

        label = label
        weight = weight


        prediction = model.forward(sagittal_image / 255.0, coronal_image / 255.0, axial_image / 255.0)
        #loss = geomloss.SamplesLoss()(prediction, label)
        loss = nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
        #loss = nn.MultiLabelSoftMarginLoss(weight=weights)(prediction, label)
        #loss = F.smooth_l1_loss(prediction, label)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0][0]))
        y_trues.append(int(label[0][1]))
        y_trues.append(int(label[0][2]))


        y_preds.append(probas[0][0].item())
        y_preds.append(probas[0][1].item())
        y_preds.append(probas[0][2].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        writer.add_scalar('Val/Loss', loss_value, epoch * len(sagittal_validation_loader) + i)
        writer.add_scalar('Val/AUC', auc, epoch * len(sagittal_validation_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | avg val loss {4} | val auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(sagittal_validation_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

        writer.add_scalar('Val/AUC_epoch', auc, epoch + i)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)
    return val_loss_epoch, val_auc_epoch, y_trues, y_preds

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run(args):
    log_root_folder = "./logs/{0}/{1}/".format(args.task, args.plane)
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)

    now = datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)

    augmentor = Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        #RandomRotate(25),
        #RandomTranslate([0.11, 0.11]),
        #RandomFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    ])

    sagittal_train_dataset = MRDataset('MRNet-v1.0/', 'sagittal', transform=augmentor, train=True)
    coronal_train_dataset = MRDataset('MRNet-v1.0/', 'coronal', transform=augmentor, train=True)
    axial_train_dataset = MRDataset('MRNet-v1.0/', 'axial', transform=augmentor, train=True)

    sagittal_train_loader = torch.utils.data.DataLoader(sagittal_train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    coronal_train_loader = torch.utils.data.DataLoader(coronal_train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    axial_train_loader = torch.utils.data.DataLoader(axial_train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)


    sagittal_validation_dataset = MRDataset( 'MRNet-v1.0/', 'sagittal', train=False)
    coronal_validation_dataset = MRDataset( 'MRNet-v1.0/',  'coronal', train=False)
    axial_validation_dataset = MRDataset( 'MRNet-v1.0/', 'axial', train=False)

    sagittal_validation_loader = torch.utils.data.DataLoader(sagittal_validation_dataset, batch_size=1, shuffle=-True, num_workers=1, drop_last=False)
    coronal_validation_loader = torch.utils.data.DataLoader(coronal_validation_dataset, batch_size=1, shuffle=-True, num_workers=1, drop_last=False)
    axial_validation_loader = torch.utils.data.DataLoader(axial_validation_dataset, batch_size=1, shuffle=-True, num_workers=1, drop_last=False)

    mrnet = alexnet.AlexNet()
    torch.cuda.empty_cache()


    if torch.cuda.is_available():
        mrnet = mrnet.cuda()


    optimizer = optim.Adam(mrnet.parameters(), lr=args.lr, weight_decay=0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)




    best_val_loss = float('inf')
    best_val_auc = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience
    log_every = args.log_every

    t_start_training = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)


        t_start = time.time()

        train_loss, train_auc, train_y_trues, train_y_preds = train_model(
            mrnet, sagittal_train_loader, coronal_train_loader, axial_train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every)
        val_loss, val_auc, val_y_trues, val_y_preds = evaluate_model(
            mrnet, sagittal_validation_loader, coronal_validation_loader, axial_validation_loader, epoch, num_epochs, writer, current_lr)

        scheduler.step(val_loss)




        t_end = time.time()
        delta = t_end - t_start

        train_true_negatives, train_false_positives, train_false_negatives, train_true_positives = metrics.confusion_matrix(train_y_trues, np.array(train_y_preds).round()).ravel()
        val_true_negatives, val_false_positives, val_false_negatives, val_true_positives = metrics.confusion_matrix(val_y_trues, np.array(val_y_preds).round()).ravel()

        print("train loss : {0} | train auc {1} | val loss {2} | val auc {3} | elapsed time {4} s".format(
            train_loss, train_auc, val_loss, val_auc, delta))

        print("train_true_negatives :{0} | train_false_positives {1} | train_false_negatives {2} | train_true_positives {3}".format(
            train_true_negatives, train_false_positives, train_false_negatives, train_true_positives))

        print("val_true_negatives :{0} | val_false_positives {1} | val_false_negatives {2} | val_true_positives {3}".format(
            val_true_negatives, val_false_positives, val_false_negatives, val_true_positives))


        iteration_change_loss += 1
        print('-' * 30)


        if val_auc > best_val_auc:
            best_val_auc = val_auc
            if bool(args.save_model):
                file_name = f'model_{args.prefix_name}_{args.task}_{args.plane}_val_auc_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_epoch_{epoch+1}.pth'
                for f in os.listdir('models/'):
                    if (args.task in f) and (args.plane in f) and (args.prefix_name in f):
                        os.remove(f'models/{f}')
                torch.save(mrnet, f'models/{file_name}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print('Early stopping after {0} iterations without the decrease of the val loss'.
                  format(iteration_change_loss))
            break

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str,
                        choices=['abnormal', 'acl', 'meniscus'], default = 'acl')
    parser.add_argument('-p', '--plane', type=str,
			choices=['Segittal_Coronal_and_Axial'], default = 'Segittal_Coronal_and_Axial')
    parser.add_argument('--lr_scheduler', type=str,
                        default='plateau', choices=['plateau', 'step'])
    parser.add_argument('--prefix_name', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--log_every', type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
