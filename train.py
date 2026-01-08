import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tensorboardX import SummaryWriter
from sklearn import metrics

from dataloader import MRDataset
import vit
import utils

def train_model(model, train_loaders, epoch, num_epochs, optimizer, writer, current_lr, device, log_every=100):
    _ = model.train()
    model.to(device)

    sagittal_train_loader, coronal_train_loader, axial_train_loader = train_loaders
    
    y_preds = []
    y_trues = []
    losses = []

    for i, ((sagittal_image, label, weight), (coronal_image, _ ,_), (axial_image, _ ,_))  in enumerate(zip(sagittal_train_loader, coronal_train_loader, axial_train_loader)):
        optimizer.zero_grad()

        sagittal_image = sagittal_image.to(device)
        coronal_image = coronal_image.to(device)
        axial_image = axial_image.to(device)
        label = label.to(device)
        weight = weight.to(device)

        sagittal_image = utils.normalize(sagittal_image / 255.0, device)
        coronal_image = utils.normalize(coronal_image / 255.0, device)
        axial_image = utils.normalize(axial_image / 255.0, device)
        prediction = model.forward(sagittal_image, coronal_image, axial_image)
        loss = nn.BCEWithLogitsLoss(weight=weight)(prediction, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            if len(np.unique(y_trues)) > 1:
                auc = metrics.roc_auc_score(y_trues, y_preds)
            else:
                auc = 0.5
        except:
            auc = 0.5

        writer.add_scalar('Train/Loss', loss_value, epoch * len(sagittal_train_loader) + i)
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


def evaluate_model(model, val_loaders, epoch, num_epochs, writer, current_lr, device, log_every=20):
    _ = model.eval()
    model.to(device)

    sagittal_validation_loader, coronal_validation_loader, axial_validation_loader = val_loaders

    y_trues = []
    y_preds = []
    losses = []

    for i, ((sagittal_image, label, weight), (coronal_image, _ ,_), (axial_image, _ ,_)) in enumerate(zip(sagittal_validation_loader, coronal_validation_loader, axial_validation_loader)):

        sagittal_image = sagittal_image.to(device)
        coronal_image = coronal_image.to(device)
        axial_image = axial_image.to(device)
        label = label.to(device)
        weight = weight.to(device)

        sagittal_image = utils.normalize(sagittal_image / 255.0, device)
        coronal_image = utils.normalize(coronal_image / 255.0, device)
        axial_image = utils.normalize(axial_image / 255.0, device)
        prediction = model.forward(sagittal_image, coronal_image, axial_image)
        loss = nn.BCEWithLogitsLoss(weight=weight)(prediction, label)

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
            if len(np.unique(y_trues)) > 1:
                auc = metrics.roc_auc_score(y_trues, y_preds)
            else:
                auc = 0.5
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


def run(args):
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    log_root_folder = "./logs/{0}/{1}/".format(args.task, args.plane)
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)

    now = datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir, exist_ok=True)

    writer = SummaryWriter(logdir)

    augmentor = transforms.Compose([
        transforms.Lambda(utils.convert_to_tensor),
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=25, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(utils.repeat_and_permute),
    ])

    val_augmentor = transforms.Compose([
        transforms.Lambda(utils.convert_to_tensor),
        transforms.Resize((224, 224)),
        transforms.Lambda(utils.repeat_and_permute),
    ])

    sagittal_train_dataset = MRDataset('MRNet-v1.0/', 'sagittal', transform=augmentor, train=True)
    coronal_train_dataset = MRDataset('MRNet-v1.0/', 'coronal', transform=augmentor, train=True)
    axial_train_dataset = MRDataset('MRNet-v1.0/', 'axial', transform=augmentor, train=True)

    sagittal_train_loader = torch.utils.data.DataLoader(sagittal_train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    coronal_train_loader = torch.utils.data.DataLoader(coronal_train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    axial_train_loader = torch.utils.data.DataLoader(axial_train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)

    sagittal_validation_dataset = MRDataset( 'MRNet-v1.0/', 'sagittal', transform=val_augmentor, train=False)
    coronal_validation_dataset = MRDataset( 'MRNet-v1.0/',  'coronal', transform=val_augmentor, train=False)
    axial_validation_dataset = MRDataset( 'MRNet-v1.0/', 'axial', transform=val_augmentor, train=False)

    sagittal_validation_loader = torch.utils.data.DataLoader(sagittal_validation_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    coronal_validation_loader = torch.utils.data.DataLoader(coronal_validation_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    axial_validation_loader = torch.utils.data.DataLoader(axial_validation_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    train_loaders = (sagittal_train_loader, coronal_train_loader, axial_train_loader)
    val_loaders = (sagittal_validation_loader, coronal_validation_loader, axial_validation_loader)

    mrnet = vit.MRNetViT()
    mrnet.to(device)

    optimizer = optim.Adam(mrnet.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.3, threshold=1e-4)

    best_val_loss = float('inf')
    best_val_auc = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience
    log_every = args.log_every

    t_start_training = time.time()

    for epoch in range(num_epochs):
        current_lr = utils.get_lr(optimizer)
        t_start = time.time()

        train_loss, train_auc, train_y_trues, train_y_preds = train_model(
            mrnet, train_loaders, epoch, num_epochs, optimizer, writer, current_lr, device, log_every)
        
        val_loss, val_auc, val_y_trues, val_y_preds = evaluate_model(
            mrnet, val_loaders, epoch, num_epochs, writer, current_lr, device)

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
                # Ensure models directory exists
                os.makedirs('models', exist_ok=True)
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
