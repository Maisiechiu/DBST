import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import sys
import random
from utils.sbi_aux import SBI_Dataset
from utils.scheduler import LinearDecayLR
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from utils.logs import log
from utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm
from model_aux import Detector
from utils.supcon import SupConLoss
import json

def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)


def main(args):
    cfg = load_json(args.config)

    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    image_size = cfg['image_size']
    batch_size = cfg['batch_size']['train']
    train_dataset = SBI_Dataset(dataset_name=cfg['dataset']['train'],
                                phase='train',
                                image_size=image_size,
                                n_frames=cfg['frame_num']['train'],
                                config=cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=8,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn)

    val_dataset_names = cfg['dataset']['val']

    val_loaders = {}

    for val_dataset_name in val_dataset_names:
        val_dataset = SBI_Dataset(dataset_name=val_dataset_name,
                                  phase='val',
                                  image_size=image_size,
                                  n_frames=cfg['frame_num']['val'],
                                  config=cfg)
        val_loaders[val_dataset_name] = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=8,
            pin_memory=False,
            worker_init_fn=val_dataset.worker_init_fn)

    model = Detector()
    model = model.to('cuda')

    iter_loss = []
    iter_map_loss = []
    iter_sup_loss = []
    train_losses = []
    train_map_losses = []
    train_sup_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    val_accs = []
    val_losses = []
    n_epoch = cfg['epoch']
    start_epoch = 0
    lr_scheduler = LinearDecayLR(model.optimizer, n_epoch,
                                 int(n_epoch / 4 * 3))
    lr_scheduler_head = LinearDecayLR(model.optimizer_head, n_epoch,
                                      int(n_epoch / 4 * 3))

    last_loss = 99999

    criterion = nn.CrossEntropyLoss()
    criterion_map = nn.BCEWithLogitsLoss()
    criterion_sup = SupConLoss()

    last_auc = 0
    last_val_auc = 0
    weight_dict = {}
    n_weight = 5

    # load checkpoint and save checkpoint
    checkpoint_path = cfg['checkpoint']
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        save_path = os.path.dirname(os.path.dirname(checkpoint_path)) + '/'
        print(f"Success load checkpoint from {checkpoint_path}!")

    else:
        now = datetime.now()
        save_path = 'output/{}_'.format(args.session_name) + now.strftime(
            os.path.splitext(os.path.basename(
                args.config))[0]) + '_' + now.strftime("%m_%d_%H_%M_%S") + '/'
        os.mkdir(save_path)
        os.mkdir(save_path + 'weights/')
        os.mkdir(save_path + 'logs/')

    logger = log(path=save_path + "logs/", file="losses.logs")

    for epoch in range(start_epoch, n_epoch):
        # if epoch % 10 == 0:    
        print("reloading data...")
        train_dataset = SBI_Dataset(dataset_name=cfg['dataset']['train'],
                            phase='train',
                            image_size=image_size,
                            n_frames=cfg['frame_num']['train'],
                            config=cfg)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size // 2,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            num_workers=8,
            pin_memory=False,
            drop_last=True,
            worker_init_fn=train_dataset.worker_init_fn)
        np.random.seed(seed + epoch)
        train_loss = 0.
        train_acc = 0.
        train_map_loss = 0.
        train_sup_loss = 0.
        model.train(mode=True)
        for step, data in enumerate(tqdm(train_loader)):
            img = data['img'].to(device, non_blocking=True).float()
            target = data['label'].to(device, non_blocking=True).long()
            mask = data['mask'].to(device, non_blocking=True).float()

            output, map, features = model.training_step(img, target, mask)

            loss = criterion(output, target)
            loss_value = loss.item()
            iter_loss.append(loss_value)
            train_loss += loss_value

            map_loss = criterion_map(map, mask)
            map_loss_value = map_loss.item()
            iter_map_loss.append(map_loss_value)
            train_map_loss += map_loss_value

            sup_loss = criterion_sup(features, target)
            sup_loss_value = sup_loss.item()
            iter_sup_loss.append(sup_loss_value)
            train_sup_loss += sup_loss_value

            acc = compute_accuray(F.log_softmax(output, dim=1), target)
            train_acc += acc
        lr_scheduler.step()
        lr_scheduler_head.step()
        train_losses.append(train_loss / len(train_loader))
        train_map_losses.append(train_map_loss / len(train_loader))
        train_sup_losses.append(train_sup_loss / len(train_loader))
        train_accs.append(train_acc / len(train_loader))

        log_text = "Epoch {}/{} | cls loss: {:.4f}, train map loss: {:4f}, train sup loss: {:4f}, train acc: {:.4f}".format(
            epoch + 1, n_epoch, train_loss / len(train_loader),
            train_map_loss / len(train_loader), train_sup_loss/len(train_loader), train_acc / len(train_loader))

        model.train(mode=False)
        val_loss = 0.
        val_acc = 0.
        output_dict = []
        target_dict = []
        val_aucs = 0
        np.random.seed(seed)
        for dataset_name, val_loader in val_loaders.items():
            for step, data in enumerate(tqdm(val_loader)):
                img = data['img'].to(device, non_blocking=True).float()
                target = data['label'].to(device, non_blocking=True).long()

                with torch.no_grad():
                    output = model(img)
                    output = output[0]
                    loss = criterion(output, target)

                loss_value = loss.item()
                iter_loss.append(loss_value)
                val_loss += loss_value
                acc = compute_accuray(F.log_softmax(output, dim=1), target)
                val_acc += acc
                output_dict += output.softmax(
                    1)[:, 1].cpu().data.numpy().tolist()
                target_dict += target.cpu().data.numpy().tolist()
            val_losses.append(val_loss / len(val_loader))
            val_accs.append(val_acc / len(val_loader))
            val_auc = roc_auc_score(target_dict, output_dict)
            val_aucs += val_auc
            log_text += " | {} val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(
                dataset_name, val_loss / len(val_loader),
                val_acc / len(val_loader), val_auc)

        if len(weight_dict) < n_weight:
            save_model_path = os.path.join(
                save_path + 'weights/',
                "{}_{:.4f}_val.tar".format(epoch + 1,
                                           val_aucs / len(val_loaders)))
            weight_dict[save_model_path] = val_aucs / len(val_loaders)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": model.optimizer.state_dict(),
                    "epoch": epoch
                }, save_model_path)
            last_val_auc = min([weight_dict[k] for k in weight_dict])

        elif val_aucs / len(val_loaders) >= last_val_auc:
            save_model_path = os.path.join(
                save_path + 'weights/',
                "{}_{:.4f}_val.tar".format(epoch + 1,
                                           val_aucs / len(val_loaders)))
            for k in weight_dict:
                if weight_dict[k] == last_val_auc:
                    del weight_dict[k]
                    os.remove(k)
                    weight_dict[save_model_path] = val_aucs / len(val_loaders)
                    break
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": model.optimizer.state_dict(),
                    "epoch": epoch
                }, save_model_path)
            last_val_auc = min([weight_dict[k] for k in weight_dict])
        print(weight_dict)
        logger.info(log_text)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n', dest='session_name')
    args = parser.parse_args()
    main(args)
