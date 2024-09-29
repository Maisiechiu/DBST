import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from utils.sbi_aug2 import SBI_Dataset
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from utils.funcs import load_json
from tqdm import tqdm


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
    batch_size = cfg['batch_size']
    train_dataset = SBI_Dataset(dataset_name=cfg['dataset']['train'],
                                phase='train',
                                image_size=image_size,
                                n_frames=cfg['frame_num']['train'],
                                config=cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size // 2,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn)

    for epoch in range(0, 1):
        for step, data in enumerate(tqdm(train_loader)):
            continue
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n', dest='session_name')
    args = parser.parse_args()
    main(args)
