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



def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)


def main(args):
    cfg = load_json(args.config)
    
    image_size = cfg['image_size']
    batch_size = cfg['batch_size']['test']
    device = torch.device('cuda')
    model = Detector()
    model = model.to(device)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(cfg['checkpoint'])["model"]

    unmatched_keys = [
        k for k, v in pretrained_dict.items()
        if k not in model_dict or model_dict[k].size() != v.size()
    ]
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }

    print(f"unmatched_keys{unmatched_keys}")

    model.load_state_dict(pretrained_dict)
    model.eval()
    batch_size = cfg['batch_size']['test']

    test_dataset_names = cfg['dataset']['test']

    test_loaders = {}

    for test_dataset_name in test_dataset_names:
        test_dataset = SBI_Dataset(dataset_name=test_dataset_name,
                                  phase='test',
                                  image_size=image_size,
                                  n_frames=cfg['frame_num']['test'],
                                  config=cfg)
        test_loaders[test_dataset_name] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=test_dataset.collate_fn,
            num_workers=8,
            pin_memory=False,
            worker_init_fn=test_dataset.worker_init_fn)


    for dataset_name, teset_loader in test_loaders.items():
        output_list = []
        target_list = []
        video_results = {}
        for step, data in enumerate(tqdm(teset_loader)):
    
            imgs = data['img'].to(device, non_blocking=True).float()
            labels = data['label'].to(device, non_blocking=True).long()
            frame_info = data['frame_info']

            with torch.no_grad():
                output = model(imgs)
                output = output[0]
                probabilities = torch.softmax(output, dim=1)[: , 1]

            for i in range(len(imgs)):
                video = frame_info[i]['video']
                frame = frame_info[i]['frame']
                face = frame_info[i]['face']
                label = labels[i]
                
                if video not in video_results:
                    video_results[video] = {}
                    video_results[video]['label'] = label.item()

                if frame not in video_results[video]:
                    video_results[video][frame] = []
                

                # Append the probability of the current face to the frame's list
                video_results[video][frame].append(probabilities[i].item())

        # Now aggregate the results
        final_results = {}
        with open(f"{'_'.join(cfg['checkpoint'].split('/')[-3:])}_{dataset_name}_predict.txt", 'w') as f:
                
            for video, frames in video_results.items():
                frame_averages = []
                for frame, probs in frames.items():
                    if frame == 'label':
                        continue
                    
                    max_prob = max(probs)  # Take the max probability for the frame
                    f.write(f"{video} {frame} {max_prob}\n")
                    frame_averages.append(max_prob)
                frame_averages = [
                    frame_averages[i] for i in np.linspace(
                        0, len(frame_averages) - 1, cfg['frame_num']['test'], dtype=int)
                ]

                video_average = sum(frame_averages) / len(frame_averages)  # Average over frames
                final_results[video] = video_average

                # Append to output and target lists
                output_list.append(video_average)
                target_list.append(video_results[video]['label'])

        # Compute AUC
        auc = roc_auc_score(target_list, output_list)
        print(f'{dataset_name}| AUC: {auc:.4f}')

    return final_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n', dest='session_name')
    args = parser.parse_args()
    main(args)
