# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, IterableDataset
from glob import glob
import os
import numpy as np
from PIL import Image

import random
import cv2
from torch import nn
import sys
import albumentations as alb

import warnings

warnings.filterwarnings('ignore')

import logging

if os.path.isfile(
        'src/utils/library/bi_online_generation.py'
):
    sys.path.append('src/utils/library/')
    print('exist library')
    exist_bi = True
else:
    exist_bi = False


class SBI_Dataset(Dataset):

    def __init__(self,
                 dataset_name,
                 phase='train',
                 image_size=224,
                 n_frames=8,
                 config=None):

        assert phase in ['train', 'val', 'test']
        self.fake_image_list, self.real_image_list, self.label_list = collect_img_and_label(
            phase=phase,
            dataset_name=dataset_name,
            frame_num=n_frames,
            config=config)
        if phase in ['val', 'test']:
            self.image_list = self.real_image_list + self.fake_image_list

        path_lm = '/landmarks/'

        self.path_lm = path_lm
        print(
            f'SBI({phase}):{dataset_name}, real image: {len(self.real_image_list)}, fake image: {len(self.fake_image_list)}, label: {len(self.label_list)}'
        )

        self.image_size = (image_size, image_size)

        self.phase = phase
        self.n_frames = n_frames

        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()

    def __len__(self):
        if self.phase == "train":
            return len(self.real_image_list)
        elif self.phase in ['val', 'test']:
            return len(self.image_list)

    def __getitem__(self, idx):
        if self.phase == 'train':
            flag = True
            while flag:
                try:
                    filename = self.real_image_list[idx]
                    img = np.array(Image.open(filename))
                    landmark = np.load(
                        filename.replace('.png', '.npy').replace(
                            '/rawframes/', self.path_lm))
                    bbox = np.load(
                        filename.replace('.png', '.npy').replace(
                            '/rawframes/', '/retina/'))
                    if len(landmark.shape)==3:
                        landmark = landmark[0]
                    if len(bbox.shape)==3:
                        bbox = bbox[0]
                    landmark = self.reorder_landmark(landmark)
                    if self.phase == 'train':
                        if np.random.rand() < 0.5:
                            img, _, landmark, bbox = self.hflip(
                                img, None, landmark, bbox)

                    img, landmark, bbox, __ = crop_face(img,
                                                        landmark,
                                                        bbox,
                                                        margin=True,
                                                        crop_by_bbox=False)

                    if np.random.rand() < 0.5:
                        img_r, img_f, mask_f = self.self_blending(
                            img.copy(), landmark.copy())
                        img_f, _, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(
                            img_f,
                            landmark,
                            bbox,
                            margin=False,
                            crop_by_bbox=True,
                            abs_coord=True,
                            phase=self.phase)
                        img_r = img_r[y0_new:y1_new, x0_new:x1_new]


                    else:
                        img_r = img.astype(np.uint8)
                        img_r, _, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(
                            img_r,
                            landmark,
                            bbox,
                            margin=False,
                            crop_by_bbox=True,
                            abs_coord=True,
                            phase=self.phase)

                        fake_filename = self.fake_image_list[np.random.randint(
                            len(self.fake_image_list))]
                        img_f = np.array(Image.open(fake_filename))
                        fake_bbox = np.load(
                            fake_filename.replace('.png', '.npy').replace(
                                '/rawframes/', '/retina/'))

                        img_f, _, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(
                            img_f,
                            None,
                            fake_bbox,
                            margin=False,
                            crop_by_bbox=True,
                            abs_coord=True,
                            phase=self.phase)

                        mask_f = cv2.imread(
                            fake_filename.replace("/c23/", "/masks/"),
                            cv2.IMREAD_GRAYSCALE)
                    mask_f = mask_f[y0_new:y1_new, x0_new:x1_new]
                    _, mask_f = cv2.threshold(mask_f, 0, 1,
                                                cv2.THRESH_BINARY)
                    img_f = cv2.resize(img_f, self.image_size,interpolation=cv2.INTER_LINEAR)
                    img_r = cv2.resize(img_r, self.image_size,interpolation=cv2.INTER_LINEAR)
                    transformed = self.transforms(image=img_f.astype('uint8'),
                                                  image1=img_r.astype('uint8'))
                    
                    img_f = transformed['image']
                    img_r = transformed['image1']

                    img_f = cv2.resize(
                        img_f, self.image_size,
                        interpolation=cv2.INTER_LINEAR).astype('float32') / 255
                    img_r = cv2.resize(
                        img_r, self.image_size,
                        interpolation=cv2.INTER_LINEAR).astype('float32') / 255

                    img_f = img_f.transpose((2, 0, 1))
                    img_r = img_r.transpose((2, 0, 1))

                    mask_f = cv2.resize(
                        mask_f, (12, 12),
                        interpolation=cv2.INTER_LINEAR).astype("float32")
                    _, mask_f = cv2.threshold(mask_f, 0, 1, cv2.THRESH_BINARY)

                    mask_r = np.zeros_like(mask_f)
                    mask_r = cv2.resize(
                        mask_r, (12, 12),
                        interpolation=cv2.INTER_LINEAR).astype("float32")
                    flag = False
                    return img_f, img_r, mask_f, mask_r

                except Exception as e:
                    print(e)
                    idx = torch.randint(low=0, high=len(self),
                                        size=(1, )).item()

        if self.phase== "val":
            flag = True
            while flag:
                try:
                    filename = self.image_list[idx]
                    label = self.label_list[idx]
                    img = np.array(Image.open(filename))
                    # bbox = np.load(
                    #     filename.replace('.png',
                    #                      '.npy').replace('/rawframes/',
                    #                                      '/retina/'))
                    # img, _, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(
                    #     img,
                    #     None,
                    #     bbox,
                    #     margin=False,
                    #     crop_by_bbox=True,
                    #     abs_coord=True,
                    #     phase=self.phase)
                    img = cv2.resize(
                        img, self.image_size,
                        interpolation=cv2.INTER_LINEAR).astype('float32') / 255
                    img = img.transpose((2, 0, 1))
                    flag = False
                    return img, label
                except Exception as e:
                    print(e)
                    idx = torch.randint(low=0, high=len(self), size=(1, )).item()
        if self.phase=="test":
            flag = True
            while flag:
                try:
                    filename = self.image_list[idx]
                    label = self.label_list[idx]
                    img = np.array(Image.open(filename))
                    parts = filename.split('/')
                    frame_info = {
                        'video': parts[-2],
                        'frame': int(parts[-1].split('_')[0]),
                        'face': int(parts[-1].split('_')[-1].split('.')[0]),
                    }
                    img = cv2.resize(
                        img, self.image_size,
                        interpolation=cv2.INTER_LINEAR).astype('float32') / 255
                    img = img.transpose((2, 0, 1))
                    flag = False
                    return img, label, frame_info
                except Exception as e:
                    print(e)
                    idx = torch.randint(low=0, high=len(self), size=(1, )).item()



    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose([
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3),
                                       sat_shift_limit=(-0.3, 0.3),
                                       val_shift_limit=(-0.3, 0.3),
                                       p=1),
                alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                             contrast_limit=(-0.1, 0.1),
                                             p=1),
            ],
                        p=1),
            alb.OneOf([
                RandomDownScale(p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ],
                      p=1),
        ],
                           p=1.)

    def get_transforms(self):
        return alb.Compose([
            alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3),
                                   sat_shift_limit=(-0.3, 0.3),
                                   val_shift_limit=(-0.3, 0.3),
                                   p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.7, 0.7),
                                         contrast_limit=(-0.7, 0.7),
                                         p=0.3),
            alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
            alb.ToGray(p=0.3),
        ],
                           additional_targets={f'image1': 'image'},
                           p=1.)

    def randaffine(self, img, mask):
        f = alb.Affine(translate_percent={
            'x': (-0.03, 0.03),
            'y': (-0.015, 0.015)
        },
                       scale=[0.95, 1 / 0.95],
                       fit_output=False,
                       p=1)

        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed['image']

        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask

    def self_blending(self, img, landmark):
        H, W = len(img), len(img[0])
        if np.random.rand() < 0.25:
            landmark = landmark[:68]
        if exist_bi:
            logging.disable(logging.FATAL)
            mask = random_get_hull(landmark, img)[:, :, 0]
            logging.disable(logging.NOTSET)
        else:
            mask = np.zeros_like(img[:, :, 0])
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

        source = img.copy()
        if np.random.rand() < 0.5:
            source = self.source_transforms(
                image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source, mask)

        img_blended, mask = B.dynamic_blend(source, img, mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img, img_blended, mask

    def reorder_landmark(self, landmark):
        landmark_add = np.zeros((13, 2))
        for idx, idx_l in enumerate(
            [77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
            landmark_add[idx] = landmark[idx_l]
        landmark[68:] = landmark_add
        return landmark

    def hflip(self, img, mask=None, landmark=None, bbox=None):
        H, W = img.shape[:2]
        landmark = landmark.copy()
        bbox = bbox.copy()

        if landmark is not None:
            landmark_new = np.zeros_like(landmark)

            landmark_new[:17] = landmark[:17][::-1]
            landmark_new[17:27] = landmark[17:27][::-1]

            landmark_new[27:31] = landmark[27:31]
            landmark_new[31:36] = landmark[31:36][::-1]

            landmark_new[36:40] = landmark[42:46][::-1]
            landmark_new[40:42] = landmark[46:48][::-1]

            landmark_new[42:46] = landmark[36:40][::-1]
            landmark_new[46:48] = landmark[40:42][::-1]

            landmark_new[48:55] = landmark[48:55][::-1]
            landmark_new[55:60] = landmark[55:60][::-1]

            landmark_new[60:65] = landmark[60:65][::-1]
            landmark_new[65:68] = landmark[65:68][::-1]
            if len(landmark) == 68:
                pass
            elif len(landmark) == 81:
                landmark_new[68:81] = landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:, 0] = W - landmark_new[:, 0]

        else:
            landmark_new = None

        if bbox is not None:
            bbox_new = np.zeros_like(bbox)
            bbox_new[0, 0] = bbox[1, 0]
            bbox_new[1, 0] = bbox[0, 0]
            bbox_new[:, 0] = W - bbox_new[:, 0]
            bbox_new[:, 1] = bbox[:, 1].copy()
            if len(bbox) > 2:
                bbox_new[2, 0] = W - bbox[3, 0]
                bbox_new[2, 1] = bbox[3, 1]
                bbox_new[3, 0] = W - bbox[2, 0]
                bbox_new[3, 1] = bbox[2, 1]
                bbox_new[4, 0] = W - bbox[4, 0]
                bbox_new[4, 1] = bbox[4, 1]
                bbox_new[5, 0] = W - bbox[6, 0]
                bbox_new[5, 1] = bbox[6, 1]
                bbox_new[6, 0] = W - bbox[5, 0]
                bbox_new[6, 1] = bbox[5, 1]
        else:
            bbox_new = None

        if mask is not None:
            mask = mask[:, ::-1]
        else:
            mask = None
        img = img[:, ::-1].copy()
        return img, mask, landmark_new, bbox_new

    def collate_fn(self, batch):
        if self.phase == "train":
            img_f, img_r, mask_f, mask_r = zip(*batch)
            data = {}
            data["img"] = torch.cat(
                [torch.tensor(img_r).float(),
                 torch.tensor(img_f).float()], 0)
            data["mask"] = torch.cat(
                [torch.tensor(mask_r).float(),
                 torch.tensor(mask_f).float()], 0)
            data["label"] = torch.tensor([0] * len(img_r) + [1] * len(img_f))
        elif self.phase== "val":
            data = {}
            img, label = zip(*batch)
            data["img"] = torch.tensor(img).float()
            data["label"] = torch.tensor(label)
        elif self.phase=="test":
            data = {}
            img, label, frame_info = zip(*batch)
            data["img"] = torch.tensor(img).float()
            data["label"] = torch.tensor(label)
            data["frame_info"] = frame_info
        return data
        

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == '__main__':
    import blend as B
    from initialize import *
    from funcs import IoUfrom2bboxes, crop_face, RandomDownScale
    if exist_bi:
        from library.bi_online_generation import random_get_hull
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    image_dataset = SBI_Dataset(phase='test', image_size=256)
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=image_dataset.collate_fn,
        num_workers=0,
        worker_init_fn=image_dataset.worker_init_fn)
    data_iter = iter(dataloader)
    data = next(data_iter)
    img = data['img']
    img = img.view((-1, 3, 256, 256))
    utils.save_image(img,
                     'loader.png',
                     nrow=batch_size,
                     normalize=False,
                     range=(0, 1))
else:
    from utils import blend as B
    from .initialize import *
    from .funcs import IoUfrom2bboxes, crop_face, RandomDownScale
    if exist_bi:
        from utils.library.bi_online_generation import random_get_hull
