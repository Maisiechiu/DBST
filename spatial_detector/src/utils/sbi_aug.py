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
        '/home/jovyan/SelfBlendedImages/src/utils/library/bi_online_generation.py'
):
    sys.path.append('/home/jovyan/SelfBlendedImages/src/utils/library/')
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
        self.rawframes2_list, self.landmark_list = init_aug(
            root_folder=
            "/home/jovyan/dataset/FaceForensics++/original_sequences/youtube/raw/rawframes2_norm1",
            target_count=33)

        self.image_size = (image_size, image_size)

        self.phase = phase
        self.n_frames = n_frames

        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()

    def __len__(self):
        if self.phase == "train":
            return len(self.rawframes2_list)
        elif self.phase in ['val', 'test']:
            return len(self.rawframes2_list)

    def __getitem__(self, idx):
        try:
            image_paths = self.rawframes2_list[idx]
            landmark_paths = self.landmark_list[idx]
            frame_save_dir = os.path.dirname(image_paths[0]).replace(
                'original_sequences', 'manipulated_sequences')
            mask_save_dir = os.path.dirname(image_paths[0]).replace(
                'original_sequences',
                'manipulated_sequences').replace('/raw/', '/masks/')
            os.makedirs(frame_save_dir, exist_ok=True)
            os.makedirs(mask_save_dir, exist_ok=True)

            source_dict = {}
            image_dict = {}
            landmarks_list = []

            h_flip = True if np.random.rand() < 0.5 else False

            for idx, frame in enumerate(image_paths):
                frame = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
                if h_flip:
                    frame, _, _, _ = self.hflip(frame, None, None, None)
                if idx == 0:
                    source_dict[f'image'] = frame
                else:
                    source_dict[f'image{str(idx).zfill(2)}'] = frame

            image_dict = source_dict.copy()

            for landmark_path in landmark_paths:
                landmark = np.load(landmark_path)
                landmark = self.reorder_landmark(landmark)
                if h_flip:
                    _, _, landmark, _ = self.hflip(frame, None, landmark, None)
                landmarks_list.append(landmark)

            masks = []

            manipulated_frames = {}

            num_frames_to_blend = random.randint(17, 33)
            frames_to_blend = random.sample(range(len(source_dict)),
                                            num_frames_to_blend)
            if random.random() < 0.5:
                transformed = self.source_transforms(**source_dict)
                source_list = []
                image_list = []
                for (key_src, src), (key_img,
                                     img) in zip(sorted(transformed.items()),
                                                 sorted(image_dict.items())):
                    source_list.append(src)
                    image_list.append(img)
            else:
                transformed = self.source_transforms(**image_dict)
                source_list = []
                image_list = []
                for (key_src, src), (key_img,
                                     img) in zip(sorted(source_dict.items()),
                                                 sorted(transformed.items())):
                    source_list.append(src)
                    image_list.append(img)
            hull_type = random.choice([0, 1, 2, 3])
            for idx, (src, img, landmark) in enumerate(
                    zip(source_list, image_list, landmarks_list)):
                if idx in frames_to_blend:
                    img, img_blended, mask = self.self_blending(
                        src, img, landmark, hull_type=hull_type)
                    mask = mask * 255
                else:
                    img_blended = img
                    mask = np.zeros_like(img, dtype=np.uint8)
                # masks.append(mask)
                cv2.imwrite(
                    os.path.join(frame_save_dir,
                                 f'image_{str(idx+1).zfill(5)}.png'),
                    cv2.cvtColor(img_blended, cv2.COLOR_BGR2RGB))
                cv2.imwrite(
                    os.path.join(mask_save_dir,
                                 f'image_{str(idx+1).zfill(5)}.png'), mask)
                # if idx == 0:
                #     manipulated_frames[f'image'] = img_blended
                # else:
                #     manipulated_frames[
                #         f'image{str(idx).zfill(2)}'] = img_blended

            # transformed = self.transforms(**manipulated_frames)
            # manipulated_frames = [
            #     value for key, value in sorted(transformed.items())
            # ]
            # for i, (frame, mask) in enumerate(zip(manipulated_frames, masks)):
            #     cv2.imwrite(
            #         os.path.join(frame_save_dir,
            #                      f'image_{str(i+1).zfill(5)}.png'),
            #         cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #     cv2.imwrite(
            #         os.path.join(mask_save_dir,
            #                      f'image_{str(i+1).zfill(5)}.png'), mask)

            return None
        except Exception as e:
            print(e)
            return None

    def get_source_transforms(self):
        return alb.Compose(
            [
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
            additional_targets={
                "image01": "image",
                "image02": "image",
                "image03": "image",
                "image04": "image",
                "image05": "image",
                "image06": "image",
                "image07": "image",
                "image08": "image",
                "image09": "image",
                "image10": "image",
                "image11": "image",
                "image12": "image",
                "image13": "image",
                "image14": "image",
                "image15": "image",
                "image16": "image",
                "image17": "image",
                "image18": "image",
                "image19": "image",
                "image20": "image",
                "image21": "image",
                "image22": "image",
                "image23": "image",
                "image24": "image",
                "image25": "image",
                "image26": "image",
                "image27": "image",
                "image28": "image",
                "image29": "image",
                "image30": "image",
                "image31": "image",
                "image32": "image",
            },
            p=1.)

    def get_transforms(self):
        return alb.Compose(
            [
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3),
                                       sat_shift_limit=(-0.3, 0.3),
                                       val_shift_limit=(-0.3, 0.3),
                                       p=0.3),
                alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3),
                                             contrast_limit=(-0.3, 0.3),
                                             p=0.3),
                alb.ImageCompression(
                    quality_lower=40, quality_upper=100, p=0.5),
            ],
            additional_targets={
                "image01": "image",
                "image02": "image",
                "image03": "image",
                "image04": "image",
                "image05": "image",
                "image06": "image",
                "image07": "image",
                "image08": "image",
                "image09": "image",
                "image10": "image",
                "image11": "image",
                "image12": "image",
                "image13": "image",
                "image14": "image",
                "image15": "image",
                "image16": "image",
                "image17": "image",
                "image18": "image",
                "image19": "image",
                "image20": "image",
                "image21": "image",
                "image22": "image",
                "image23": "image",
                "image24": "image",
                "image25": "image",
                "image26": "image",
                "image27": "image",
                "image28": "image",
                "image29": "image",
                "image30": "image",
                "image31": "image",
                "image32": "image",
            },
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
            alpha_affine=0,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed['image']

        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask

    # def self_blending(self, img, landmark):
    #     H, W = len(img), len(img[0])
    #     if np.random.rand() < 0.25:
    #         landmark = landmark[:68]
    #     if exist_bi:
    #         logging.disable(logging.FATAL)
    #         mask = random_get_hull(landmark, img)[:, :, 0]
    #         logging.disable(logging.NOTSET)
    #     else:
    #         mask = np.zeros_like(img[:, :, 0])
    #         cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

    #     source = img.copy()
    #     if np.random.rand() < 0.5:
    #         source = self.source_transforms(
    #             image=source.astype(np.uint8))['image']
    #     else:
    #         img = self.source_transforms(image=img.astype(np.uint8))['image']

    #     source, mask = self.randaffine(source, mask)

    #     img_blended, mask = B.dynamic_blend(source, img, mask)
    #     img_blended = img_blended.astype(np.uint8)
    #     img = img.astype(np.uint8)

    #     return img, img_blended, mask

    def self_blending(self, source, img, landmark, hull_type=None):
        H, W = len(img), len(img[0])
        if np.random.rand() < 0.25:
            landmark = landmark[:68]
        if exist_bi:
            logging.disable(logging.FATAL)
            mask = random_get_hull(landmark, img, hull_type=hull_type)[:, :, 0]
            logging.disable(logging.NOTSET)
        else:
            mask = np.zeros_like(img[:, :, 0])
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

        # source = img.copy()
        # if np.random.rand() < 0.5:
        #     source = self.source_transforms(
        #         image=source.astype(np.uint8))['image']
        # else:
        #     img = self.source_transforms(image=img.astype(np.uint8))['image']

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

        if landmark is not None:
            landmark = landmark.copy()
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
            bbox = bbox.copy()
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

        return None

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
