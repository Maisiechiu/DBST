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

    def write_frame_info(self, frame_info):
        with open('frame_info.txt', 'a') as file:
            for frame_info in self.frame_info_list:
                file.write(frame_info + '\n')

    def __init__(self,
                 dataset_name,
                 phase='train',
                 image_size=224,
                 n_frames=8,
                 config=None):

        assert phase in ['train', 'val', 'test']
        self.rawframes2_list, self.landmark_list, self.retina_list = init_aug_rawframe2(
            root_folder=
            "/home/jovyan/dataset/FaceForensics++/original_sequences/youtube/c23/rawframes_aug"
        )

        self.image_size = (image_size, image_size)

        self.phase = phase
        self.n_frames = n_frames

        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()
        self.frame_info_list = []

    def __len__(self):
        if self.phase == "train":
            return len(self.rawframes2_list)
        elif self.phase in ['val', 'test']:
            return len(self.rawframes2_list)

    def __getitem__(self, idx):
        try:
            image_paths = self.rawframes2_list[idx]
            landmark_paths = self.landmark_list[idx]
            retina_paths = self.retina_list[idx]

            frame_save_dir = os.path.dirname(image_paths[0]).replace(
                'original_sequences', 'manipulated_sequences').replace(
                    'rawframes_aug', 'rawframes2_random_margin_5frame')
            mask_save_dir = os.path.dirname(image_paths[0]).replace(
                'original_sequences',
                'manipulated_sequences').replace('/c23/', '/masks/').replace(
                    'rawframes_aug', 'rawframes2_random_margin_5frame')
            # retina_save_dir = os.path.dirname(image_paths[0]).replace(
            #     'original_sequences',
            #     'manipulated_sequences').replace('/rawframes_aug/', '/retina/')
            os.makedirs(frame_save_dir, exist_ok=True)
            os.makedirs(mask_save_dir, exist_ok=True)

            # os.makedirs(retina_save_dir, exist_ok=True)

            source_dict = {}
            image_dict = {}
            landmarks_list = []
            bbox_list = []
            num_frames_to_blend = (idx % 28) + 5
            hull_type = idx % 5
            # hull_type = 5
            if hull_type == 4:
                mask_region = random.randint(0, 6)
            else:
                mask_region = None
            center_bbox = np.load(retina_paths[15])
            center_landmark = np.load(landmark_paths[15])
            img = cv2.cvtColor(cv2.imread(image_paths[15]), cv2.COLOR_BGR2RGB)
            _, _, center_bbox, __, y0_new, y1_new, x0_new, x1_new = crop_face(
                img,
                center_landmark,
                center_bbox,
                margin=True,
                crop_by_bbox=False,
                abs_coord=True)
            for idx, frame in enumerate(image_paths):
                frame = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
                frame = frame[y0_new:y1_new, x0_new:x1_new]
                if idx == 0:
                    source_dict[f'image'] = frame
                else:
                    source_dict[f'image{str(idx).zfill(2)}'] = frame

            image_dict = source_dict.copy()

            for landmark_path in landmark_paths:
                landmark = np.load(landmark_path)
                landmark = self.reorder_landmark(landmark)
                landmark_cropped = np.zeros_like(landmark)
                for i, (p, q) in enumerate(landmark):
                    landmark_cropped[i] = [p - x0_new, q - y0_new]

                landmarks_list.append(landmark_cropped)

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
            _, _, _, __, y0_crop, y1_crop, x0_crop, x1_crop = self.crop_face(
                img=img,
                landmark=None,
                bbox=center_bbox,
                abs_coord=True,
                only_img=False)
            for idx, (src, img, landmark) in enumerate(
                    zip(source_list, image_list, landmarks_list)):

                img, img_blended, mask = self.self_blending(
                    src,
                    img,
                    landmark,
                    hull_type=hull_type,
                    mask_region=mask_region,
                )
                mask = mask * 255
                if idx not in frames_to_blend:
                    img_blended = img
                    mask = np.zeros_like(img, dtype=np.uint8)

                img_blended = img_blended[y0_crop:y1_crop, x0_crop:x1_crop]
                mask = mask[y0_crop:y1_crop, x0_crop:x1_crop]
                mask = cv2.resize(mask, (224, 224))
                img_blended = cv2.resize(img_blended, (224, 224))
                cv2.imwrite(
                    os.path.join(frame_save_dir, f'{str(idx).zfill(3)}.png'),
                    cv2.cvtColor(img_blended, cv2.COLOR_BGR2RGB))
                cv2.imwrite(
                    os.path.join(mask_save_dir, f'{str(idx).zfill(3)}.png'),
                    mask)
                # np.save(
                #     os.path.join(retina_save_dir, f'{str(idx).zfill(3)}.npy'),
                #     bbox)
            frame_info = f"Image Directory: {frame_save_dir}, Frames to Blend: {frames_to_blend}, Hull Type: {hull_type}, Mask Region: {mask_region}"
            with open('frame_info.txt', 'a') as file:
                file.write(frame_info + '\n')
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
            },
            p=1.)

    def randaffine(self, img, mask, bbox=None, mask_region=None):
        f = alb.Affine(translate_percent={
            'x': (-0.03, 0.03),
            'y': (-0.015, 0.015)
        },scale=[0.95, 1 / 0.95],fit_output=False,p=1)
        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            # alpha_affine=0,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask, bbox

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

    def self_blending(self,
                      source,
                      img,
                      landmark,
                      hull_type=None,
                      mask_region=None,
                      bbox=None):
        H, W = len(img), len(img[0])
        if np.random.rand() < 0.25:
            landmark = landmark[:68]
        if exist_bi:
            logging.disable(logging.FATAL)
            mask = random_get_hull(landmark,
                                   img,
                                   hull_type=hull_type,
                                   mask_region=mask_region)[:, :, 0]
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

        source, mask, _ = self.randaffine(source,
                                          mask,
                                          bbox=None,
                                          mask_region=None)

        img_blended, mask = B.dynamic_blend(source,
                                            img,
                                            mask,
                                            hull_type=hull_type)
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

    def collate_fn(self, batch):

        return None

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def crop_face(
        self,
        img,
        landmark=None,
        bbox=None,
        abs_coord=False,
        only_img=False,
    ):



        assert landmark is not None or bbox is not None

        H, W = len(img), len(img[0])

        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 4  # 0#np.random.rand()*(w/8)
        w1_margin = w / 4
        h0_margin = h / 4  # 0#np.random.rand()*(h/5)
        h1_margin = h / 4

        w0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        w1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        h0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        h1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()	

        y0_new = max(0, int(y0 - h0_margin))
        y1_new = min(H, int(y1 + h1_margin) + 1)
        x0_new = max(0, int(x0 - w0_margin))
        x1_new = min(W, int(x1 + w1_margin) + 1)

        img_cropped = img[y0_new:y1_new, x0_new:x1_new]
        if landmark is not None:
            landmark_cropped = np.zeros_like(landmark)
            for i, (p, q) in enumerate(landmark):
                landmark_cropped[i] = [p - x0_new, q - y0_new]
        else:
            landmark_cropped = None
        if bbox is not None:
            bbox_cropped = np.zeros_like(bbox)
            for i, (p, q) in enumerate(bbox):
                bbox_cropped[i] = [p - x0_new, q - y0_new]
        else:
            bbox_cropped = None

        if only_img:
            return img_cropped
        if abs_coord:
            return (
                img_cropped,
                landmark_cropped,
                bbox_cropped,
                (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1),
                y0_new,
                y1_new,
                x0_new,
                x1_new,
            )
        else:
            return (
                img_cropped,
                landmark_cropped,
                bbox_cropped,
                (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1),
            )


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
