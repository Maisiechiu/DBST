from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob
import os
import pandas as pd


def init_ff(phase, level='frame', n_frames=8):
    dataset_path = 'data/FaceForensics++/original_sequences/youtube/raw/frames/'

    image_list = []
    label_list = []

    folder_list = sorted(glob(dataset_path + '*'))
    filelist = []
    list_dict = json.load(open(f'data/FaceForensics++/{phase}.json', 'r'))
    for i in list_dict:
        filelist += i
    folder_list = [
        i for i in folder_list if os.path.basename(i)[:3] in filelist
    ]

    if level == 'video':
        label_list = [0] * len(folder_list)
        return folder_list, label_list
    for i in range(len(folder_list)):
        # images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
        images_temp = sorted(glob(folder_list[i] + '/*.png'))
        if n_frames < len(images_temp):
            images_temp = [
                images_temp[round(i)]
                for i in np.linspace(0,
                                     len(images_temp) - 1, n_frames)
            ]
        image_list += images_temp
        label_list += [0] * len(images_temp)

    return image_list, label_list


def collect_img_and_label(phase, dataset_name, frame_num, config):
    if phase == 'val':
        loading_phase = 'test'
    else:
        loading_phase = phase
    label_dict = {
        # DFD
        "DFD_fake": 1,
        "DFD_real": 0,
        "FF-FH": 1,
        "FF-DF": 1,
        "FF-F2F": 1,
        "FF-FS": 1,
        "FF-NT": 1,
        "FF-real": 0,
        # CelebDF
        "CelebDFv1_real": 0,
        "CelebDFv1_fake": 1,
        "CelebDFv2_real": 0,
        "CelebDFv2_fake": 1,
        # DFDCP
        "DFDCP_Real": 0,
        "DFDCP_FakeA": 1,
        "DFDCP_FakeB": 1,
        # DFDC
        "DFDC_Fake": 1,
        "DFDC_Real": 0,
        # DeeperForensics-1.0
        "DF_fake": 1,
        "DF_real": 0,
        # UADFV
        "UADFV_Fake": 1,
        "UADFV_Real": 0
    }

    label_list = []
    real_img_list = []
    fake_img_list = []
    img_list = []

    try:
        print(f"loading {dataset_name}....")
        with open(os.path.join('data', dataset_name + '.json'), 'r') as f:
            dataset_info = json.load(f)
    except Exception as e:
        print(e)
        raise ValueError(f'{dataset_name} json file not exist!')

    # Get the information for the current dataset
    for label in dataset_info[dataset_name]:

        sub_dataset_info = dataset_info[dataset_name][label][loading_phase]

        # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
        if dataset_name in [
                'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++',
                'DeepFakeDetection', 'FF-FH', "FaceForensics++_without_Deepfakes" , "FaceForensics++_without_Face2Face" , \
                "FaceForensics++_without_FaceSwap", "FaceForensics++_without_NeuralTextures", "FaceForensics++_robust",\
                "FaceForensics++_BW_1", "FaceForensics++_BW_2", "FaceForensics++_BW_3", "FaceForensics++_BW_4", "FaceForensics++_BW_5",\
                "FaceForensics++_CS_1", "FaceForensics++_CS_2", "FaceForensics++_CS_3" , "FaceForensics++_CS_4", "FaceForensics++_CS_5",\
                "FaceForensics++_CC_1", "FaceForensics++_GB_1", "FaceForensics++_GB_2", "FaceForensics++_GB_3", "FaceForensics++_GB_4", "FaceForensics++_GB_5",\
                "FaceForensics++_GNC_1", "FaceForensics++_GNC_2", "FaceForensics++_GNC_3", "FaceForensics++_GNC_4", "FaceForensics++_GNC_5",\
                "FaceForensics++_JPEG_1", "FaceForensics++_JPEG_2", "FaceForensics++_JPEG_3", "FaceForensics++_JPEG_4", "FaceForensics++_JPEG_5"
        ]:
            sub_dataset_info = sub_dataset_info[config['compression']]

        # Iterate over the videos in the dataset
        
        for video_name, video_info in sub_dataset_info.items():
            # Get the label and frame paths for the current video
            if video_info['label'] not in label_dict:
                
                raise ValueError(
                    f'Label {video_info["label"]} is not found in the configuration file.'
                )
            label = label_dict[video_info['label']]
            frame_paths = video_info['frames']
            frame_paths = sorted(frame_paths)
            # Select frame_num frames evenly distributed throughout the video
            total_frames = len(frame_paths)
            if phase == 'test':
                frame_num = total_frames

            # num of dataset frame > required frames
            if frame_num < total_frames:
                selected_frames = [
                    frame_paths[i] for i in np.linspace(
                        0, total_frames - 1, frame_num, dtype=int)
                ]
                if label == 0:
                    real_img_list.extend(selected_frames)
                    img_list.extend(selected_frames)
                    label_list.extend([label] * len(selected_frames))
                elif label == 1:
                    fake_img_list.extend(selected_frames)
                    img_list.extend(selected_frames)
                    if phase in ['test', 'val']:
                        label_list.extend([label] * len(selected_frames))
            # num of dataset frame < required frames
            else:
                if label == 0:
                    real_img_list.extend(frame_paths)
                    label_list.extend([label] * len(frame_paths))
                elif label == 1:
                    fake_img_list.extend(frame_paths)
                    if phase in ['test', 'val']:
                        label_list.extend([label] * len(frame_paths))

    return fake_img_list, real_img_list, label_list


def init_aug(
        root_folder="/home/jovyan/dataset/FaceForensics++/original_sequences/youtube/raw/rawframes2_norm1",
        target_count=33):
    rawframes2_list = []
    landmarks_list = []
    subfolders = os.listdir(root_folder)
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        if os.path.isdir(subfolder_path):
            for subsubfolder in os.listdir(subfolder_path):
                subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                landmarkfolder_path = subsubfolder_path.replace(
                    'rawframes2', 'rawframes2_landmarks')
                if os.path.isfile(subsubfolder_path.replace("rawframes2" , "rawframes2_retina")+".npy")\
                     and os.path.isdir(subsubfolder_path)\
                     and os.path.isdir(landmarkfolder_path):
                    file = [
                        os.path.join(subsubfolder_path, f)
                        for f in os.listdir(subsubfolder_path)
                    ]
                    landmark = [
                        os.path.join(landmarkfolder_path, f)
                        for f in os.listdir(landmarkfolder_path)
                    ]
                    if len(file) == 33 and len(landmark) == 33:
                        file.sort()
                        rawframes2_list.append(file)

                        landmark.sort()
                        landmarks_list.append(landmark)
                        with open('new_augment_data.txt', 'a') as f:
                            f.write(subsubfolder_path)
    return rawframes2_list, landmarks_list


if __name__ == '__main__':
    rawframes2_list, landmarks_list = init_aug()
    print(rawframes2_list[0], landmarks_list[0])
