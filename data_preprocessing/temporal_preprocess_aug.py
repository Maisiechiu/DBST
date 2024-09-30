import sys

sys.path.append("core")

from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.submission import get_cfg
from core.utils.misc import process_cfg
import datasets
from utils import flow_viz
from utils import frame_utils
import cv2
import math
import concurrent.futures
import os.path as osp
import traceback
import random
from core.FlowFormer import build_flowformer
import albumentations as alb
from utils.utils import InputPadder, forward_interpolate
import itertools
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
import warnings
import json
# warnings.filterwarnings("ignore")
torch.cuda.set_device(1)
# device = torch.device("cuda:1")
device = torch.device("cuda")
TRAIN_SIZE = [432, 960]


def reorder_landmark(landmark):
    landmark_add = np.zeros((13, 2))
    for idx, idx_l in enumerate(
        [77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
        landmark_add[idx] = landmark[idx_l]
    landmark[68:] = landmark_add
    return landmark


def crop_face(
    img,
    landmark=None,
    bbox=None,
    abs_coord=False,
    only_img=False,
):

    # crop face------------------------------------------
    H, W = len(img), len(img[0])

    assert landmark is not None or bbox is not None

    H, W = len(img), len(img[0])
    x0 = np.min(bbox[2:, 0])
    y0 = np.min(bbox[2:, 1])
    x1 = np.max(bbox[2:, 0])
    y1 = np.max(bbox[2:, 1])
    w = x1 - x0
    h = y1 - y0
    w_margin = w * 0.7
    h_margin = h * 0.7

    y0_new = max(0, int(y0 - h_margin))
    y1_new = min(H, int(y1 + h_margin) + 1)
    x0_new = max(0, int(x0 - w_margin))
    x1_new = min(W, int(x1 + w_margin) + 1)

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


## face detection , transformed imgae , save rawframes2 , call optical flow
def process(video, num_frames=8, context_range=16, face_detection_model=None):
    with torch.no_grad():
        try:
            frames = []
            cap = cv2.VideoCapture(video)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for cnt_frame in range(frame_count):
                ret_org, frame = cap.read()
                if not ret_org:
                    tqdm.write(
                        f'Frame read {cnt_frame} Error! : {os.path.basename(video)}'
                    )
                    continue
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) > num_frames:
                selected_anchor_frames_idx = np.linspace(context_range,
                                                         frame_count -
                                                         context_range - 1,
                                                         num_frames,
                                                         dtype=int)
            else:
                selected_anchor_frames_idx = range(len(frames))

            # every video rawframes2 dir, ex: 'original_sequences/youtube/raw/rawframes2/000'
            frame_dir = video.replace("videos",
                                      "rawframes2_train").replace(".mp4", "")
            # save this video's anchor frmaes' bbox(for whole image)
            bbox_dir = video.replace("videos",
                                     "rawframes2_train_retina").replace(
                                         ".mp4", "")
            img_tall_dir = frame_dir.replace("rawframes2_train",
                                             "rawframes2_train_tall")
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(bbox_dir, exist_ok=True)
            os.makedirs(img_tall_dir, exist_ok=True)
            for frame_idx in selected_anchor_frames_idx:
                # every sequence rawframes2 dir, ex: 'original_sequences/youtube/raw/rawframes2/000'
                sub_frame_dir = os.path.join(frame_dir,
                                             f"{str(frame_idx).zfill(3)}")
                bbox_path = os.path.join(bbox_dir,
                                         f"{str(frame_idx).zfill(3)}.npy")
                os.makedirs(sub_frame_dir, exist_ok=True)
                if len(os.listdir(sub_frame_dir)) >= 32:
                    continue

                frame = frames[frame_idx]

                # face detection
                retina_faces = face_detection_model.predict_jsons(frame)
                if retina_faces[0] == None:
                    continue

                bboxes = []
                size_list = []
                for face_idx in range(len(retina_faces)):
                    x0, y0, x1, y1 = retina_faces[face_idx]["bbox"]
                    bbox = np.array([[x0, y0], [x1, y1]] +
                                    retina_faces[face_idx]["landmarks"])
                    face_s = (x1 - x0) * (y1 - y0)
                    size_list.append(face_s)
                    bboxes.append(bbox)
                bboxes = np.concatenate(bboxes).reshape((len(size_list), ) +
                                                        bbox.shape)
                bboxes = bboxes[np.argsort(np.array(size_list))[::-1]]
                np.save(bbox_path, bboxes[0])

                ## crop frames
                start_frame = frame_idx - context_range
                end_frame = frame_idx + context_range
                assert start_frame >= 0
                assert end_frame < frame_count
                if start_frame < 0 or end_frame >= frame_count:
                    tqdm.write(
                        f"Frame range {start_frame}-{end_frame} out of bounds for {os.path.basename(video)}"
                    )
                    continue
                tall_img = np.zeros((6 * 224, 6 * 224, 3)).astype(np.uint8)

                for idx in range(start_frame, end_frame):
                    img = frames[idx]
                    img, _, _, __, y0_new, y1_new, x0_new, x1_new = crop_face(
                        img, None, bboxes[0], abs_coord=True)
                    img = cv2.resize(img, (224, 224))

                    tall_img[
                        (idx - start_frame) // 6 *
                        224:((idx - start_frame) // 6 + 1) * 224,
                        (idx - start_frame) % 6 *
                        224:((idx - start_frame) % 6 + 1) * 224,
                        :,
                    ] = img
                    img_path = os.path.join(
                        sub_frame_dir,
                        f"img_{str(idx -start_frame + 1).zfill(5)}.png")
                    cv2.imwrite(img_path, img[:, :, [2, 1, 0]])

                cv2.imwrite(
                    os.path.join(img_tall_dir,
                                 f"{str(frame_idx).zfill(3)}.png"),
                    tall_img[:, :, [2, 1, 0]])

        except Exception as e:
            with open("error_ori.txt", "a") as f:
                traceback.print_exc()
                error_message = f"Error in video {video} : {e}"
                f.write(error_message)
                f.write("\n")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_frames", default=8, type=int)
    parser.add_argument("--context_range", default=16, type=int)

    args = parser.parse_args()
    device = torch.device("cuda:1")
    face_detection_model = get_model("resnet50_2020-07-20",
                                     max_size=2048,
                                     device=device)
    face_detection_model.eval()

    video_dirs = [
        # "/home/jovyan/dataset/FaceForensics++/manipulated_sequences/Deepfakes/c23/videos" , \
        #           "/home/jovyan/dataset/FaceForensics++/manipulated_sequences/Face2Face/c23/videos", \
        #           "/home/jovyan/dataset/FaceForensics++/manipulated_sequences/FaceSwap/c23/videos", \
        #           "/home/jovyan/dataset/FaceForensics++/manipulated_sequences/NeuralTextures/c23/videos", \
                  "/home/jovyan/dataset/FaceForensics++/original_sequences/youtube/c23/videos"


                ]

    with open(file=os.path.join(
            os.path.join("/home/jovyan/dataset/FaceForensics++/",
                         "train.json")),
              mode="r") as f:
        train_json = json.load(f)
    train_list = []
    for d1, d2 in train_json:
        train_list.append(d1 + ".mp4")
        train_list.append(d2 + ".mp4")
        train_list.append(d1 + "_" + d2 + ".mp4")
        train_list.append(d2 + "_" + d1 + ".mp4")
    with open(file=os.path.join(
            os.path.join("/home/jovyan/dataset/FaceForensics++/",
                         "val.json")),
              mode="r") as f:
        train_json = json.load(f)
    for d1, d2 in train_json:
        train_list.append(d1 + ".mp4")
        train_list.append(d2 + ".mp4")
        train_list.append(d1 + "_" + d2 + ".mp4")
        train_list.append(d2 + "_" + d1 + ".mp4")
    videos = [
        os.path.join(video_dir, vid) for video_dir in video_dirs
        for vid in os.listdir(video_dir) if vid in train_list
    ]

    print(f"Processing {len(videos)} videos...")
    start_time = time.monotonic()

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for video in videos:
            with torch.no_grad():
                futures.append(
                    executor.submit(process, video, args.num_frames,
                                    args.context_range, face_detection_model))
        # Wait for all futures to complete and log any errors
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(videos)):
            # Print the current time
            try:
                future.result()
            except Exception as e:
                with open("error.txt", "a") as f:
                    traceback.print_exc()
                    error_message = f"Error in video {video} : {e}"
                    f.write(error_message)
                    f.write("\n")
                    executor.shutdown(wait=True)

        # End timer
        executor.shutdown(wait=True)
        end_time = time.monotonic()
        duration_minutes = (end_time - start_time) / 60
        print(f"Total duration: {duration_minutes:.2f} minutes")
