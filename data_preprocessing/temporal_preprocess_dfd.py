import sys

sys.path.append("core")
import argparse
import os
import time
import numpy as np
import torch
import cv2
import concurrent.futures
import traceback
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
import json

torch.cuda.set_device(1)
device = torch.device("cuda")


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
    phase = "train"
):


    # crop face------------------------------------------
    H, W = len(img), len(img[0])

    assert landmark is not None or bbox is not None

    H, W = len(img), len(img[0])


    x0, y0 = bbox[0]
    x1, y1 = bbox[1]
    w = x1 - x0
    h = y1 - y0
    w0_margin = w / 2  # 0#np.random.rand()*(w/8)
    w1_margin = w / 2
    h0_margin = h / 4  # 0#np.random.rand()*(h/5)
    h1_margin = h / 4


    if phase == "test":
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5

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




## face detection , transformed imgae , save rawframes2 , call optical flow
def process(video, num_frames=32, context_range=8, face_detection_model=None):
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
                                      "rawframes2_test").replace(".mp4", "")
            # save this video's anchor frmaes' bbox(for whole image)
            bbox_dir = video.replace("videos",
                                     "rawframes2_test_retina").replace(
                                         ".mp4", "")
            img_tall_dir = frame_dir.replace("rawframes2_test",
                                             "rawframes2_test_tall")
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(bbox_dir, exist_ok=True)
            os.makedirs(img_tall_dir, exist_ok=True)

            for frame_idx in selected_anchor_frames_idx:
                # every sequence rawframes2 dir, ex: 'original_sequences/youtube/raw/rawframes2/000'
                anchor_frame_dir = os.path.join(frame_dir,
                                                f"{str(frame_idx).zfill(3)}")

                frame = frames[frame_idx]

                # face detection
                retina_faces = face_detection_model.predict_jsons(frame)
                if len(retina_faces) == 0:
                    tqdm.write('No faces in {}:{}'.format(
                        frame_idx, os.path.basename(anchor_frame_dir)))
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

                    # Crop the face and resize it

                max_size = max(size_list)

                # Filter faces that are larger than half of the largest face
                filtered_bboxes = [
                    i for face_idx, i in enumerate(bboxes)
                    if size_list[face_idx] >= max_size / 2
                ]

                # Sort faces by size
                combined_list = list(zip(size_list, filtered_bboxes))
                combined_list_sorted = sorted(combined_list,
                                              key=lambda x: x[0],
                                              reverse=True)
                sorted_bboxes = [x[1] for x in combined_list_sorted]

                ## crop frames for each face id
                for face_id, face_bbox in enumerate(sorted_bboxes):
                    bbox_path = os.path.join(
                        bbox_dir,
                        f"{str(frame_idx).zfill(3)}_{str(face_id).zfill(3)}.npy"
                    )
                    face_bbox = np.concatenate(face_bbox).reshape(
                        (1, ) + face_bbox.shape)
                    np.save(bbox_path, face_bbox)  # 保存当前人脸的边界框

                    anchor_frame_face_dir = os.path.join(
                        anchor_frame_dir, f"{str(face_id).zfill(3)}")
                    os.makedirs(anchor_frame_face_dir, exist_ok=True)
                    if len(os.listdir(anchor_frame_face_dir)) >= 16:
                        continue

                    start_frame = frame_idx - context_range
                    end_frame = frame_idx + context_range
                    if start_frame < 0 or end_frame >= frame_count:
                        tqdm.write(
                            f"Frame range {start_frame}-{end_frame} out of bounds for {os.path.basename(video)}"
                        )
                        continue

                    tall_img = np.zeros((4 * 224, 4 * 224, 3)).astype(np.uint8)
                    for idx in range(start_frame, end_frame):
                        img = frames[idx]
                        img, _, _, __, y0_new, y1_new, x0_new, x1_new = crop_face(
                            img, None, face_bbox[0], abs_coord=True, phase="test")
                        img = cv2.resize(img, (224, 224))
                        tall_img[
                            (idx - start_frame) // 4 *
                            224:((idx - start_frame) // 4 + 1) * 224,
                            (idx - start_frame) % 4 *
                            224:((idx - start_frame) % 4 + 1) * 224,
                            :,
                        ] = img
                        img_path = os.path.join(
                            anchor_frame_face_dir,
                            f"img_{str(idx + 1).zfill(5)}.png")
                        cv2.imwrite(img_path, img[:, :, [2, 1, 0]])

                        cv2.imwrite(
                            os.path.join(
                                img_tall_dir,
                                f"{str(frame_idx).zfill(3)}_{str(face_id).zfill(3)}.png"
                            ), tall_img[:, :, [2, 1, 0]])

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
    parser.add_argument("--context_range", default=8, type=int)

    args = parser.parse_args()
    device = torch.device("cuda")

    video_dirs = ["/home/jovyan/dataset/FaceForensics++/original_sequences/actors/c23/videos" , \
                  "/home/jovyan/dataset/FaceForensics++/manipulated_sequences/DeepFakeDetection/c23/videos"]
    videos = sorted(os.listdir(args.video_dir))

    videos = [
        os.path.join(video_dir, vid) for video_dir in video_dirs
        for vid in os.listdir(video_dir) if vid.endswith(".mp4")
    ]
    print(f"Processing {len(videos)} videos...")
    args = parser.parse_args()
    device = torch.device("cuda")
    face_detection_model = get_model("resnet50_2020-07-20",
                                     max_size=2048,
                                     device=device)
    face_detection_model.eval()

    
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

