import concurrent.futures
import argparse
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
import dlib
from imutils import face_utils
from retinaface.pre_trained_models import get_model
import traceback
import time
import multiprocessing as mp


def crop_face(img, mask_frame, landmark=None, bbox=None):

    H, W = img.shape[:2]

    x0, y0 = bbox[0]
    x1, y1 = bbox[1]
    w = x1 - x0
    h = y1 - y0
    w0_margin = w
    w1_margin = w
    h0_margin = h / 2
    h1_margin = h / 2

    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin) + 1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)

    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    mask_cropped = mask_frame[y0_new:y1_new, x0_new:x1_new]
    landmark_cropped = [[p - x0_new, q - y0_new] for p, q in landmark]
    bbox_cropped = [[p - x0_new, q - y0_new] for p, q in bbox]

    return (img_cropped, mask_cropped, np.array(landmark_cropped), np.array(bbox_cropped))

def process_frame(frame_org, mask_frame_org, cnt_frame, dlib_face_detector, dlib_face_predictor, retina_predictor,
                  frames_path, lands_path, bboxes_path, masks_path):
    frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
    faces = dlib_face_detector(frame, 1)
    retina_faces = retina_predictor.predict_jsons(frame)
    try:
        if not faces or retina_faces[0]==None:
            tqdm.write(f'No faces in frame {cnt_frame} of {os.path.basename(frames_path)}')
            return

        image_path = os.path.join(frames_path, f'{cnt_frame:03d}.png')
        land_path = os.path.join(lands_path, f'{cnt_frame:03d}.npy')
        bbox_path = os.path.join(bboxes_path, f'{cnt_frame:03d}.npy')
        mask_path = os.path.join(masks_path, f'{cnt_frame:03d}.png')


        landmarks = []
        size_list = []
        for face_idx in range(len(faces)):
            landmark = dlib_face_predictor(frame, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
            x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
            face_s = (x1 - x0) * (y1 - y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        landmarks = np.concatenate(landmarks).reshape((len(size_list), ) +
                                                      landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]

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

        img_cropped, mask_cropped, landmarks_cropped, bboxes_cropped = crop_face(frame, mask_frame_org, landmarks[0], bboxes[0])



        cv2.imwrite(image_path, cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask_cropped)
        np.save(land_path, landmarks_cropped)
        np.save(bbox_path, bboxes_cropped)
    except:
        print("error in video {} frame {}", frame_org , cnt_frame)
        return
    
        
        
def facecrop(org_path, save_path, mask_org_path, mask_save_path, dlib_face_detector, dlib_face_predictor, retina_predictor):
 
    cap_org = cv2.VideoCapture(org_path)
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
    mask_org = cv2.VideoCapture(mask_org_path)
    
    
    frames_path, lands_path, bboxes_path = [
        os.path.join(save_path, subdir, os.path.basename(org_path).replace(".mp4", "/"))
        for subdir in ("rawframes/", "landmarks/", "retina/")
    ]
    masks_path = os.path.join(mask_save_path, 'rawframes/', os.path.basename(mask_org_path).replace(".mp4", "/"))
    
    for path in (frames_path, lands_path, bboxes_path, masks_path):
        os.makedirs(path, exist_ok=True)

    if len(os.listdir(frames_path)) >=frame_count_org: 
        return
    for cnt_frame in range(frame_count_org):
        ret_org, frame_org = cap_org.read()
        mask_ret_org, mask_frame_org = mask_org.read()
        if not ret_org:
            tqdm.write(f'Frame read {cnt_frame} Error! : {os.path.basename(org_path)}')
            continue

        process_frame(frame_org, mask_frame_org, cnt_frame, dlib_face_detector, dlib_face_predictor, retina_predictor,
                      frames_path, lands_path, bboxes_path, masks_path)

    cap_org.release()



if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        dest='dataset',
                        choices=[
                            'FaceShifter', 'Face2Face', 'Deepfakes',
                            'FaceSwap', 'NeuralTextures', 'Original'
                        ])
    parser.add_argument('-c',
                        dest='comp',
                        choices=['raw', 'c23', 'c40'],
                        default='c23')
    parser.add_argument('-n', dest='num_frames', type=int, default=32)
    args = parser.parse_args()
    if args.dataset == 'Original':
        dataset_path = '/home/jovyan/dataset/FaceForensics++/original_sequences/youtube/{}/'.format(
            args.comp)
    elif args.dataset in [
            'DeepFakeDetection', 'FaceShifter', 'Face2Face', 'Deepfakes',
            'FaceSwap', 'NeuralTextures'
    ]:
        dataset_path = '/mnt/sdc/maisie/FaceForensics++/manipulated_sequences/{}/{}/'.format(
            args.dataset, args.comp)
    
    
    device = torch.device("cuda")

    face_detector = dlib.get_frontal_face_detector()
    predictor_path = 'shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)

    model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    model.eval()

    movies_path = os.path.join(dataset_path, 'videos/')
    movies_path_list = sorted(glob(os.path.join(movies_path, '*.mp4')))
    
    mask_path = dataset_path.replace(args.comp , "masks")
    mask_path_list = [movie.replace(args.comp , "masks") for movie in movies_path_list]

    # for movie , mask in zip(movies_path_list , mask_path_list):
    #       facecrop(movie, dataset_path, mask, mask_path, face_detector, face_predictor, model)

    
    #start time
    start_time = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for movie , mask in zip(movies_path_list , mask_path_list):
            with torch.no_grad():
                futures.append(
                    executor.submit(
                        facecrop,
                        movie,
                        dataset_path,
                        mask,
                        mask_path,
                        face_detector,
                        face_predictor,
                        model,
                    ))
        # Wait for all futures to complete and log any errors
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(movies_path_list)):
            # Print the current time
            try:
                future.result()
            except Exception as e:
                traceback.print_exc()
        # End timer
        end_time = time.monotonic()
        duration_minutes = (end_time - start_time) / 60
        print(f"Total duration: {duration_minutes:.2f} minutes")

