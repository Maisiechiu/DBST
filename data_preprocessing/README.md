# Data Preprocessing Guide

This guide provides instructions on how to preprocess datasets for temporal and spatial branches. 

## Dataset Structure
```

your space
├── FaceForensics++(DFD, FSh)
│   ├── original_sequences
│   │   ├── youtube
│   │   │   ├── raw
│   │   │   │   └── videos
│   │   │   │       └── *.mp4
│   │   │   └── c23
│   │   │       └── videos
│   │   │           └── *.mp4
│   │   └── actors
│   │       └── raw
│   │           └── videos
│   │               └── *.mp4
│   ├── manipulated_sequences
│   │   ├── Deepfakes
│   │   │   └── raw
│   │   │       └── videos
│   │   │           └── *.mp4
│   │   ├── Face2Face
│   │   │   └── raw
│   │   │       └── videos
│   │   │           └── *.mp4
│   │   ├── FaceSwap
│   │   │   └── raw
│   │   │       └── videos
│   │   │           └── *.mp4
│   │   ├── NeuralTextures
│   │   │   └── raw
│   │   │       └── videos
│   │   │           └── *.mp4
│   │   ├── FaceShifter
│   │   │   └── raw
│   │   │       └── videos
│   │   │           └── *.mp4
│   │   └── DeepFakeDetection
│   │       └── raw
│   │           └── videos
│   │               └── *.mp4
│   ├── train.json
│   ├── val.json
│   └── test.json
│
├── Celeb-DF-v2
│   ├── Celeb-real
│   │   └── videos
│   │       └── *.mp4
│   ├── Celeb-synthesis
│   │   └── videos
│   │       └── *.mp4
│   ├── Youtube-real
│   │   └── videos
│   │       └── *.mp4
│   └── List_of_testing_videos.txt
│
├── DFDC
│   ├── videos
│   │   └── *.mp4
│   └── labels.csv
│
├── DFDCP
│   ├── method_A
│   │   └── videos
│   │       ├── 643049
│   │       │   └── 643049_A
│   │       │       └── *.mp4
│   │       └── ...
│   ├── method_B
│   │   └── videos
│   │       ├── 1224068
│   │       │   └── 1224068_C
│   │       │       └── *.mp4
│   │       └── ...
│   ├── original_videos
│   │   └── videos
│   │       ├── 643049
│   │       │   └── *.mp4
│   │       └── ...
│   └── dataset.json
```

*Download link*             
[FaceForensics++ and DeepFakeDetection](https://github.com/ondyari/FaceForensics)  
[Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics)  
[DFDC and DFDCP](https://dfdc.ai/login)  
[FFIW](https://github.com/tfzhou/FFIW)  
[DeeperForensics](https://github.com/EndlessSora/DeeperForensics-1.0)



## Temporal Preprocessing

Temporal preprocessing involves processing video data to uniform sample clips from video and detect face. Use the following scripts to preprocess different datasets.

```bash
python temporal_preprocess_ff.py          # Preprocess the FaceForensics++ (FF++) training set
python temporal_preprocess_ff_test.py     # Preprocess the FaceForensics++ (FF++) and Fsh test sets
python temporal_preprocess_dfd.py         # Preprocess the DeepFake Detection (DFD) dataset
python temporal_preprocess_dfdc.py        # Preprocess the DeepFake Detection Challenge (DFDC) dataset
python temporal_preprocess_cdf.py         # Preprocess the Celeb-DF (CDF) dataset
```


## Spatial Preprocessing
Spatial preprocessing focuses on cropping faces from video frames.
1. Crop Faces for Spatial Branch Inference
    ```bash
    python spatial_preprocess_test.py -d [dataset] -n [nframes]
    ```
    Parameters:             
    -d [dataset]: Specify the dataset to preprocess. Options are DFDC, FF, CDF, DFD.                
    -n [nframes]: Specify the number of frames to sample uniformly from each video.

2. Crop Faces, mask and detect landmark for FaceForensics++ to Training         
- Download landmark detector (shape_predictor_81_face_landmarks.dat) from [here](https://github.com/codeniko/shape_predictor_81_face_landmarks) and place it in `data_preprocessing` folder.  
- Crop Faces, mask and detect landmark
    ```bash
    python crop_retina_dlib.py -d Deepfakes -c c23
    python crop_retina_dlib.py -d NeuralTextures -c c23
    python crop_retina_dlib.py -d Face2Face -c c23
    python crop_retina_dlib.py -d FaceSwap -c c23
    python crop_retina_dlib_no_mask -d Original -c c23
    ```