# Data Preprocessing Guide

This guide provides instructions on how to preprocess datasets for temporal and spatial branches. 

---

## Temporal Preprocessing

Temporal preprocessing involves processing video data to uniform sample clips from video and detect face. Use the following scripts to preprocess different datasets.

```bash
python temporal_preprocess_ff.py          # Preprocess the FaceForensics++ (FF++) training set
python temporal_preprocess_ff_test.py     # Preprocess the FaceForensics++ (FF++) and Fsh test sets
python temporal_preprocess_dfd.py         # Preprocess the DeepFake Detection (DFD) dataset
python temporal_preprocess_dfdc.py        # Preprocess the DeepFake Detection Challenge (DFDC) dataset
python temporal_preprocess_cdf.py         # Preprocess the Celeb-DF (CDF) dataset
```

---

## Spatial Preprocessing
Spatial preprocessing focuses on cropping faces from video frames.
1. Crop Faces for Spatial Branch Inference
    ```bash
    python spatial_preprocess_test.py -d [dataset] -n [nframes]
    ```
    Parameters:             
    -d [dataset]: Specify the dataset to preprocess. Options are DFDC, FF, CDF, DFD.                
    -n [nframes]: Specify the number of frames to sample uniformly from each video.

2. Crop Faces for FaceForensics++ Training
    ```bash
    python crop_retina_dlib.py -d Deepfakes -c c23
    python crop_retina_dlib.py -d NeuralTextures -c c23
    python crop_retina_dlib.py -d Face2Face -c c23
    python crop_retina_dlib.py -d FaceSwap -c c23
    python crop_retina_dlib_no_mask -d Original -c c23
    ```