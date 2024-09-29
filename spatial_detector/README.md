## Setup
1. Dataset: follow the instruction of data_preprocessing here [https://github.com/Maisiechiu/DBST/tree/main/data_preprocessing]
2. Rearrange: create json file of training and inference sample.
3. Create Environment

## Training
Run the training:
```bash
cd spatial_detector
CUDA_VISIBLE_DEVICES=* python3 src/train_sbi.py \
src/configs/sbi/base.json \
-n sbi
```
Top five checkpoints will be saved in `./output/` folder. As described in our paper, we use the latest one for evaluations.

## Testing
```bash
cd spatial_detector
CUDA_VISIBLE_DEVICES=* python3 src/inference_dataset.py \
src/configs/sbi/inference.json \
-n sbi
```

The code primiary from SBI [https://github.com/open-mmlab/mmaction2.git]