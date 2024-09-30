## Setup
1. Dataset: follow the instruction of data_preprocessing here [https://github.com/Maisiechiu/DBST/tree/main/data_preprocessing]
2. Create training and testing list : using rearrange.ipynb
3. Create Environment
```bash
cd spatial_detector
./install.sh
```
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
1. predict
    ```bash
    cd spatial_detector
    CUDA_VISIBLE_DEVICES=* python3 src/inference_dataset.py \
    src/configs/sbi/inference.json \
    -n sbi
    ```
2. Taking the prediction results (yet the code is not elegant and needs improvement.)    
After running test.py, we get `{checkpoint'}_{dataset_name}.txt", 'w') as f:`.    
The filename can be modified at `./src/inference_dataset.py`.

The code primiary from SBI [https://github.com/mapooon/SelfBlendedImages/tree/master]