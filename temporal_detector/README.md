## Setup
1. Create Envrioment
    ```bash
    cd temporal_detector
    ./install.sh
    ```
2. Makesure there's nothing wrog with mmaction2, please do the quick run following the instruction here [https://mmaction2.readthedocs.io/en/stable/get_started/quick_run.html#testing]

3. Create training and testing list : using rearrange.ipynb


## Training 
After training, `workdirs` will be created and the checkpoint can be found in there
```bash
cd mmaction2
python tools/test.py configs/deepfake/deepfake_margin.py 
```

## Testing
1. Predict
    ```bash
    cd mmaction2
    python tools/test.py configs/deepfake/deepfake_margin.py [checkpoint_file]
    ```
2. Taking the prediction results (yet the code is not elegant and needs improvement.)    
After running test.py, we get `current_date.txt`.    
The filename can be modified at `./mmaction2/mmaction/evaluation/metrics/auc_metric.py`.

