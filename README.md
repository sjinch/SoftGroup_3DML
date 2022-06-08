SoftGroup_3DML
================
# Dataset
Considering remaining capacity of our desktop, we only utilize S3DIS Dataset.
## Dataset preparation
```python
bash prepare_data.sh # from author's code
```

# Implementation details
We use all modules, config files, dataset preprocessing, train, test, bottom-up and top-down refinement codes from [author's codes](https://github.com/thangvubk/SoftGroup).
We use pretrained model from [HAIS checkpoint](https://github.com/hustvl/HAIS)
We only redefine some of the details in MLP and Loss function. The number of MLP layers is defined as 2 but we figured out that it is quite shallow to learn point-wise semantic scores and offsets. Also we put more weights on semantic loss and offset loss which we believe has more significant impact on performance. 

## modifications that we perform
1. The number of MLP layers : 2 -> 3 (we thought 2 is not enough to learn per-point features such as semantic scores and offset coordinates)
2. Activation function for MLP : ReLU -> LeakyReLU (To prevent side effect stemmed from the increased layers)
3. Loss function : multi-task loss -> weighted multi-task loss (details on the report)
4. Epochs : 20 -> 8 (to reduce training time)
# Usage

## spconv, dependency installation and setup
```python
pip install spconv-cu102 # we used cuda 11.3 version
pip install -r requirements.txt
sudo apt-get install libsparsehash-dev
python setup.py build_ext develop
```

## train
### fine-tuning backbone networks
```python
python train.py --config configs/softgroup_s3dis_backbone_fold5.yaml --work_dir $WORK_DIR --skip_validate # from author's code
python train.py --config configs/softgroup_s3dis_fold5.yaml --work_dir $WORK_DIR --skip_validate # from author's code
```

## test

```python
python test.py --config configs/softgroup_s3dis_fold5.yaml --out $RESULT --checkpoint $CHECKPOINT # from author's code
```
