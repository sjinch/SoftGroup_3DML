SoftGroup_3DML
================
# Dataset
Considering remaining capacity of our desktop, we only utilize S3DIS Dataset.
Already preprocessed (Downsampling)

# Implementation details
We use all modules, pretrained HAIS checkpoint, bottom-up and top-down refinement codes from author's codes.
We only redefine some of the details in MLP and Loss function. The number of MLP layers is defined as 2 but we figured out that it is quite shallow to learn point-wise semantic scores and offsets. Also we put more weights on semantic loss and offset loss which we believe has more significant impact on performance. 

1. the number of MLP layers : 2 -> 3 
2. Loss function : multi-task loss -> weighted multi-task loss

# Usage

## spconv, dependency installation and setup
```python
pip install spconv-cu102
pip install -r requirements.txt
sudo apt-get install libsparsehash-dev
python setup.py build_ext develop
```

## train
### fine-tuning backbone networks
```python
python train.py --config configs/softgroup_s3dis_backbone_fold5.yaml --work_dir 1 --skip_validate
```
