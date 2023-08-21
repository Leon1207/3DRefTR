## 0. Installation

+ **(1)** Install environment with `environment.yml` file:
  ```
  conda env create -f environment.yml --name 3dreftr
  ```
  + or you can install manually:
    ```
    conda create -n 3dreftr python=3.7
    conda activate 3dreftr
    conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
    pip install numpy ipython psutil traitlets transformers termcolor ipdb scipy tensorboardX h5py wandb plyfile tabulate
    ```
+ **(2)** Install spacy for text parsing
  ```
  pip install spacy
  # 3.3.0
  pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0.tar.gz
  ```
+ **(3)** Compile pointnet++
  ```
  cd ~/3DRefTR
  sh init.sh
  ```
+ **(4)** Install segmentator from https://github.com/Karbo123/segmentator


## 1. Quick visualization demo 
We showing visualization via wandb for superpoints, kps points, bad case analyse, predict/ground_truth masks and box.
+ superpoints in 'src/joint_det_dataset.py' line 71
```
self.visualization_superpoint = False
```
+ others in 'src/groungd_evaluation.py' line 66 ~ 70
```
self.visualization_pred = False
self.visualization_gt = False
self.bad_case_visualization = False
self.kps_points_visualization = False
self.bad_case_threshold = 0.15
```

## 2. Data preparation

The final required files are as follows:
```
├── [DATA_ROOT]
│	├── [1] train_v3scans.pkl # Packaged ScanNet training set
│	├── [2] val_v3scans.pkl   # Packaged ScanNet validation set
│	├── [3] ScanRefer/        # ScanRefer utterance data
│	│	│	├── ScanRefer_filtered_train.json
│	│	│	├── ScanRefer_filtered_val.json
│	│	│	└── ...
│	├── [4] ReferIt3D/        # NR3D/SR3D utterance data
│	│	│	├── nr3d.csv
│	│	│	├── sr3d.csv
│	│	│	└── ...
│	├── [5] group_free_pred_bboxes/  # detected boxes (optional)
│	├── [6] gf_detector_l6o256.pth   # pointnet++ checkpoint (optional)
│	├── [7] roberta-base/     # roberta pretrained language model
│	├── [8] checkpoints/      # 3dreftr pretrained models
```

+ **[1] [2] Prepare ScanNet Point Clouds Data**
  + **1)** Download ScanNet v2 data. Follow the [ScanNet instructions](https://github.com/ScanNet/ScanNet) to apply for dataset permission, and you will get the official download script `download-scannet.py`. Then use the following command to download the necessary files:
    ```
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.ply
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.labels.ply
    python2 download-scannet.py -o [SCANNET_PATH] --type .aggregation.json
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.0.010000.segs.json
    python2 download-scannet.py -o [SCANNET_PATH] --type .txt
    ```
    where `[SCANNET_PATH]` is the output folder. The scannet dataset structure should look like below:
    ```
    ├── [SCANNET_PATH]
    │   ├── scans
    │   │   ├── scene0000_00
    │   │   │   ├── scene0000_00.txt
    │   │   │   ├── scene0000_00.aggregation.json
    │   │   │   ├── scene0000_00_vh_clean_2.ply
    │   │   │   ├── scene0000_00_vh_clean_2.labels.ply
    │   │   │   ├── scene0000_00_vh_clean_2.0.010000.segs.json
    │   │   ├── scene.......
    ```
  + **2)** Package the above files into two .pkl files(`train_v3scans.pkl` and `val_v3scans.pkl`):
    ```
    python Pack_scan_files.py --scannet_data [SCANNET_PATH] --data_root [DATA_ROOT]
    ```
+ **[3] ScanRefer**: Download ScanRefer annotations following the instructions [HERE](https://github.com/daveredrum/ScanRefer). Unzip inside `[DATA_ROOT]`.
+ **[4] ReferIt3D**: Download ReferIt3D annotations following the instructions [HERE](https://github.com/referit3d/referit3d). Unzip inside `[DATA_ROOT]`.
+ **[5] group_free_pred_bboxes**: Download [object detector's outputs](https://1drv.ms/u/s!AsnjK0KGPk10gYBjpUjJm7TkADS8vg?e=1AXJdR). Unzip inside `[DATA_ROOT]`. (not used in single-stage method)
+ **[6] gf_detector_l6o256.pth**: Download PointNet++ [checkpoint](https://1drv.ms/u/s!AsnjK0KGPk10gYBXZWDnWle7SvCNBg?e=SNyUK8) into `[DATA_ROOT]`.
+ **[7] roberta-base**: Download the roberta pytorch model:
  ```
  cd [DATA_ROOT]
  git clone https://huggingface.co/roberta-base
  cd roberta-base
  rm -rf pytorch_model.bin
  wget https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
  ```
+ **[8] checkpoints**: Our pre-trained models (see 3. Models).
+ **[9] ScanNetv2**: Prepare the preporcessed ScanNetv2 dataset follow "Data Preparation" section from https://github.com/sunjiahao1999/SPFormer, obtaining the dataset file with the following structure:
```
ScanNetv2
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```
+ **[10] superpoints**: Prepare superpoints for each scene preprocessed from Step. 9.
  ```
  cd [DATA_ROOT]
  python superpoint_maker.py  # modify data_root & split
  ```

## 3. Models

|Dataset/Model  | REC mAP@0.25 | RES mIoU | Model |
|:---:|:---:|:---:|:---:|
|ScanRefer/3DRefTR-SP| 55.45 | 40.76 |[GoogleDrive](https://drive.google.com/file/d/1--489HTfjOCuK6ibQ92G2Blm09eRTnfL/view?usp=sharing)
|ScanRefer/3DRefTR-SP (Single-Stage)| 54.43 | 40.23 |[GoogleDrive](https://drive.google.com/file/d/1kMWl4XfRfw9aVVmGektO5CYaHag3kIUI/view?usp=sharing)
|ScanRefer/3DRefTR-HR| 55.04 | 41.24 |[GoogleDrive](https://drive.google.com/file/d/1gThEp7QwnlCioUyTEQxT3jbsUyCM5BSK/view?usp=sharing)
|ScanRefer/3DRefTR-HR (Single-Stage)| 54.40 | 40.75 |[GoogleDrive](https://drive.google.com/file/d/1qCybMQCnhuvikn9O-H90S9l82KRi1Tzq/view?usp=sharing)
|SR3D/3DRefTR-SP | 68.45 | 44.61 | [GoogleDrive](https://drive.google.com/file/d/1AtqDwpVVAEHDtkiuJnC9EFpGK48nEh9s/view?usp=sharing) 
|NR3D/3DRefTR-SP | 52.55 | 36.17 | [GoogleDrive](https://drive.google.com/file/d/1Y4SXRz3snPIeRxzCVMqs3AMfWC7JyGVW/view?usp=sharing) 

## 4. Training

+ Please specify the paths of `--data_root`, `--log_dir`, `--pp_checkpoint` in the `train_*.sh` script first.
+ For **ScanRefer** training
  ```
  sh scripts/train_scanrefer_3dreftr_hr.sh
  sh scripts/train_scanrefer_3dreftr_sp.sh
  ```
+ For **ScanRefer (single stage)** training
  ```
  sh scripts/train_scanrefer_3dreftr_hr_single.sh
  sh scripts/train_scanrefer_3dreftr_sp_single.sh
  ```
+ For **SR3D** training
  ```
  sh scripts/train_sr3d_3dreftr_hr.sh
  sh scripts/train_sr3d_3dreftr_sp.sh
  ```
+ For **NR3D** training
  ```
  sh scripts/train_nr3d_3dreftr_hr.sh
  sh scripts/train_nr3d_3dreftr_sp.sh
  ```

## 5. Evaluation

+ Please specify the paths of `--data_root`, `--log_dir`, `--checkpoint_path` in the `test_*.sh` script first.
+ For **ScanRefer** evaluation
  ```
  sh scripts/test_scanrefer_3dreftr_hr.sh
  sh scripts/test_scanrefer_3dreftr_sp.sh
  ```
+ For **ScanRefer (single stage)** evaluation
  ```
  sh scripts/test_scanrefer_3dreftr_hr_single.sh
  sh scripts/test_scanrefer_3dreftr_sp_single.sh
  ```
+ For **SR3D** evaluation
  ```
  sh scripts/test_sr3d_3dreftr_hr.sh
  sh scripts/test_sr3d_3dreftr_sp.sh
  ```
+ For **NR3D** evaluation
  ```
  sh scripts/test_nr3d_3dreftr_hr.sh
  sh scripts/test_nr3d_3dreftr_sp.sh
  ```

## 6. Acknowledgements

We are quite grateful for [EDA](https://github.com/yanmin-wu/EDA), [SPFormer](https://github.com/sunjiahao1999/SPFormer), [BUTD-DETR](https://github.com/nickgkan/butd_detr), [GroupFree](https://github.com/zeliu98/Group-Free-3D), [ScanRefer](https://github.com/daveredrum/ScanRefer), and [SceneGraphParser](https://github.com/vacancy/SceneGraphParser).

## 7. Citation

If you find our work useful in your research, please consider citing:
```

```