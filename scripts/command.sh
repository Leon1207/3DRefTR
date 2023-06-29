## dataset preprocessing
python Pack_scan_files.py --scannet_data /userhome/backup_lhj/zyx/dataset/pointcloud/scannet_raw/scannet_cp_addAgg/ --data_root /userhome/backup_lhj/zyx/dataset/pointcloud/data_for_eda/scannet_others_processed/

## SR3D 
# train
sh scripts/train_sr3d.sh
sh scripts/train_sr3d_1gpu.sh

## ScanRefer 
# train
sh scripts/train_scanrefer.sh
sh scripts/train_scanrefer_single.sh
sh scripts/train_scanrefer_single_2gpu.sh
sh scripts/train_scanrefer_2gpu.sh
sh scripts/train_scanrefer_bertlite.sh
# test
sh scripts/test_scanrefer.sh
sh scripts/test_scanrefer_single.sh
sh scripts/test_scannet.sh
# measure latency
sh scripts/measure_latency_scanrefer.sh

## Nr3D 
# train
sh scripts/train_nr3d_2gpu.sh