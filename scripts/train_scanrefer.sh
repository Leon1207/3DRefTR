TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node 4 --master_port 3333 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /userhome/backup_lhj/zyx/dataset/pointcloud/data_for_eda/scannet_others_processed/ \
    --val_freq 3 --batch_size 8 --save_freq 3 --print_freq 500 \
    --lr_backbone=2e-3 --lr=2e-4 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir log/dks_1 \
    --lr_decay_epochs 50 75 \
    --pp_checkpoint /userhome/backup_lhj/zyx/dataset/pointcloud/data_for_eda/scannet_others_processed/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det \
    --max_epoch 80 \
    --model BeaUTyDETR_dks \
    --num_target 512 \
    --exp EDA_spunet \