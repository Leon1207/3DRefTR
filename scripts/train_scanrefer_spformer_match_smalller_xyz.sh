TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node 4 --master_port 4444 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /userhome/backup_lhj/zyx/dataset/pointcloud/data_for_eda/scannet_others_processed/ \
    --val_freq 1 --batch_size 12 --save_freq 1 --print_freq 500 \
    --lr_backbone=2e-3 --lr=2e-4 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir log/spformer_731version_width_match_smalllr_xyzpos \
    --lr_decay_epochs 50 75 \
    --pp_checkpoint /userhome/backup_lhj/zyx/dataset/pointcloud/data_for_eda/scannet_others_processed/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det \
    --max_epoch 100 \
    --model BeaUTyDETR_spseg_width \
    --mask_loss \
    --exp EDA_spseg \
    --self_position_embedding xyz_learned \