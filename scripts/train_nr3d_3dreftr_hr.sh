TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node 4 --master_port 8888 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /userhome/backup_lhj/zyx/dataset/pointcloud/data_for_eda/scannet_others_processed/ \
    --val_freq 1 --batch_size 12 --save_freq 1 --print_freq 500 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset nr3d --test_dataset nr3d \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir log/eda_nr3d_spformer \
    --lr_decay_epochs 150 \
    --pp_checkpoint /userhome/backup_lhj/zyx/dataset/pointcloud/data_for_eda/scannet_others_processed/gf_detector_l6o256.pth \
    --butd_cls --self_attend \
    --checkpoint_path /userhome/lyd/3dvlm/log/NR3D_52_1.pth \
    --max_epoch 240 \
    --small_lr \
    --model ThreeDRefTR_HR \
    --exp ThreeDRefTR_HR \