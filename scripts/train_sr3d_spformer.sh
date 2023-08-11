TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node 2 --master_port 7777 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /userhome/backup_lhj/zyx/dataset/pointcloud/data_for_eda/scannet_others_processed/ \
    --val_freq 1 --batch_size 12 --save_freq 1 --print_freq 500 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset sr3d --test_dataset sr3d \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir log/eda_sr3d_spformer \
    --lr_decay_epochs 30 40 \
    --pp_checkpoint /userhome/backup_lhj/zyx/dataset/pointcloud/data_for_eda/scannet_others_processed/gf_detector_l6o256.pth \
    --butd_cls --self_attend \
    --checkpoint_path /userhome/lyd/3dvlm/log/SR3D_68_1.pth \
    --max_epoch 140 \
    --mask_loss \
    --small_lr \
    --model BeaUTyDETR_spseg_width \
    --exp EDA_spseg_width \