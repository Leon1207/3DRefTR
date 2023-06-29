TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port 1111 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /userhome/backup_lhj/zyx/dataset/pointcloud/data_for_eda/scannet_others_processed/ \
    --val_freq 3 --batch_size 12 --save_freq 3 --print_freq 500 \
    --lr_backbone=2e-3 --lr=2e-4 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir log/ \
    --lr_decay_epochs 50 75 \
    --butd --det_ref --self_attend --augment_det \
    --checkpoint_path /userhome/backup_lhj/lhj/pointcloud/EDA-master/log/scanrefer/scanrefer_2gpu/1681527693/ckpt_epoch_70.pth \
    --model BeaUTyDETR \
    --eval \
    --exp scanrefer_2gpu_run2_test/
