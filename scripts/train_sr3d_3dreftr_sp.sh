TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node 2 --master_port 7777 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ~/DATA_ROOT/ \
    --val_freq 1 --batch_size 12 --save_freq 1 --print_freq 500 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset sr3d --test_dataset sr3d \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ~/DATA_ROOT/output/logs/ \
    --lr_decay_epochs 30 40 \
    --pp_checkpoint ~/DATA_ROOT/checkpoints/ckpt.pth \
    --butd_cls --self_attend \
    --checkpoint_path ~/DATA_ROOT/checkpoints/ckpt.pth \
    --max_epoch 140 \
    --small_lr \
    --model ThreeDRefTR_SP \
    --exp ThreeDRefTR_SP \