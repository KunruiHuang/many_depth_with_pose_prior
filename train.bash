python -m manydepth.train \
    --data_path /home/hkr/Dataset/iphone12_collect_data \
    --log_dir /home/hkr/Code/many_depth_log  \
    --dataset iphone \
    --split iphone \
    --height 384 \
    --width 768 \
    --batch_size 8 \
    --use_vio_pose_prior