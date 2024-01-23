python -m manydepth.train \
    --data_path /home/hkr/Dataset/iphone12_collect_data \
    --log_dir /home/hkr/Code/many_depth_vio_debug  \
    --dataset iphone \
    --split iphone \
    --height 384 \
    --width 768 \
    --batch_size 8 \
    --use_vio_pose_prior \
    --use_future_frame \
    --no_matching_augmentation \
    --disable_automasking
    # --load_weights_folder /home/hkr/Code/manydepth/checkpoints/KITTI_HR \
    # --mono_weights_folder /home/hkr/Code/manydepth/checkpoints/KITTI_HR \
    # --load_weights_folder /home/hkr/Code/many_depth_vio_debug/mdp/models/weights_14 \