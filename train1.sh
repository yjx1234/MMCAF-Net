python train1.py --data_dir=/home/vesselseg3/publicdata/Lung-pet-ct \
                --save_dir=/home/vesselseg3/yujianxun/train_result \
                --ckpt_path= \
                \
                --name=ISBI_total \
                --model=ISBI_1 \
                --batch_size=4 \
                --gpu_ids=0 \
                --iters_per_print=8 \
                --iters_per_visual=8000 \
                --learning_rate=1e-4 \
                --lr_decay_step=600000 \
                --lr_scheduler=cosine_warmup \
                --num_epochs=50 \
                --num_slices=12 \
                --weight_decay=1e-3 \
                \
                --phase=train \
                \
                --agg_method=max \
                --best_ckpt_metric=val_loss \
                --crop_shape=192,192 \
                --cudnn_benchmark=False \
                --dataset=pe \
                --do_classify=True \
                --epochs_per_eval=1 \
                --epochs_per_save=1 \
                --fine_tune=False \
                --fine_tuning_boundary=classifier \
                --fine_tuning_lr=1e-2 \
                --include_normals=True \
                --lr_warmup_steps=10000 \
                --model_depth=50 \
                --num_classes=1 \
                --num_visuals=8 \
                --num_workers=4 \
                --optimizer=sgd \
                --pe_types='["central","segmental"]' \
                --resize_shape=192,192 \
                --sgd_dampening=0.9 \
                --sgd_momentum=0.9 \
                --use_pretrained=False \
                
                
		
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
