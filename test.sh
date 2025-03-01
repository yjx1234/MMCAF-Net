python test.py  --data_dir /home/vesselseg3/publicdata/Lung-pet-ct \
                --results_dir /home/vesselseg3/yujianxun/data_process1/results \
                --phase test \
                --dataset pe \
                --gpu_ids 0 \
		--num_slices=12 \
                \
                --ckpt_path /home/vesselseg3/yujianxun/train_result/Img_try_20250114_212843/ckpt/best30.pth.tar \
                --name penet_multiscale \
                
                
