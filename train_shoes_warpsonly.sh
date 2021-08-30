# --continue_train --epoch_count 118 
python train.py --display_freq 1000 --dataroot ./datasets/shoes --dataset_mode warped --warped_ds_input warped --model pix2pix --name shoes_p2p_mimicshoes_warpsonly --direction AtoB --input_nc 1 --output_nc 3 --no_flip
