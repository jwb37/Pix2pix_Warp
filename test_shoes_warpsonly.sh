# V1 - use RTN (pre-trained VGGS, 3 iterations, unblurred flows)
# V2 - use advanced RTN (with fine-tuned VGGs, 4 iterations not 3, blurred flows)
# --continue_train --epoch_count 118 
python test.py --dataroot ./datasets/shoes --dataset_mode warped --warped_ds_input warped --model pix2pix --name shoes_p2p_mimicshoes_warpsonly --direction AtoB --input_nc 1 --output_nc 3 --epoch 130 --num_test 666 --no_flip
