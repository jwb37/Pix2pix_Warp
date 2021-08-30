# Exp1 - use RTN (pre-trained VGGS, 3 iterations, unblurred flows)
# Exp2 - use advanced RTN (with fine-tuned VGGs, 4 iterations not 3, blurred flows)
# Exp3 - as above, but no blurs
# Exp4 - train on output from Mimic network instead of RTN network
python train.py --display_freq 1000 --dataroot ./datasets/shoes --dataset_mode warped --model pix2pix --name shoes_p2p_mimicshoes --direction AtoB --input_nc 2 --output_nc 3 --no_flip
