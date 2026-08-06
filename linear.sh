python3 main_linear.py --epochs 20 --print_freq 10 --learning_rate 0.01 --lr_decay_epochs "5, 10, 15" --model "vgg16" --datasets 'imagenet100' --backbone_model_direct "/save/SupCon/imagenet100_models/imagenet100_vgg16_original_data__vanilia__SimCLR_trail_0_128_256_split_128" --backbone_model_name "last.pth" --trail 0 --method "SimCLR"


python3 main_linear.py --epochs 20 --print_freq 10 --learning_rate 0.01 --lr_decay_epochs "5, 10, 15" --model "vgg16" --datasets 'imagenet100' --backbone_model_direct "/save/SupCon/imagenet100_models/imagenet100_vgg16_original_data__vanilia__Joint_0.6_0.4_trail_0_128_256_split_128" --backbone_model_name "ckpt_epoch_90.pth" --trail 0 --method "SimCLR"
