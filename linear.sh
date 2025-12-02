python3 main_linear.py --epochs 30 --print_freq 10 --learning_rate 0.01 --lr_decay_epochs "5, 10, 20" --model "resnet18" --datasets 'tinyimgnet' --backbone_model_direct "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_5_128_256" --backbone_model_name "last.pth" --trail 5 --method "SimCLR"


#python3 main_linear.py --epochs 30 --print_freq 10 --learning_rate 0.01 --lr_decay_epochs "5, 10, 20" --model "resnet18" --datasets 'tinyimgnet' --backbone_model_direct "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3,4_SimCLR_1.0_1.0_0.05_trail_5_128_256_old_augmented" --backbone_model_name "last.pth" --trail 5 --method "SimCLR"


#python3 main_linear.py --epochs 30 --print_freq 10 --learning_rate 0.01 --lr_decay_epochs "5, 10, 20" --model "resnet18" --datasets 'imagenet100' --backbone_model_direct "/save/SupCon/imagenet100_models/imagenet100_resnet18_original_data__mixup_positive_random_intra_True_alpha_0.2_p_0.5_SimCLR_1.0_trail_0_128_256_split_32" --backbone_model_name "ckpt_epoch_100.pth" --trail 0

#23441585
#python3 main_linear.py --epochs 30 --print_freq 10 --learning_rate 0.01 --lr_decay_epochs "5, 10, 20" --model "resnet18" --datasets 'imagenet100' --backbone_model_direct "/save/SupCon/imagenet100_models/imagenet100_resnet18_original_data__vanilia__MoCo_trail_0_128_64_split_64" --backbone_model_name "ckpt_epoch_30.pth" --trail 0 --method "MoCo"


#python3 main_linear.py --epochs 30 --print_freq 10 --learning_rate 0.01 --lr_decay_epochs "5, 10, 20" --model "resnet18" --datasets 'cifar10' --backbone_model_direct "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_cutmix_intra_True_alpha_0.2_p_0.5_MoCo_trail_1_128_64_split_64" --backbone_model_name "last.pth" --trail 1 --method "MoCo"

#python3 main_linear.py --epochs 30 --print_freq 10 --learning_rate 0.01 --lr_decay_epochs "5, 10, 20" --model "resnet18" --datasets 'cifar10' --backbone_model_direct "/save/SupCon/cifar10_models/cifar10_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_1_128_256" --backbone_model_name "last.pth" --trail 1 --method "SimCLR"
#python3 main_linear.py --epochs 30 --print_freq 10 --learning_rate 0.01 --lr_decay_epochs "5, 10, 20" --model "resnet18" --datasets 'cifar10' --backbone_model_direct "/save/SupCon/cifar10_models/cifar10_resnet18_vanilia__SimCLR_1.0_1.2_0.05_trail_1_128_256" --backbone_model_name "last.pth" --trail 1 --method "SimCLR"
#python3 main_linear.py --epochs 30 --print_freq 10 --learning_rate 0.01 --lr_decay_epochs "5, 10, 20" --model "resnet18" --datasets 'cifar10' --backbone_model_direct "/save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_4_SimCLR_1.0_1.2_0.05_trail_1_128_256_twostage" --backbone_model_name "last.pth" --trail 1 --method "SimCLR"
