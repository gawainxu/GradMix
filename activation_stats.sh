#python3 activation_map_stats.py --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_0_128_256_twostage/ckpt_epoch_2.pth" --trail 0 --threshold 1e-4

#python3 activation_map_stats.py --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__vanilia__SimCLR_1.0_1.2_0.05_trail_0_128_256_twostage/ckpt_epoch_2.pth" --trail 0 --threshold 1e-4

#python3 activation_map_stats.py --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_2_128_256_twostage/ckpt_epoch_2.pth" --trail 2 --threshold 1e-4

#python3 activation_map_stats.py --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__vanilia__SimCLR_1.0_1.2_0.05_trail_2_128_256_twostage/ckpt_epoch_2.pth" --trail 2 --threshold 1e-4

#python3 activation_map_stats.py --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_3_128_256_twostage/ckpt_epoch_2.pth" --trail 3 --threshold 1e-4

#python3 activation_map_stats.py --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__vanilia__SimCLR_1.0_1.2_0.05_trail_3_128_256_twostage/ckpt_epoch_2.pth" --trail 3 --threshold 1e-4

#python3 activation_map_stats.py --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_4_128_256_twostage/ckpt_epoch_2.pth" --trail 4 --threshold 1e-4

#python3 activation_map_stats.py --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__vanilia__SimCLR_1.0_1.2_0.05_trail_4_128_256_twostage/ckpt_epoch_2.pth" --trail 4 --threshold 1e-4



#python3 main_supcon.py --epochs 10 --save_freq 2 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --grad_splits 32 --positive_method "random" --mixup_positive true --augmentation_method "mixup_positive" --alpha_vanilla 0.1 --beta_vanilla 0.1 --last_model_path "./save/SupCon/cifar10_models/cifar10_resnet18_1_original_data__mixup_positive_alpha_0.1_beta_0.1_random_no_SimCLR_1.0_1.2_0.05_trail_2_128_256/last.pth" --grad_splits 1


python3 activation_map_stats.py --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256/ckpt_epoch_600.pth" --trail 0 --threshold 1e-3 --datasets "tinyimgnet" --num_classes 20

python3 activation_map_stats.py --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__vanilia__SimCLR_1.0_1.0_0.05_trail_1_128_256/ckpt_epoch_600.pth" --trail 1 --threshold 1e-3 --datasets "tinyimgnet" --num_classes 20

python3 activation_map_stats.py --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__vanilia__SimCLR_1.0_1.0_0.05_trail_2_128_256/ckpt_epoch_600.pth" --trail 2 --threshold 1e-3 --datasets "tinyimgnet" --num_classes 20

python3 activation_map_stats.py --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__vanilia__SimCLR_1.0_1.0_0.05_trail_3_128_256/ckpt_epoch_600.pth" --trail 3 --threshold 1e-3 --datasets "tinyimgnet" --num_classes 20

python3 activation_map_stats.py --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__vanilia__SimCLR_1.0_1.0_0.05_trail_4_128_256/ckpt_epoch_600.pth" --trail 4 --threshold 1e-3 --datasets "tinyimgnet" --num_classes 20


python3 activation_map_stats.py --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_0_128_256/ckpt_epoch_600.pth" --trail 0 --threshold 1e-3 --datasets "tinyimgnet" --num_classes 20

python3 activation_map_stats.py --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_1_128_256/ckpt_epoch_600.pth" --trail 1 --threshold 1e-3 --datasets "tinyimgnet" --num_classes 20


python3 activation_map_stats.py --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_2_128_256/ckpt_epoch_600.pth" --trail 2 --threshold 1e-3 --datasets "tinyimgnet" --num_classes 20

python3 activation_map_stats.py --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_3_128_256/ckpt_epoch_600.pth" --trail 3 --threshold 1e-3 --datasets "tinyimgnet" --num_classes 20


python3 activation_map_stats.py --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_4_128_256/ckpt_epoch_600.pth" --trail 4 --threshold 1e-3 --datasets "tinyimgnet" --num_classes 20




