#python3 main_linear.py --epochs 30 --learning_rate 0.001 --lr_decay_epochs "10" --model "resnet18" --datasets "imagenet100" --trail 0 --backbone_model_direct "/save/SupCon/imagenet100_models/imagenet100_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_3_SimCLR_0.0_1.0_0.05_trail_0_128_256/" --backbone_model_name "ckpt_epoch_70.pth"



#python3 main_linear.py --epochs 30 --learning_rate 0.001 --lr_decay_epochs "10,20" --model "resnet18" --datasets "tinyimgnet" --trail 5 --backbone_model_direct "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_mixup_positive_alpha_10_beta_0.3_cv2saliency_3_SimCLR_0.0_1.0_0.05_trail_5_128_256/" --backbone_model_name "ckpt_epoch_100.pth"


#python3 main_supcon_old.py --epochs 10 --save_freq 2 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive True --old_augmented True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3,4" --last_model_path "./save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3,4_SimCLR_0.0_1.0_0.05_trail_1_128_256_old_augmented/last.pth"

#python3 main_supcon_old.py --epochs 10 --save_freq 2 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive True --old_augmented True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3,4" --last_model_path "./save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3,4_SimCLR_0.0_1.0_0.05_trail_2_128_256_old_augmented/last.pth"

#python3 main_supcon_old.py --epochs 10 --save_freq 2 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 3 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive True --old_augmented True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3,4" --last_model_path "./save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3,4_SimCLR_0.0_1.0_0.05_trail_3_128_256_old_augmented/last.pth"

#python3 main_supcon_old.py --epochs 10 --save_freq 2 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 4 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive True --old_augmented True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3,4" --last_model_path "./save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3,4_SimCLR_0.0_1.0_0.05_trail_4_128_256_old_augmented/last.pth"


#python3 main_supcon_split.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --grad_splits 32


python3 main_linear.py --epochs 30 --print_freq 10 --learning_rate 0.01 --lr_decay_epochs "5, 10, 20" --model "resnet18" --datasets 'imagenet100' --backbone_model_direct "/save/SupCon/imagenet100_models/imagenet100_resnet18_original_data__mixup_positive_cutmix_intra_True_alpha_0.2_p_0.5_MoCo_trail_0_128_64_split_64" --backbone_model_name "ckpt_epoch_30.pth" --trail 0 --method "MoCo"
