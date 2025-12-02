#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 5 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --augmix true

#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 5 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3,4" --augmix true


#python3 main_supcon_split.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --augmentation_method "mixup_positive" --mixup_positive true --positive_method "cutmix" --grad_splits 32

#python3 main_supcon_split.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --augmentation_method "mixup_positive" --mixup_positive true --positive_method "random" --grad_splits 32


#python3 main_supcon_split.py --epochs 200 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --grad_splits 32



python3 main_linear.py --epochs 30 --print_freq 10 --learning_rate 0.01 --lr_decay_epochs "5, 10, 20" --model "resnet18" --datasets 'imagenet100' --backbone_model_direct "/save/SupCon/imagenet100_models/imagenet100_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3,4_MoCo_trail_0_128_256_old_augmented" --backbone_model_name "ckpt_epoch_30.pth" --trail 0 --method "MoCo"
