#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "4"


#python3 main_supcon_old.py --epochs 200 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 5 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3,4" --augmix true --old_augmented true --last_model_path "./save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3,4_SimCLR_0.0_1.0_0.05_trail_5_128_256_augmix_old_augmented/last.pth"


#python3 main_supcon_old.py --epochs 10 --save_freq 2 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "2,3" --old_augmented true  --last_model_path "./save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3_SimCLR_0.0_1.0_0.05_trail_1_128_256_old_augmented/last.pth"


#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive False

#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 0.0 --randaug 0 --mixup_positive False


#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive False

python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 0.0 --randaug 0 --mixup_positive False








