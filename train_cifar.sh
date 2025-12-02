#python3 main_supcon.py --epochs 10 --save_freq 1 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar-10-100-10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --last_model_path "./save/SupCon/cifar-10-100-10_models/cifar-10-100-10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_0.0_1.0_0.05_trail_1_128_256/last.pth"



#python3 main_supcon.py --epochs 1 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar-10-100-10" --size 256 --trail 4 --temp 0.05 --method "SimCLR" --method_gama 0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "cutmix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --mixed_precision True


#python3 main_supcon.py --epochs 650 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar-10-100-10" --size 256 --trail 4 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --last_model_path "./save/SupCon/cifar-10-100-10_models/cifar-10-100-10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_0.0_1.0_0.05_trail_4_128_256/ckpt_epoch_150.pth"


#python3 main_supcon.py --epochs 600 --save_freq 10 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 



#python3 main_supcon.py --epochs 800 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 


# JobId=23353231
#python3 main_supcon.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3"


#python3 main_supcon.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3"


#python3 main_supcon.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 3 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3"


#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3"



#23382150
#python3 main_supcon_old.py --epochs 200 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug 0


#python3 main_supcon_old.py --epochs 10 --save_freq 2 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "2" --old_augmented true  --last_model_path "./save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2_SimCLR_0.0_1.0_0.05_trail_1_128_256_old_augmented/last.pth"


#python3 main_supcon_old.py --epochs 100 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "MoCo" --method_gama 0.0 --method_lam 1.0 --mixup_positive True --positive_method "cutmix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0



#python3 main_supcon_split.py --epochs 100 --save_freq 50 --batch_size 64 --moco_step 4 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "MoCo" --method_gama 0.0 --method_lam 1.0


#python3 main_supcon_split.py --epochs 100 --save_freq 50 --batch_size 64 --moco_step 4 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "MoCo" --method_gama 0.0 --method_lam 1.0 --mixup_positive True --positive_method "cutmix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0


#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 2 --mixup_positive False

#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 0.0 --randaug 2 --mixup_positive False

#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 3 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 3 --mixup_positive False

#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 3 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 0.0 --randaug 3 --mixup_positive False

python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 4 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive False

python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 4 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 0.0 --randaug 0 --mixup_positive False

python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive False

python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 0.0 --randaug 0 --mixup_positive False

