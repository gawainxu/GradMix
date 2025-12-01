#23425843
#python3 main_supcon_split.py --epochs 200 --save_freq 1 --batch_size 256 --learning_rate 0.0008 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --grad_splits 32 --last_model_path "./save/SupCon/imagenet100_models/imagenet100_resnet18_original_data__vanilia__SimCLR_1.0_trail_0_128_256_split_32/ckpt_epoch_77.pth"


#python3 main_supcon_old.py --epochs 400 --save_freq 10 --print_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3,4" --old_augmented true --last_model_path "./save/SupCon/imagenet100_models/imagenet100_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3,4_SimCLR_0.0_1.0_0.05_trail_0_128_256_old_augmented/ckpt_epoch_70.pth"


#python3 main_supcon_split.py --epochs 200 --save_freq 10 --batch_size 128 --moco_step 4 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100" --size 256 --trail 0 --temp 0.05 --method "MoCo" --method_gama 0.0 --method_lam 1.0


#python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --mixup_positive True --positive_method "cutmix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0


#python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --mixup_positive True --positive_method "random" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0


#python3 main_supcon_old.py --epochs 200 --save_freq 10 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 0.0 --mixup_positive False --num_workers 2

###
#python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --mixup_positive False --num_workers 2

###
#python3 main_supcon_old.py --epochs 40 --save_freq 10 --batch_size 200 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "2,3,4" --old_augmented true  --last_model_path "./save/SupCon/imagenet100_m_models/imagenet100_m_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.0_0.05_trail_0_128_200_old_augmented/ckpt_epoch_60.pth"


#python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 230 --learning_rate 0.0001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "2,3,4" --old_augmented true  --last_model_path "./save/SupCon/imagenet100_m_models/imagenet100_m_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.0_0.05_trail_0_128_230_old_augmented/ckpt_epoch_70.pth"


#python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "4" --old_augmented true


python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 230 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 0.5 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "4" --old_augmented true  



python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 200 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 0.0 --mixup_positive True --positive_method False


python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 200 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --mixup_positive False



python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 230 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 0.0 --mixup_positive True --positive_method False


python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 230 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --mixup_positive False



###
python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 200 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "2,3,4" --old_augmented true


python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 200 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --mixup_positive True --positive_method "random" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0


python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 200 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --mixup_positive True --positive_method "random" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0

