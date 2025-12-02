# 23353523
#python3 main_supcon.py --epochs 800 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3,4"

#python3 main_supcon.py --epochs 800 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3,4"

# 23354346
#python3 main_supcon.py --epochs 400 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3,4"  --last_model_path "./save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_3,4_SimCLR_1.0_1.0_0.05_trail_2_128_256"

#python3 main_supcon.py --epochs 800 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 3 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3,4"


#python3 main_supcon_old.py --epochs 200 --save_freq 10 --batch_size 200 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet50" --datasets "tinyimgnet" --size 256 --trail 5 --temp 0.05 --method "SimCLR" --method_gama 0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "cv2saliency" --augmentation_method "mixup_positive" 


#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3"

#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "4"


#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3,4"


#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "3" --old_augmented true

#python3 main_supcon_old.py --epochs 600 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "4" --old_augmented true


python3 main_supcon_old.py --epochs 150 --save_freq 50 --batch_size 256 --learning_rate 0.0008 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 4 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_layers "2,3" --old_augmented true --last_model_path "./save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3_SimCLR_1.0_1.0_0.05_trail_4_128_256_old_augmented/ckpt_epoch_450.pth"


