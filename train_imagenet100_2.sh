#python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 4 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --mixup_positive True --positive_method "cutmix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0

python3 main_supcon_old.py --epochs 100 --save_freq 10 --batch_size 16 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100_m" --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 0.0 --mixup_positive False


#python3 main_supcon_split.py --epochs 200 --save_freq 1 --batch_size 256 --learning_rate 0.0008 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --augmentation_method "mixup_positive" --mixup_positive true --positive_method "random" --grad_splits 32 --last_model_path "./save/SupCon/imagenet100_models/imagenet100_resnet18_original_data__mixup_positive_random_intra_True_alpha_0.2_p_0.5_SimCLR_1.0_trail_0_128_256_split_32/ckpt_epoch_81.pth"


#python3 main_supcon_old.py --epochs 200 --save_freq 10 --batch_size 64 --moco_step 4 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "imagenet100" --size 256 --trail 0 --temp 0.05 --method "MoCo" --method_gama 0.0 --method_lam 1.0 --mixup_positive True --positive_method "cutmix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0
