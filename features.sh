#python3 feature_reading.py --dataset "cifar-10-100-10" --model "resnet18" --model_path "/save/SupCon/cifar-10-100-10_models/cifar-10-100-10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_2_128_256_twostage/ckpt_epoch_0.pth" --epoch 0 --trail 2 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown" 



#python3 feature_reading_old.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__vanilia__SimCLR_1.0_1.0_0.05_trail_1_128_256/ckpt_epoch_600.pth" --epoch 600 --trail 1 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train"
#python3 feature_reading_old.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__vanilia__SimCLR_1.0_1.0_0.05_trail_1_128_256/ckpt_epoch_600.pth" --epoch 600 --trail 1 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known"
#python3 feature_reading_old.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__vanilia__SimCLR_1.0_1.0_0.05_trail_1_128_256/ckpt_epoch_600.pth" --epoch 600 --trail 1 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown"

#python3 feature_reading_old.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__vanilia__SimCLR_1.0_1.0_0.05_trail_2_128_256/ckpt_epoch_600.pth" --epoch 600 --trail 2 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train"


#python3 feature_reading_old.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.0_0.05_trail_4_128_256_old_augmented/last.pth" --epoch 600 --trail 4 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train"
#python3 feature_reading_old.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.0_0.05_trail_4_128_256_old_augmented/last.pth" --epoch 600 --trail 4 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known"
#python3 feature_reading_old.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.0_0.05_trail_4_128_256_old_augmented/last.pth" --epoch 600 --trail 4 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown"




#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_vanilia__SimCLR_1.0_1.2_0.05_trail_2_128_256/last.pth" --epoch 600 --trail 2 --lr 0.01 --training_bz 256 --if_train "train"
#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_vanilia__SimCLR_1.0_1.2_0.05_trail_2_128_256/last.pth" --epoch 600 --trail 2 --lr 0.01 --training_bz 256 --if_train "test_known"
#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_vanilia__SimCLR_1.0_1.2_0.05_trail_2_128_256/last.pth" --epoch 600 --trail 2 --lr 0.01 --training_bz 256 --if_train "test_unknown"



python3 feature_reading_old.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_5_128_256/last.pth" --epoch 600 --trail 5 --lr 0.01 --training_bz 256 --if_train "train"
python3 feature_reading_old.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_5_128_256/last.pth" --epoch 600 --trail 5 --lr 0.01 --training_bz 256 --if_train "test_known"




#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.2_0.05_trail_2_128_256_twostage_old_augmented/last.pth" --epoch 600 --trail 2 --lr 0.01 --training_bz 256 --if_train "train"
#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.2_0.05_trail_2_128_256_twostage_old_augmented/last.pth" --epoch 600 --trail 2 --lr 0.01 --training_bz 256 --if_train "test_known"
#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.2_0.05_trail_2_128_256_twostage_old_augmented/last.pth" --epoch 600 --trail 2 --lr 0.01 --training_bz 256 --if_train "test_unknown"



#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256/last.pth" --epoch 600 --trail 0 --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train"
#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256/last.pth" --epoch 600 --trail 0 --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known"
#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256/last.pth" --epoch 600 --trail 0 --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown"



#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.2_0.05_trail_1_128_256_twostage_old_augmented/last.pth" --epoch 600 --trail 0 --lr 0.01 --training_bz 256 --if_train "train"
#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.2_0.05_trail_1_128_256_twostage_old_augmented/last.pth" --epoch 600 --trail 0 --lr 0.01 --training_bz 256 --if_train "test_known"
#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.2_0.05_trail_1_128_256_twostage_old_augmented/last.pth" --epoch 600 --trail 0 --lr 0.01 --training_bz 256 --if_train "test_unknown"




#python3 feature_reading_old.py --dataset "cub" --model "resnet18" --model_path "/save/SupCon/cub_models/cub_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3_SimCLR_1.0_1.2_0.05_trail_0_128_256/last.pth" --epoch 600 --trail 1 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train"
#python3 feature_reading_old.py --dataset "cub" --model "resnet18" --model_path "/save/SupCon/cub_models/cub_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3_SimCLR_1.0_1.2_0.05_trail_0_128_256/last.pth" --epoch 600 --trail 1 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known"
#python3 feature_reading_old.py --dataset "cub" --model "resnet18" --model_path "/save/SupCon/cub_models/cub_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3_SimCLR_1.0_1.2_0.05_trail_0_128_256/last.pth" --epoch 600 --trail 1 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown"



#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_0_128_256/last.pth" --epoch 600 --trail 0 --lr 0.01 --training_bz 256 --if_train "train"
#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_0_128_256/last.pth" --epoch 600 --trail 0 --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known"
#python3 feature_reading_old.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_0_128_256/last.pth" --epoch 600 --trail 0 --lr 0.01 --training_bz 256 --if_train "test_unknown"




