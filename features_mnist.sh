#python3 feature_reading.py --dataset "mnist" --model "resnet18" --model_path "/save/SupCon/mnist_models/mnist_resnet18_original_data__vanilia__SimCLR_1.0_3.0_0.05_trail_0_128_256/ckpt_epoch_7.pth" --epoch 7 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train"
#python3 feature_reading.py --dataset "mnist" --model "resnet18" --model_path "/save/SupCon/mnist_models/mnist_resnet18_original_data__vanilia__SimCLR_1.0_3.0_0.05_trail_0_128_256/ckpt_epoch_7.pth" --epoch 7 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known" 
#python3 feature_reading.py --dataset "mnist" --model "resnet18" --model_path "/save/SupCon/mnist_models/mnist_resnet18_original_data__vanilia__SimCLR_1.0_3.0_0.05_trail_0_128_256/ckpt_epoch_7.pth" --epoch 7 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown" 



python3 main_testing.py --datasets "tinyimgnet" --ensembles 1 --K 32 --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_0_128_256_twostage/ckpt_epoch_0.pth" --exemplar_features_path "/features/mnist_resnet18_original_data__vanilia__SimCLR_1.0_3.0_0.05_trail_0_128_256_15_train" --testing_known_features_path "/features/mnist_resnet18_original_data__vanilia__SimCLR_1.0_3.0_0.05_trail_0_128_256_15_test_known" --testing_unknown_features_path "/features/mnist_resnet18_original_data__vanilia__SimCLR_1.0_3.0_0.05_trail_0_128_256_15_test_unknown" --num_classes 6 --trail 0 --auroc_save_path "/plots/cifar10_resnet18_temp_0.005_id_4_lr_0.001_bz_256_auroc.pdf" --downsampling_ratio_unknown 30 --downsampling_ratio_known 30


python3 main_testing.py --datasets "tinyimgnet" --ensembles 1 --K 35 --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_0_128_256_twostage/ckpt_epoch_0.pth" --exemplar_features_path "/features/mnist_resnet18_original_data__vanilia__SimCLR_1.0_3.0_0.05_trail_0_128_256_15_train" --testing_known_features_path "/features/mnist_resnet18_original_data__vanilia__SimCLR_1.0_3.0_0.05_trail_0_128_256_15_test_known" --testing_unknown_features_path "/features/mnist_resnet18_original_data__vanilia__SimCLR_1.0_3.0_0.05_trail_0_128_256_15_test_unknown" --num_classes 6 --trail 0 --auroc_save_path "/plots/cifar10_resnet18_temp_0.005_id_4_lr_0.001_bz_256_auroc.pdf" --downsampling_ratio_unknown 30 --downsampling_ratio_known 30





