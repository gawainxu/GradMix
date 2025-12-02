#python3 feature_reading_masked.py --dataset "imaget100_masked" --model "resnet18" --model_path "/save/SupCon/imagenet100_models/imagenet100_m_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_0_128_256/ckpt_epoch_50.pth"


#python3 feature_reading_masked.py --dataset "imaget100_masked" --model "resnet18" --model_path "/save/SupCon/imagenet100_models/imagenet100_m_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256/ckpt_epoch_50.pth"


#python3 feature_reading_masked.py --dataset "imaget100_masked" --model "resnet18" --model_path "/save/SupCon/imagenet100_models/imagenet100_m_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.0_0.05_trail_0_128_230_old_augmented/ckpt_epoch_50.pth"






#python3 feature_specturm.py --feature_path "./features/imagenet100_m_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_0_128_256_imaget100_masked_1"
#python3 feature_specturm.py --feature_path "./features/imagenet100_m_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256_imaget100_masked_1"
#python3 feature_specturm.py --feature_path "./features/imagenet100_m_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.0_0.05_trail_0_128_230_old_augmented_imaget100_masked_1"


python3 bg_correlation.py --bsz 1 --data_root "../datasets" --num_classes 100 --backbone_model_direct "./save/SupCon/imagenet100_models/imagenet100_m_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_0_128_256/" --backbone_model_name "ckpt_epoch_50.pth" --linear_model_name "ckpt_epoch_50_linear.pth" --output_path "./features/energy_entropy" --threshold 0.001 --method "supcon" --if_train true

python3 bg_correlation.py --bsz 1 --data_root "../datasets" --num_classes 100 --backbone_model_direct "./save/SupCon/imagenet100_models/imagenet100_m_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256/" --backbone_model_name "ckpt_epoch_50.pth" --linear_model_name "ckpt_epoch_50_linear.pth" --output_path "./features/energy_entropy" --threshold 0.001 --method "ssl" --if_train true

python3 bg_correlation.py --bsz 1 --data_root "../datasets" --num_classes 100 --backbone_model_direct "./save/SupCon/imagenet100_models/imagenet100_m_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.0_0.05_trail_0_128_230_old_augmented/" --backbone_model_name "ckpt_epoch_50.pth" --linear_model_name "ckpt_epoch_50_linear.pth" --output_path "./features/energy_entropy" --threshold 0.001 --method "grad" --if_train true
