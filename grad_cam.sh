#python3 gradcam.py --datasets "tinyimgnet" --num_classes 200 --trail 5 --img_size 64 --bsz 256 --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_5_128_256_old_augmented/last.pth" --mode "ssl" --use_cuda true


#python3 gradcam.py --datasets "imagenet100" --num_classes 100 --trail 0 --img_size 224 --bsz 256 --model_path "/save/SupCon/imagenet100_m_models/imagenet100_m_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256/last.pth" --mode "ssl" 

python3 gradcam.py --datasets "imagenet100" --num_classes 100 --trail 0 --img_size 224 --bsz 256 --model_path "/save/SupCon/imagenet100_m_models/imagenet100_m_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256/last.pth" --mode "supcon" 


python3 gradcam.py --datasets "imagenet100" --num_classes 100 --trail 0 --img_size 224 --bsz 256 --model_path "/save/SupCon/imagenet100_models/imagenet100_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3,4_SimCLR_0.0_1.0_0.05_trail_0_128_256_old_augmented/ckpt_epoch_100.pth" --mode "ssl" 

python3 gradcam.py --datasets "imagenet100" --num_classes 100 --trail 0 --img_size 224 --bsz 256 --model_path "/save/SupCon/imagenet100_models/imagenet100_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3,4_SimCLR_0.0_1.0_0.05_trail_0_128_256_old_augmented/ckpt_epoch_100.pth" --mode "supcon" 
