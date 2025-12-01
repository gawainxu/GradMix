#python3 main_supcon.py --epochs 200 --save_freq 40 --batch_size 256 --learning_rate 0.03 --lr_decay_epochs "120, 160" --lr_decay_rate 0.1 --model "resnet34" --datasets "tinyimgnet" --size 256 --trail 5 --temp 0.05 --method "MoCo" --K 8192 --momentum_moco 0.999 --method_gama 0.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0


#23356849
python3 main_supcon.py --epochs 200 --save_freq 50 --batch_size 64 --learning_rate 0.03 --lr_decay_epochs "120, 160" --lr_decay_rate 0.1 --model "resnet50" --datasets "tinyimgnet" --size 256 --trail 5 --temp 0.05 --method "MoCo" --K 8192 --momentum_moco 0.999 --method_gama 0.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --grad_accumulationstep 4 --grad_layers "3"
