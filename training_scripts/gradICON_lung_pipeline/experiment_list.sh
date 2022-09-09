output_root=""
dataset_root=""

# Comparison between LNCC, AdaptiveNCC and NCC
python halfres_train_lung.py --exp="GradICON_noaug_LNCC" --output_root="$output_root" --dataset_root="$dataset_root" --reg="GradientICON" --augmentation=0 --sim="LNCC" --lamda=1
python halfres_train_lung.py --exp="GradICON_noaug_adaNCC" --output_root="$output_root" --dataset_root="$dataset_root" --reg="GradientICON" --augmentation=0 --sim="AdaptiveNCC" --lamda=1
python halfres_train_lung.py --exp="GradICON_noaug_NCC" --output_root="$output_root" --dataset_root="$dataset_root" --reg="GradientICON" --augmentation=0 --sim="NCC" --lamda=1

# Comparison between augmentation vs no augmentation
# python halfres_train_lung.py --exp="gradICON_aug_lncc" --output_root="$output_root" --dataset_root="$dataset_root" --reg="GradientICON" --augmentation=1 --sim="LNCC" --lamda=1
# python halfres_train_lung.py --exp="gradICON_aug_adaNCC" --output_root="$output_root" --dataset_root="$dataset_root" --reg="GradientICON" --augmentation=1 --sim="AdaptiveNCC" --lamda=1

# Comparison between ICON and gradICON
# python halfres_train_lung.py --exp="ICON_aug_LNCC" --output_root="$output_root" --dataset_root="$dataset_root" --reg="InverseConsistentNet" --augmentation=1 --sim="LNCC" --lamda=700