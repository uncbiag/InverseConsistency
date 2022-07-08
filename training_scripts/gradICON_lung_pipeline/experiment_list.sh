python halfres_train_lung.py --exp="gradICON_aug_lncc" --output_root="results/" --dataset_root="/playpen-ssd/tgreer/ICON_lung/results/half_res_preprocessed_transposed_SI/" --reg="GradientICON" --augmentation=1 --sim="LNCC" --lambda=1
python halfres_train_lung.py --exp="gradICON_aug_adaNCC" --output_root="results/" --dataset_root="/playpen-ssd/tgreer/ICON_lung/results/half_res_preprocessed_transposed_SI/" --reg="GradientICON" --augmentation=1 --sim="AdaptiveNCC" --lambda=0.1
python halfres_train_lung.py --exp="ICON_aug_LNCC" --output_root="results/" --dataset_root="/playpen-ssd/tgreer/ICON_lung/results/half_res_preprocessed_transposed_SI/" --reg="InverseConsistentNet" --augmentation=1 --sim="LNCC" --lambda=700
python halfres_train_lung.py --exp="ICON_aug_adaNCC" --output_root="results/" --dataset_root="/playpen-ssd/tgreer/ICON_lung/results/half_res_preprocessed_transposed_SI/" --reg="InverseConsistentNet" --augmentation=1 --sim="AdaptiveNCC" --lambda=700

python halfres_train_lung.py --exp="GradICON_aug_NCC" --output_root="results/" --dataset_root="/playpen-ssd/tgreer/ICON_lung/results/half_res_preprocessed_transposed_SI/" --reg="GradientICON" --augmentation=1 --sim="NCC" --lambda=1


python halfres_train_lung.py --exp="GradICON_noaug_LNCC" --output_root="results/" --dataset_root="/playpen-ssd/tgreer/ICON_lung/results/half_res_preprocessed_transposed_SI/" --reg="GradientICON" --augmentation=0 --sim="LNCC" --lambda=1
python halfres_train_lung.py --exp="GradICON_noaug_adaNCC" --output_root="results/" --dataset_root="/playpen-ssd/tgreer/ICON_lung/results/half_res_preprocessed_transposed_SI/" --reg="GradientICON" --augmentation=0 --sim="AdaptiveNCC" --lambda=.1


