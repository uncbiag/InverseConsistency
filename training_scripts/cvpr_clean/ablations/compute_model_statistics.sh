for operand in vm_unet icon_unet vm lapirn_disp lapirn_diff icon gradicon_stage1 gradicon; do
    python ./training_scripts/cvpr_clean/ablations/compute_model_statistics.py --model_name "$operand"
done > ./training_scripts/cvpr_clean/ablations/model_statistics_log.txt