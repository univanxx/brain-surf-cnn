#!/bin/bash

source activate brain_surf_cnn

PROJECT_DIR=/media/ssd-3t/isviridov/fmri_generation/brain-surf-cnn
cd $PROJECT_DIR

MICCAI_DIR=$PROJECT_DIR/data/MICCAI2020
DATA_DIR=/mnt/s3-data2/datasets/fmri_data/preprocessed_data
NUM_ICS=50
NUM_SAMPLES=8
FILTER=True

LOSS=multiple_mse

RSFC_DIR=$DATA_DIR/rsfc_d${NUM_ICS}_sample$NUM_SAMPLES

SUBJ_LIST_FILE=/home/isviridov/skoltech_fmri/brain-surf-cnn/data/MICCAI2020/HCP_test_retest_subj_ids.csv

MESH_TEMPLATES_DIR=$PROJECT_DIR/data/fs_LR_mesh_templates

OUTPUTS_DIR=$MICCAI_DIR/sample_outputs_unet_plus_plus/HCP_sample_feat64_s${NUM_SAMPLES}_c${NUM_ICS}_lr0.01_seed28_filter_${FILTER}_${LOSS}/finetuned_feat64_s8_c50_lr0.01_seed28_filter_True_multiple_rc

echo $OUTPUTS_DIR

python -u predict.py \
       --gpus 1 \
       --ver best_corr \
       --subj_list $SUBJ_LIST_FILE \
       --rsfc_dir $RSFC_DIR \
       --mesh_dir $MESH_TEMPLATES_DIR \
       --checkpoint_file $OUTPUTS_DIR/best_corr.pth \
       --save_dir $OUTPUTS_DIR/predict_on_test_subj \
       --use_unet_plus_plus
