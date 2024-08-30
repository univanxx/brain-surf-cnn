#!/bin/bash

source activate brain_surf_cnn

PROJECT_DIR=/media/ssd-3t/isviridov/fmri_generation/brain-surf-cnn
cd $PROJECT_DIR

NUM_ICS=50
NUM_SAMPLES=8
NUM_VAL_SUBJ=5
FILTER=True
LOSS=multiple_mse

MICCAI_DIR=$PROJECT_DIR/data/MICCAI2020
DATA_DIR=/mnt/s3-data2/datasets/fmri_data/preprocessed_data
RSFC_DIR=$DATA_DIR/rsfc_d${NUM_ICS}_sample$NUM_SAMPLES
CONTRASTS_DIR=$DATA_DIR/contrasts

SUBJ_LIST_FILE=$MICCAI_DIR/HCP_train_val_subj_ids.csv

MESH_TEMPLATES_DIR=$PROJECT_DIR/data/fs_LR_mesh_templates

OUTPUTS_DIR=$MICCAI_DIR/sample_outputs_unet_plus_plus

python -u train.py \
       --gpus 0 \
       --ver finetuned \
       --loss multiple_rc \
       --checkpoint_file $OUTPUTS_DIR/HCP_sample_feat64_s${NUM_SAMPLES}_c${NUM_ICS}_lr0.01_seed28_filter_${FILTER}_${LOSS}/best_corr.pth \
       --subj_list $SUBJ_LIST_FILE \
       --rsfc_dir $RSFC_DIR \
       --contrast_dir $CONTRASTS_DIR \
       --mesh_dir $MESH_TEMPLATES_DIR \
       --save_dir $OUTPUTS_DIR/HCP_sample_feat64_s${NUM_SAMPLES}_c${NUM_ICS}_lr0.01_seed28_filter_${FILTER}_${LOSS} \
       --n_val_subj $NUM_VAL_SUBJ \
       --use_dataset_filtering \
       --use_unet_plus_plus \
       --init_within_subj_margin 4.964731128223573 \
       --init_across_subj_margin 5.318829739418066
