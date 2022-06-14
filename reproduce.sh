#!/bin/bash

set -e

DATASETS=(
    # HDM05 normalized
    data/hdm05/class130-actions-segment120_shift16-coords_normPOS-fps12.data
    data/hdm05/class130-actions-segment80_shift16-coords_normPOS-fps12.data
    data/hdm05/class130-actions-segment40_shift20-coords_normPOS-fps12.data

    # HDM05 non-normalized
    data/hdm05/class130-actions_segment80_shift16-coords-fps12.data

    # PKU-MMD normalized
    data/pku-mmd/actions_singlesubject-segment20_shift4_initialshift0-coords_normPOS-fps30.data
)

LATENT_DIMS=(8 16 32 64 128 256)

for DATASET in ${DATASETS[@]}; do

    case $DATASET in
        *hdm05*segment40*   ) INPUT_LENGTH=4  ;;
        *hdm05*segment80*   ) INPUT_LENGTH=8  ;;
        *hdm05*segment120*  ) INPUT_LENGTH=12 ;;
        *pku-mmd*segment20* ) INPUT_LENGTH=20 ;;
        *) echo "$DATASET: Unsupported segment lenght, skipping ..."; continue ;;
    esac

    case $DATASET in
        *hdm05*   )
            INPUT_DIM=93
            EPOCHS=250
            TRAIN_SPLIT="data/hdm05/2foldsBal_2-class122.txt"
            VALID_SPLIT="data/hdm05/2foldsBal_1-class122.txt"
             TEST_SPLIT="${VALID_SPLIT}"
        ;;
        *pku-mmd* )
            INPUT_DIM=75
            EPOCHS=100
            TRAIN_SPLIT="data/pku-mmd/CS_train_objects_messif-lines.txt"
            VALID_SPLIT="data/pku-mmd/CS_test_objects_messif-lines.txt"
             TEST_SPLIT="${VALID_SPLIT}"
        ;;
        *) echo "$DATASET: Unsupported splits, skipping ..."; continue ;;
    esac

    for LATENT_DIM in ${LATENT_DIMS[@]}; do
        python train.py \
            $DATASET \
            --train-split $TRAIN_SPLIT \
            --valid-split $VALID_SPLIT \
            --test-split $TEST_SPLIT \
            --epochs $EPOCHS \
            --latent-dim $LATENT_DIM \
            --input-dim $INPUT_DIM \
            --input-length $INPUT_LENGTH
    done
done