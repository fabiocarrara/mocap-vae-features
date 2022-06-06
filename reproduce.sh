#!/bin/bash

set -e

DATASETS=(
    data/class130-actions-segment120_shift16-coords_normPOS-fps12.data
    data/class130-actions-segment80_shift16-coords_normPOS-fps12.data
    data/class130-actions-segment40_shift20-coords_normPOS-fps12.data

    data/class130-actions_segment80_shift16-coords-fps12.data
)

LATENT_DIMS=(8 16 32 64 128 256)

for DATASET in ${DATASETS[@]}; do

    case $DATASET in
        *segment40* ) INPUT_LENGTH=4  ;;
        *segment80* ) INPUT_LENGTH=8  ;;
        *segment120*) INPUT_LENGTH=12 ;;
        *) echo "$DATASET: Unsupported segment lenght, skipping ..."; continue ;;
    esac

    for LATENT_DIM in ${LATENT_DIMS[@]}; do
        python train.py \
            $DATASET \
            --train-split data/2foldsBal_2-class122.txt \
            --valid-split data/2foldsBal_1-class122.txt \
            --test-split data/2foldsBal_1-class122.txt \
            --epochs 250 \
            --latent-dim $LATENT_DIM \
            --input-length $INPUT_LENGTH
    done
done