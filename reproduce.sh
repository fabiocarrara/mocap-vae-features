#!/bin/bash


DATASETS=(
    # data/class130-actions-segment40_shift20-coords_normPOS-fps12.data
    data/class130-actions-segment80_shift16-coords_normPOS-fps12.data
    # data/class130-actions-segment120_shift16-coords_normPOS-fps12.data
)

LATENT_DIMS=(8 16 32 64 128 256)

for DATASET in ${DATASETS[@]}; do
    for LATENT_DIM in ${LATENT_DIMS[@]}; do
        python train.py $DATASET -d $LATENT_DIM
    done
done