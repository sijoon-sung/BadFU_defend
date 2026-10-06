#!/bin/bash
set -e

if [ -d ./record ]; then
    echo "[INFO] Removing contents of ./record (root)..."
    rm -rf ./record/*
else
    echo "[INFO] ./record (root) does not exist. Creating it now..."
    mkdir -p ./record
fi

cd data_prepare

if [ -d ./record ]; then
    echo "[INFO] Removing contents of ./record (root)..."
    rm -rf ./record/*
else
    echo "[INFO] ./record (root) does not exist. Creating it now..."
    mkdir -p ./record
fi

python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name badnet_dataset --add_cover 1 --epoch 00 --pratio 0.017 --cratio 0.017 --attack_target 0

python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset --device cuda:0 --ft_epoch 1 --ap_epochs 1

rm -rf ./record/badnet_dataset/cv_train_dataset/pert/*

mv ./record/badnet_dataset/bd_train_dataset/[1-9] ./record/badnet_dataset/cv_train_dataset/pert/

cp -r record ../

cd ../