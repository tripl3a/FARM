#!/usr/bin/env bash

#if [ "$HOSTNAME" = "tripl3a-t440s" ]; then
#    export WORK_DIR=/tlhd/models/bert02
#    export MODEL_DIR=/tlhd/models/bert02/output
#    export CODE_DIR=~/git-reps/pytorch-transformers
#else
#    export WORK_DIR=/home/aallhorn
#    export MODEL_DIR=/home/aallhorn/output
#    export CODE_DIR=/home/aallhorn/code/pytorch-transformers
#fi

#python hate_detector/run_experiments.py

python hate_detector/nohate_classifier.py