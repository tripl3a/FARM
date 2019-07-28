#!/usr/bin/env bash

#export RUN_PY_SCRIPT=hate_detector/nohate_classifier.py
#export RUN_PY_SCRIPT=hate_detector/run_experiments.py

#if [ "$HOSTNAME" = "tripl3a-t440s" ]; then
#  export CACHE_DIR=/tlhd/cache
#  export DATA_DIR=/tlhd/data/modeling/FU_data_full
#  export OUTPUT_DIR=/tlhd/models/nohate01
#else
#  export CACHE_DIR=/home/aallhorn/cache
#  export DATA_DIR =/home/aallhorn/data/FU_data_full
#  export OUTPUT_DIR=/home/aallhorn/output/nohate01
#fi

echo "Running Python script: " $RUN_PY_SCRIPT
python $RUN_PY_SCRIPT  \
  --cache_dir $CACHE_DIR \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR
