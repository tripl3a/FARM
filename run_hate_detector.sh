#!/usr/bin/env bash

if [ "$HOSTNAME" = "tripl3a-t440s" ]; then
  export RUN_PY_SCRIPT=hate_detector/nohate_clf_train.py
  #export RUN_PY_SCRIPT=hate_detector/run_nohate_experiments.py
  export CACHE_DIR=/tlhd/cache
  export DATA_DIR=/tlhd/data/modeling/FU_data_full
  export OUTPUT_DIR=/tlhd/models/nohate01
fi

if [ "$RUN_PY_SCRIPT" = hate_detector/nohate_clf_train.py ]; then
  echo "*********************************"
  echo "Running Python script: " $RUN_PY_SCRIPT
  echo "with args..." $RUN_PY_SCRIPT
  echo "--> CACHE_DIR=" $CACHE_DIR
  echo "--> DATA_DIR=" $DATA_DIR
  echo "--> OUTPUT_DIR=" $OUTPUT_DIR
  echo "*********************************"
  python $RUN_PY_SCRIPT  \
    --cache_dir $CACHE_DIR \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR
else
  echo "Running Python script: " $RUN_PY_SCRIPT
  python $RUN_PY_SCRIPT
fi



