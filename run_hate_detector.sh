#!/usr/bin/env bash

if [ "$HOSTNAME" = "tripl3a-t440s" ]; then
  #export RUN_PY_SCRIPT=hate_detector/nohate_clf_train.py
  export RUN_PY_SCRIPT=run_all_experiments.py
  #export CACHE_DIR=/tlhd/cache
  #export DATA_DIR=/tlhd/data/modeling/FU_data_subsample
  #export OUTPUT_DIR=/tlhd/models/nohate01
else
  nvidia-smi
fi

echo "Running Python script: " $RUN_PY_SCRIPT
echo "with parameters: " $PARAMETERS

if [ "$DISTRIBUTED" = "true" ]; then
  echo "Launching distributed training..."
  python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE $RUN_PY_SCRIPT \
  $PARAMETERS
else
  python $RUN_PY_SCRIPT \
  $PARAMETERS
fi



