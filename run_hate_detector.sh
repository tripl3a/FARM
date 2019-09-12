#!/usr/bin/env bash

if [ "$HOSTNAME" = "tripl3a-t440s" ]; then
  #export RUN_PY_SCRIPT=hate_detector/nohate_clf_train.py
  export RUN_PY_SCRIPT=run_nohate_experiments.py
  export CACHE_DIR=/tlhd/cache
  export DATA_DIR=/tlhd/data/modeling/FU_data_subsample
  export OUTPUT_DIR=/tlhd/models/nohate01
fi

case "$RUN_PY_SCRIPT" in
  hate_detector/nohate_clf_train.py)
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
    ;;
  hate_detector/nohate_lm_finetuning.py)
    python $RUN_PY_SCRIPT \
      --bert_model=$BERT_MODEL \
      --data_dir=$DATA_DIR \
      --output_dir=$OUTPUT_DIR \
      --num_train_epochs=$NUM_TRAIN_EPOCHS \
      --learning_rate=$LEARNING_RATE \
      --max_seq_length=$MAX_SEQ_LENGTH \
      --train_batch_size=$TRAIN_BATCH_SIZE \
      --eval_every=$EVAL_EVERY \
      --warmup_proportion=$WARMUP_PROPORTION \
      --mlflow_run_name=$MLFLOW_RUN_NAME \
      --embeds_dropout_prob=$EMBEDS_DROPOUT_PROB
      ;;
  *)
    echo "Running Python script: " $RUN_PY_SCRIPT
    python $RUN_PY_SCRIPT
    ;;
esac



