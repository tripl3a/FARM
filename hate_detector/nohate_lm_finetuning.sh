python nohate_lm_finetuning.py \
  --bert_model="bert-base-german-cased" \
  --data_dir="../data/nohate" \
  --output_dir="saved_models/bert-nohate-lm" \
  --num_train_epochs=0.01 \
  --train_batch_size=8 \
  --eval_every=100 \
  --mlflow_run_name="LM_finetuning_debug"

