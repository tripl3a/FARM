python nohate_lm_finetuning.py \
  --bert_model="bert-base-german-cased" \
  --data_dir="../data/nohate" \
  --output_dir="saved_models/bert-nohate-lm" \
  --num_train_epochs=0.01 \
  --learning_rate=2e-05 \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --eval_every=100 \
  --warmup_proportion=0.1 \
  --mlflow_run_name="LM_finetuning_debug" \
  --max_docs=30 \
  --embeds_dropout_prob=0.1

