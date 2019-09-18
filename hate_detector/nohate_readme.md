
## Saving & loading models 

### In FARM

The output of a saved (adaptive) model, with only a `TextClassificationHead`, looks like this on disk:

```bash
-rw-r--r-- 1 root root 436373175 Sep 18 13:01 language_model.bin
-rw-r--r-- 1 root root       515 Sep 18 13:01 language_model_config.json
-rw-r--r-- 1 root root      7146 Sep 18 13:01 prediction_head_0.bin
-rw-r--r-- 1 root root       259 Sep 18 13:01 prediction_head_0_config.json
-rw-r--r-- 1 root root       717 Sep 18 13:00 processor_config.json
-rw-r--r-- 1 root root    254365 Sep 18 13:00 vocab.txt
```

Debugging showed that specifying `model=<saved_model_dir>` when running experiments leads to loading the following files from disk: 

* `language_model.bin`
* `language_model_config.json`
* `vocab.txt`

While these output files are being ignored:

* `prediction_head_0.bin`
* `prediction_head_0_config.json`
* `processor_config.json`

TODO: Could it be a problem, that new class weights are being calculated in `experiment.get_adaptive_model()`?

### Loading a LM that was fine-tuned directly with `pytorch-transformers`

It should be possible, as only `language_model.bin`, `language_model_config.json` and `vocab.txt` should be needed. 
So the improved lm-finetuning method could be used.

## LM fine-tuning

Number of optimization steps:

```
nohate_lm_finetutning.py:
    n_batches=len(data_silo.loaders["train"]),

optimization.py:
    optimization_steps = int(n_batches / grad_acc_steps) * n_epochs
```
