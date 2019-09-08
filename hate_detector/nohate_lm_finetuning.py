import logging
import torch
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import BertStyleLMProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import BertLMHead, NextSentenceHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.modeling.optimization import initialize_optimizer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
import argparse
from utils import tools


def main(args):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_all_seeds(seed=args.seed)
    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(
        experiment_name=args.experiment_name, run_name=args.run_name
    )

    device, n_gpu = initialize_device_settings(use_cuda=(not args.no_cuda))

    # 1.Create a tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.bert_model, do_lower_case=args.do_lower_case
    )

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    processor = BertStyleLMProcessor(
        data_dir=args.data_dir, tokenizer=tokenizer, max_seq_len=args.max_seq_len, max_docs=30
    )
    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(processor=processor, batch_size=args.train_batch_size)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = Bert.load(args.bert_model)
    # b) and *two* prediction heads on top that are suited for our task => Language Model finetuning
    lm_prediction_head = BertLMHead.load(args.bert_model)
    next_sentence_head = NextSentenceHead.load(args.bert_model)

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[lm_prediction_head, next_sentence_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token", "per_sequence"],
        device=device,
    )

    # 5. Create an optimizer
    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        warmup_proportion=args.warmup_proportion,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=args.num_train_epochs,
    )

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=args.num_train_epochs,
        n_gpu=n_gpu,
        warmup_linear=warmup_linear,
        evaluate_every=args.eval_every,
        device=device,
    )

    # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
    model = trainer.train(model)

    # 8. Hooray! You have a model. Store it:
    model.save(args.output_dir)
    processor.save(args.output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Require parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Folder from which the data is read, i.e. the location of the training corpus.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese, etc.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    # Optional FARM parameters
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")

    # Logging
    parser.add_argument("--eval_every",
                        default=1000,
                        type=int,
                        help="Steps per training loop (batches) required for evaluation on dev set. \n"
                             "Set to 0 when you do not want to do evaluation on dev set during training.")
    parser.add_argument("--mlflow_experiment", default="NOHATE", type=str,
                        help="Experiment name used for MLflow.")
    parser.add_argument("--mlflow_run_name", default="LM_finetuning_run", type=str,
                        help="Name of the particular run for MLflow")

    ## Other parameters, as in Huggingface's `simple_lm_finetuning`
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    # parser.add_argument("--do_train",
    #                     action='store_true',
    #                     help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--adam_epsilon",
    #                     default=1e-8,
    #                     type=float,
    #                     help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_cuda",
                        action='store_true', default=False,
                        help="Whether not to use CUDA when available")
    # parser.add_argument("--on_memory",
    #                     action='store_true',
    #                     help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        action='store_true', default=False,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    print("Arguments: " + str(args))
    print("Start datetime:", tools.get_current_datetime())
    main(args)
    print("End datetime:", tools.get_current_datetime())
