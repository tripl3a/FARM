# fmt: off
import argparse
import logging

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings
from farm.utils import MLFlowLogger as MlLogger

logger = logging.getLogger(__name__)

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)


def main():

    set_all_seeds(seed=args.seed)
    device, n_gpu = initialize_device_settings(use_cuda=(not args.no_cuda))

    # 1.Create a tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.bert_model,
        do_lower_case=False)

    # train on GermEval 2018 data
    germeval_model, germeval_processor = train_germeval(device, n_gpu, tokenizer)

    # Hooray! You have a model. Store it:
    germeval_save_dir=args.output_dir + "/germeval"
    germeval_model.save(germeval_save_dir)
    germeval_processor.save(germeval_save_dir)

    nohate_model, nohate_processor = further_train_nohate(device, n_gpu, tokenizer, germeval_save_dir)
    nohate_save_dir = args.output_dir + "/nohate"
    nohate_model.save(nohate_save_dir)
    nohate_processor.save(nohate_save_dir)

    #infer()


def further_train_nohate(device, n_gpu, tokenizer, prev_model_dir):

    ml_logger = MlLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name=args.mlflow_experiment,
                              run_name=args.mlflow_run_name,
                              nested=True)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # Here we load NOHATE Data.

    label_list = ["nohate", "hate"]
    metric = "f1_macro"

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=args.max_seq_length,
                                            data_dir=args.nohate_data_dir,
                                            labels=label_list,
                                            metric=metric,
                                            source_field="label",
                                            train_filename="coarse_test.tsv",
                                            dev_filename="coarse_dev.tsv",
                                            test_filename="coarse_test.tsv"
                                            )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=args.train_batch_size)

    # 4. Create an AdaptiveModel
    # # a) which consists of a pretrained language model as a basis
    # language_model = Bert.load(args.bert_model)
    # # b) and a prediction head on top that is suited for our task => Text classification
    # prediction_head = TextClassificationHead(
    #     layer_dims=[768, len(processor.tasks["text_classification"]["label_list"])])
    #
    # model = AdaptiveModel(
    #     language_model=language_model,
    #     prediction_heads=[prediction_head],
    #     embeds_dropout_prob=0.1,
    #     lm_output_types=["per_sequence"],
    #     device=device)

    model = AdaptiveModel.load(prev_model_dir, device)

    # 5. Create an optimizer
    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        warmup_proportion=0.1,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=args.num_train_epochs)

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=args.num_train_epochs,
        n_gpu=n_gpu,
        warmup_linear=warmup_linear,
        evaluate_every=args.eval_every,
        device=device)

    # 7. Let it grow
    model = trainer.train(model)

    return model, processor


def train_germeval(device, n_gpu, tokenizer):

    ml_logger = MlLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name=args.mlflow_experiment,
                              run_name=args.mlflow_run_name,
                              nested=True)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # Here we load GermEval 2018 Data.

    label_list = ["OTHER", "OFFENSE"]
    metric = "f1_macro"

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=args.max_seq_length,
                                            data_dir=args.germeval_data_dir,
                                            labels=label_list,
                                            metric=metric,
                                            source_field="coarse_label"
                                            )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=args.train_batch_size)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = Bert.load(args.bert_model)
    # b) and a prediction head on top that is suited for our task => Text classification
    prediction_head = TextClassificationHead(layer_dims=[768, len(processor.tasks["text_classification"]["label_list"])])

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    # 5. Create an optimizer
    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        warmup_proportion=0.1,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=args.num_train_epochs)

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=args.num_train_epochs,
        n_gpu=n_gpu,
        warmup_linear=warmup_linear,
        evaluate_every=args.eval_every,
        device=device)

    # 7. Let it grow
    model = trainer.train(model)

    return model, processor


def infer():
    # 9. Load it & harvest your fruits (Inference)
    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
        {"text": "Martin Müller spielt Handball in Berlin"},
    ]
    model = Inferencer.load(args.output_dir)
    result = model.run_inference(dicts=basic_texts)
    print(result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--germeval_data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--nohate_data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-german-cased, bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese, etc.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=150,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--num_train_epochs",
                        default=4,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--no_cuda",
                        action='store_true', default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # Logging
    parser.add_argument("--eval_every",
                        default=30,
                        type=int,
                        help="Steps per training loop (batches) required for evaluation on dev set. \n"
                             "Set to 0 when you do not want to do evaluation on dev set during training.")
    parser.add_argument("--mlflow_experiment", default="NOHATE", type=str,
                        help="Experiment name used for MLflow.")
    parser.add_argument("--mlflow_run_name", default="GermEval->NoHate clf fine-tuning", type=str,
                        help="Name of the particular run for MLflow")

    args = parser.parse_args()

    main()

# fmt: on
