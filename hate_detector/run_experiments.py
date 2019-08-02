import logging
from farm.experiment import run_experiment, load_experiments
from farm.utils import MLFlowLogger

logger = logging.getLogger(__name__)


def main():

    experiments = load_experiments("experiments/text_classification/germEval18Coarse_config.json")

    for args in experiments:
        logger.info(
            "\n***********************************************"
            f"\n************* Experiment: {args.name} ************"
            "\n************************************************"
        )
        ml_logger = MLFlowLogger(tracking_uri=args.mlflow_url)
        ml_logger.init_experiment(
            experiment_name=args.mlflow_experiment,
            run_name=args.mlflow_run_name,
            nested=args.mlflow_nested,
        )
        run_experiment(args)


if __name__ == "__main__":
    main()
