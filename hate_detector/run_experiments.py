from farm.experiment import run_experiment, load_experiments
experiments = load_experiments("experiments/text_classification/germEval18Coarse_config.json")
run_experiment(experiments[0])