from farm.experiment import run_experiment, load_experiments


def main():
    config_files = [
        #"experiments/text_classification/noHateCoarse_config08.json",
        #"experiments/text_classification/noHateCoarse_config09.json",
        #"experiments/text_classification/germEval18Coarse_config_grid.json",
        #"experiments/text_classification/germEval18Coarse_config_best.json",
        #"experiments/text_classification/noHateCoarse_config_grid10.json",
        #"experiments/text_classification/germEval18Coarse_config_merged_grid.json",
        #"experiments/text_classification/germEval18Coarse_config_merged_best.json",
        "experiments/text_classification/noHateCoarse_config_grid11.json",
    ]

    for conf_file in config_files:
        experiments = load_experiments(conf_file)
        for experiment in experiments:
            run_experiment(experiment)


if __name__ == "__main__":
    main()
