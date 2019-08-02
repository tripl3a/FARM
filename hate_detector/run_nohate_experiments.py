from farm.experiment import run_experiment, load_experiments


def main():
    config_files = [
        "../experiments/text_classification/noHateCoarse_config.json"
    ]

    for conf_file in config_files:
        experiments = load_experiments(conf_file)
        for experiment in experiments:
            run_experiment(experiment)


if __name__ == "__main__":
    main()
