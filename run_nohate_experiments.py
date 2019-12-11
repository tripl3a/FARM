import argparse
import datetime
from farm.experiment import run_experiment, load_experiments


def get_current_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    #config_files = [
        #"experiments/text_classification/noHateCoarse_config08.json",
        #"experiments/text_classification/noHateCoarse_config09.json",
        #"experiments/text_classification/germEval18Coarse_config_grid.json",
        #"experiments/text_classification/germEval18Coarse_config_best.json",
        #"experiments/text_classification/noHateCoarse_config10_grid.json",
        #"experiments/text_classification/germEval18Coarse_config_merged_grid.json",
        #"experiments/text_classification/germEval18Coarse_config_merged_best.json",
        #"experiments/text_classification/noHateCoarse_config11_grid.json",
        #"experiments/text_classification/noHateCoarse_config12_grid.json",
        #"experiments/text_classification/noHateCoarse_config13.json",
        #"experiments/text_classification/noHateCoarse_config14_lm1.json",
        #"experiments/text_classification/noHateCoarse_config14_lm2.json",
        #"experiments/text_classification/noHateCoarse_config14_lm3.json",
        #experiments/text_classification/noHateCoarse_config14_lm4.json",
        #experiments/text_classification/noHateCoarse_config14_lm5.json",
    #]

    for conf_file in args.config_files:
        experiments = load_experiments(conf_file)
        for experiment in experiments:
            run_experiment(experiment)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-cf', '--config_files', action='append', help='<Required> Config files to run', required=True)

    args = parser.parse_args()

    print("Arguments: " + str(args))
    print("Start datetime:", get_current_datetime())
    main()
    print("End datetime:", get_current_datetime())

