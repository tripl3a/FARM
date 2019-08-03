import itertools
import pandas as pd

batch_size = [16, 32]
learning_rate = [5e-5, 3e-5, 2e-5]
epochs = [3, 4]

dev_size = [0.1, 0.2, 0.3]

grid = list(itertools.product(batch_size, learning_rate, epochs, dev_size))

df = pd.DataFrame(grid, columns=["batch_size", "learning_rate", "epochs", "dev_size"])
print(df)


# BERT also seems to be working on smaller datasets. You could easily test how well BERT performs with FARM by setting
# the "dev_size" parameter in one of the Germeval 2018 configs (in folder experiments/text_classification) to be a list
# of values, e.g. [0.5,0.7,0.9,0.95]. That way you will create small training sets which are evaluated on remaining
# large development sets. FARM automativally iterates over these values and logs results to our public MLflow server.
# Use the script run_all_experiments.py and comment out all other configs than the one you modified.
# https://github.com/deepset-ai/FARM/issues/17#issuecomment-516347171