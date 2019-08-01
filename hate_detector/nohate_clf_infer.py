# coding=utf-8

from farm.infer import Inferencer
import pandas as pd
import os

DATA_DIR = "/tlhd/data/modeling/FU_data_subsample"
OUTPUT_DIR = "/tlhd/models/nohate01"

# load test data
test_file = os.path.join(DATA_DIR, "coarse_test.tsv")
df_test = pd.read_csv(filepath_or_buffer=test_file, delimiter="\t")

# build list of dicts for FARM
texts = []
for index, row in df_test.iterrows():
    texts.append({"text": row["text"],
                  "true_label": row["label"]})

# Load saved model and make predictions
model = Inferencer.load(OUTPUT_DIR)
result = model.run_inference(dicts=texts)

y_true, y_pred, probs, contexts = [], [], [], []
for batch in result:
    for p in batch["predictions"]:
        y_true.append(p["true_label"])
        y_pred.append(p["label"])
        probs.append(p["probability"])
        contexts.append(p["context"])

df_result = pd.DataFrame({
    "y_true": y_true,
    "y_pred": y_pred,
    "probability": probs,
    "text": contexts
})
print(df_result)


