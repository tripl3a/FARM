# coding=utf-8
#%%
from farm.infer import Inferencer
import pandas as pd
import os

DATA_DIR = "/tlhd/data/modeling/FU_data_subsample"
SAVE_DIR = "/tlhd/models/nohate01"

# load test data
test_file = os.path.join(DATA_DIR, "coarse_test.tsv")
df_test = pd.read_csv(filepath_or_buffer=test_file, delimiter="\t")

# build list of dicts for FARM
texts = []
for text in df_test["text"].values:
    texts.append({"text": text})

# Load saved model to make predictions
model = Inferencer.load(SAVE_DIR)
result = model.run_inference(dicts=texts)

y_pred, probs = [], []
i = 0
for batch in result:
    for p in batch["predictions"]:
        if df_test.iloc[i]["text"] != p["context"]:
            raise Exception("Order of input data and inference results does not match!")
        i += 1
        y_pred.append(p["label"])
        probs.append(p["probability"])

#%%
df_result = pd.DataFrame({
    "y_true": df_test["label"],
    "y_pred": y_pred,
    "probability": probs,
    "text": df_test["text"]
})
print(df_result)
df_result.to_csv(os.path.join(DATA_DIR, "coarse_test_infer_results.csv"))
