# coding=utf-8

from farm.infer import Inferencer
import pandas as pd
import os
from sklearn.metrics import classification_report

#%%
#DATA_DIR = "/tlhd/data/modeling/FU_data_dev20"
#SAVE_DIR = "saved_models/nohate_cfg13"
MODEL_DIR = "/network-ceph/aallhorn/output/nohate_coarse_merged_lm2cp3_fold1_best/network-ceph/aallhorn/output/lm-finetuning/step216000-german-NoHateCoarse"
IN_FILE = "/network-ceph/aallhorn/data/FU_data_merged/coarse_dev_fold1.tsv"
OUT_FILE = "/network-ceph/aallhorn/error-analysis/merged_coarse_dev_infer_results.csv"

# load test data
df = pd.read_csv(filepath_or_buffer=IN_FILE, delimiter="\t")

# build list of dicts for FARM
texts = []
for text in df["text"].values:
    texts.append({"text": text})

# Load saved model to make predictions
model = Inferencer.load(MODEL_DIR)
result = model.run_inference(dicts=texts)

y_pred, probs = [], []
i = 0
for batch in result:
    for p in batch["predictions"]:
        if df.iloc[i]["text"] != p["context"]:
            raise Exception("Order of input data and inference results does not match!")
        i += 1
        y_pred.append(p["label"])
        probs.append(p["probability"])

#%%
df_result = pd.DataFrame({
    "y_true": df["label"],
    "y_pred": y_pred,
    "probability": probs,
    "pred_type": "null",
    "text": df["text"]
})

df_result.loc[(df_result.y_true=="hate") & (df_result.y_pred=="hate"), "pred_type"] = "TP"
df_result.loc[(df_result.y_true=="hate") & (df_result.y_pred=="nohate"), "pred_type"] = "FN"
df_result.loc[(df_result.y_true=="nohate") & (df_result.y_pred=="hate"), "pred_type"] = "FP"
df_result.loc[(df_result.y_true=="nohate") & (df_result.y_pred=="nohate"), "pred_type"] = "TN"

print(df_result)
df_result.to_csv(OUT_FILE)

#%%
#df_result = pd.read_csv("/tlhd/docs/error-analysis/coarse_dev_infer_results.csv")
print(classification_report(df_result.y_true, df_result.y_pred))
