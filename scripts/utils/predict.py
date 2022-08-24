import numpy as np
import pandas as pd


# 推論関数の定義
def predict_models(input_x, models_list: list, sub_df):

    preds = []

    for model in models_list:
        pred = model.predict(input_x)
        preds.append(pred)

    preds_array = np.array(preds)
    preds_mean = np.mean(preds_array, axis=0)

    sub_df['TARGET'] = preds_mean

    return sub_df
