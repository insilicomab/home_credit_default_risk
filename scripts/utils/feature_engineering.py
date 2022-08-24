import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


# LightGBM用のラベルエンコーディング
def le_lgb(df):
    # object型の変数の取得
    categories = df.columns[df.dtypes == 'object']

    # 欠損値を数値に変換
    for cat in categories:
        le = LabelEncoder()
        print(cat)

        df[cat].fillna('missing', inplace=True)
        le = le.fit(df[cat])
        df[cat] = le.transform(df[cat])
        # LabelEncoderは数値に変換するだけであるため、最後にastype('category')としておく
        df[cat] = df[cat].astype('category')

    return df


# XGBoost用のラベルエンコーディング
def le_xgb(df):
    # object型の変数の取得
    categories = df.columns[df.dtypes == 'object']
    print(categories)

    # 欠損値を数値に変換
    for cat in categories:
        le = LabelEncoder()
        print(cat)

        df[cat].fillna('missing', inplace=True)
        le = le.fit(df[cat])
        df[cat] = le.transform(df[cat])
        # LabelEncoderは数値に変換するだけであるため、最後にastype('int8')としておく
        df[cat] = df[cat].astype('int8')

    return df
