import numpy as np
import pandas as pd

from utils import reduce_mem_usage


# DAYS_EMPLOYEDの欠損値の対応
def fill_null_days_employed(df):
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    return df


# 総所得金額を世帯人数で割った値
def income_div_person(df):
    df['INCOME_div_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    return df


# 総所得金額を就労期間で割った値
def income_div_employed(df):
    df['INCOME_div_EMPLOYED'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    return df


# 外部スコアの平均値など
def ext_source_stats(df):
    df["EXT_SOURCE_mean"] = df[["EXT_SOURCE_1",
                                "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_SOURCE_max"] = df[["EXT_SOURCE_1",
                               "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1)
    df["EXT_SOURCE_min"] = df[["EXT_SOURCE_1",
                               "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
    df["EXT_SOURCE_std"] = df[["EXT_SOURCE_1",
                               "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    df["EXT_SOURCE_count"] = df[["EXT_SOURCE_1",
                                 "EXT_SOURCE_2", "EXT_SOURCE_3"]].notnull().sum(axis=1)
    return df


# 就労期間を年齢で割った値 (年齢に占める就労期間の割合)
def days_employed_div_birth(df):
    df['DAYS_EMPLOYED_div_BIRTH'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    return df


# 年金支払額を所得金額で割った値
def annuity_div_income(df):
    df['ANNUITY_div_INCOME'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    return df


# 年金支払額を借入金で割った値
def annuity_div_credit(df):
    df['ANNUITY_div_CREDIT'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    return df


def main():

    # データの読み込み
    train = pd.read_csv(
        '../input/home-credit-default-risk/application_train.csv')
    test = pd.read_csv(
        '../input/home-credit-default-risk/application_test.csv')
    featured_pos = pd.read_csv('../input/processed/featured_pos.csv')

    # メモリの削減
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    featured_pos = reduce_mem_usage(featured_pos)

    # 学習データとテストデータの連結
    df = pd.concat([train, test], sort=False).reset_index(drop=True)

    # 特徴量エンジニアリング
    df = fill_null_days_employed(df)
    df = income_div_person(df)
    df = income_div_employed(df)
    df = ext_source_stats(df)
    df = days_employed_div_birth(df)
    df = annuity_div_income(df)
    df = annuity_div_credit(df)

    # featured_posを結合
    df = pd.merge(train, featured_pos, on='SK_ID_CURR', how='left')

    # 要素数が1つしかないカラムを削除
    df = df.loc[:, df.nunique() != 1]

    print(df.shape)

    # CSVファイルとして出力
    df.to_csv('../input/processed/featured_df.csv', header=True, index=False)


if __name__ == '__main__':
    main()
