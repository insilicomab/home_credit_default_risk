import pandas as pd

from utils import reduce_mem_usage


def main():
    pos = pd.read_csv("../input/home-credit-default-risk/POS_CASH_balance.csv")
    pos = reduce_mem_usage(pos)
    print(pos.shape)

    pos_ohe = pd.get_dummies(
        pos, columns=["NAME_CONTRACT_STATUS"], dummy_na=True)
    col_ohe = sorted(list(set(pos_ohe.columns) - set(pos.columns)))
    print(len(col_ohe))

    pos_ohe_agg = pos_ohe.groupby("SK_ID_CURR").agg(
        {
            # 数値の集約
            "MONTHS_BALANCE": ["mean", "std", "min", "max"],
            "CNT_INSTALMENT": ["mean", "std", "min", "max"],
            "CNT_INSTALMENT_FUTURE": ["mean", "std", "min", "max"],
            "SK_DPD": ["mean", "std", "min", "max"],
            "SK_DPD_DEF": ["mean", "std", "min", "max"],
            # カテゴリ変数をone-hot-encodingした値の集約
            "NAME_CONTRACT_STATUS_Active": ["mean"],
            "NAME_CONTRACT_STATUS_Amortized debt": ["mean"],
            "NAME_CONTRACT_STATUS_Approved": ["mean"],
            "NAME_CONTRACT_STATUS_Canceled": ["mean"],
            "NAME_CONTRACT_STATUS_Completed": ["mean"],
            "NAME_CONTRACT_STATUS_Demand": ["mean"],
            "NAME_CONTRACT_STATUS_Returned to the store": ["mean"],
            "NAME_CONTRACT_STATUS_Signed": ["mean"],
            "NAME_CONTRACT_STATUS_XNA": ["mean"],
            "NAME_CONTRACT_STATUS_nan": ["mean"],
            # IDのユニーク数をカウント (ついでにレコード数もカウント)
            "SK_ID_PREV": ["count", "nunique"],
        }
    )

    # カラム名の付与
    pos_ohe_agg.columns = [i + "_" + j for i, j in pos_ohe_agg.columns]
    pos_ohe_agg = pos_ohe_agg.reset_index(drop=False)
    print(pos_ohe_agg.shape)

    pos_ohe_agg.to_csv('../input/processed/featured_pos.csv', index=False)


if __name__ == '__main__':
    main()
