import numpy as np
import pandas as pd

import random
from pprint import pprint

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import mlflow
import mlflow.lightgbm

from utils import (
    reduce_mem_usage, le_lgb, fetch_logged_data, predict_models
)

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):

    # ランダムシードの設定
    np.random.seed(1234)
    random.seed(1234)

    # データの読み込みとメモリ削減
    df = reduce_mem_usage(pd.read_csv(cfg.lgb.df))

    # ラベルエンコーディング
    df = le_lgb(df)

    # trainとtestに再分割する
    train = df[~df['TARGET'].isnull()]
    test = df[df['TARGET'].isnull()]

    """
    train
    """

    # 説明変数と目的変数を指定
    X_train = train.drop(cfg.lgb.drop_col, axis=1)
    Y_train = train['TARGET']

    # K分割する
    skf = StratifiedKFold(n_splits=cfg.lgb.folds.num,
                          shuffle=cfg.lgb.folds.shuffle,
                          random_state=1234)

    models = []
    aucs = []

    params = {
        'boosting_type': cfg.lgb.params.boosting_type,
        'objective': cfg.lgb.params.objective,
        'metric': cfg.lgb.params.metric,
        'learning_rate': cfg.lgb.params.learning_rate,
        'num_leaves': cfg.lgb.params.num_leaves,
        'n_estimators': cfg.lgb.params.n_estimators,
        'random_state': cfg.lgb.params.random_state,
        'importance_type': cfg.lgb.params.importance_type,
    }

    # auto logging
    mlflow.lightgbm.autolog()

    for train_index, val_index in skf.split(X_train, Y_train):
        x_train = X_train.iloc[train_index]
        x_valid = X_train.iloc[val_index]
        y_train = Y_train.iloc[train_index]
        y_valid = Y_train.iloc[val_index]

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=lgb_eval,
            num_boost_round=cfg.lgb.model.num_boost_round,
            early_stopping_rounds=cfg.lgb.model.early_stopping_rounds,
            verbose_eval=cfg.lgb.model.verbose_eval
        )

        y_pred = model.predict(x_valid, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_valid, y_pred)
        print(auc)
        aucs.append(auc)

        models.append(model)

        run_id = mlflow.last_active_run().info.run_id
        print("Logged data and model in run {}".format(run_id))

        # show logged data
        for key, data in fetch_logged_data(run_id).items():
            print("\n---------- logged {} ----------".format(key))
            pprint(data)

    aucs = np.array(aucs)
    print(f'aucs: {np.mean(aucs):.2f} ± {np.std(aucs):.2f}')

    """
    inference
    """

    # 説明変数を指定
    X_test = test.drop(cfg.lgb.drop_col, axis=1)

    # 提出用サンプルの読み込み
    sub = pd.read_csv(
        '../input/home-credit-default-risk/sample_submission.csv')

    # 推論
    sub = predict_models(X_test, models, sub)

    # ファイルのエクスポート
    sub.to_csv(cfg.lgb.sub.name, index=False)


if __name__ == '__main__':
    main()
