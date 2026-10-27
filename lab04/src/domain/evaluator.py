import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from domain.features import FeatureEngineeringFactory

class ModelBenchmarkingSuite:
    def __init__(self, seed: int = 73841):
        self.seed = seed

    def benchmark(self, X_train, X_test, y_train, y_test, num_cols, cat_cols, out_json: str):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        scaler = StandardScaler()

        # Step A: Without Feature Engineering
        X_tr_cat = ohe.fit_transform(X_train[cat_cols])
        X_te_cat = ohe.transform(X_test[cat_cols])

        X_tr_num = scaler.fit_transform(X_train[num_cols])
        X_te_num = scaler.transform(X_test[num_cols])

        mat_tr_base = np.hstack([X_tr_num, X_tr_cat])
        mat_te_base = np.hstack([X_te_num, X_te_cat])

        lr_base = LogisticRegression(max_iter=1000, random_state=self.seed)
        lr_base.fit(mat_tr_base, y_train)
        pred_base = lr_base.predict(mat_te_base)

        res_without = {
            "accuracy": round(float(accuracy_score(y_test, pred_base)), 4),
            "f1": round(float(f1_score(y_test, pred_base)), 4),
            "precision": round(float(precision_score(y_test, pred_base)), 4),
            "recall": round(float(recall_score(y_test, pred_base)), 4)
        }

        # Step B: With Feature Engineering
        X_tr_eng = FeatureEngineeringFactory.construct_features(X_train)
        X_te_eng = FeatureEngineeringFactory.construct_features(X_test)
        eng_nums = num_cols + ["capital_net", "hours_per_age"]

        scaler2 = StandardScaler()
        X_tr_num_eng = scaler2.fit_transform(X_tr_eng[eng_nums])
        X_te_num_eng = scaler2.transform(X_te_eng[eng_nums])

        mat_tr_eng = np.hstack([X_tr_num_eng, X_tr_cat])
        mat_te_eng = np.hstack([X_te_num_eng, X_te_cat])

        lr_eng = LogisticRegression(max_iter=1000, random_state=self.seed)
        lr_eng.fit(mat_tr_eng, y_train)
        pred_eng = lr_eng.predict(mat_te_eng)

        res_with = {
            "accuracy": round(float(accuracy_score(y_test, pred_eng)), 4),
            "f1": round(float(f1_score(y_test, pred_eng)), 4),
            "precision": round(float(precision_score(y_test, pred_eng)), 4),
            "recall": round(float(recall_score(y_test, pred_eng)), 4)
        }

        summary = {
            "baseline_model": res_without,
            "engineered_features_model": res_with,
            "delta": {
                "accuracy_improvement": round(res_with["accuracy"] - res_without["accuracy"], 4),
                "f1_improvement": round(res_with["f1"] - res_without["f1"], 4)
            }
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        return summary
