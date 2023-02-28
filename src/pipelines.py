from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier


@dataclass
class Params:
    columns_to_keep: list
    columns_to_drop: list
    target_column: str
    id_column: str


class Pipeline:
    def __init__(self, train_data: Path, test_data: Path, params: Path):
        self.train_data_path = train_data
        self.test_data_path = test_data
        self.params_path = params
        self.x_test_kaggle = None
        self.id_test_kaggle = None
        self.params: Params = self._get_params()
        self.train_data, self.target = self._get_data()

    def _get_params(self):
        # load from yaml file
        with open(self.params_path) as f:
            params = yaml.safe_load(f)
        return params

    def _get_data(self):
        train_data = pd.read_csv(self.train_data_path, low_memory=False)
        train_data_sampled = train_data[self.params["columns_to_keep"]]
        target = train_data[self.params["target_column"]]

        return train_data_sampled, target

    def create_train_test_split(self):
        # split data from scikit-learn
        x_train, x_test, y_train, y_test = train_test_split(
            self.train_data,
            self.target,
            test_size=0.2,
            random_state=42,
            stratify=self.target,
        )
        return x_train, x_test, y_train, y_test

    def impute_missing_values(self, x_train, x_test):
        columns = x_train.columns
        x_test_kaggle = pd.read_csv(self.test_data_path, low_memory=False)
        id_column = x_test_kaggle["ID"]
        x_test_kaggle = x_test_kaggle[self.params["columns_to_keep"]]

        # convert to category
        nu_idade_column_train = x_train["NU_IDADE_N"].values
        nu_idade_column_test = x_test["NU_IDADE_N"].values
        nu_idade_column_test_kaggle = x_test_kaggle["NU_IDADE_N"].values

        x_train.drop(columns=["NU_IDADE_N"], inplace=True)
        x_test.drop(columns=["NU_IDADE_N"], inplace=True)
        x_test_kaggle.drop(columns=["NU_IDADE_N"], inplace=True)

        x_train = x_train.astype("category")
        x_test = x_test.astype("category")
        x_test_kaggle = x_test_kaggle.astype("category")

        x_train["NU_IDADE_N"] = nu_idade_column_train
        x_test["NU_IDADE_N"] = nu_idade_column_test
        x_test_kaggle["NU_IDADE_N"] = nu_idade_column_test_kaggle

        for column in columns:
            df = x_train[column].value_counts().reset_index()
            unique_values = df["index"].values
            df["proportion"] = df[column] / df[column].sum()
            x_train[column].fillna(
                np.random.choice(
                    list(unique_values), 1, p=list(df["proportion"].values)
                ).item(),
                inplace=True,
            )
            x_test[column].fillna(
                np.random.choice(
                    list(unique_values), 1, p=list(df["proportion"].values)
                ).item(),
                inplace=True,
            )
            x_test_kaggle[column].fillna(
                np.random.choice(
                    list(unique_values), 1, p=list(df["proportion"].values)
                ).item(),
                inplace=True,
            )

        self.id_test_kaggle = id_column
        self.x_test_kaggle = x_test_kaggle

        return x_train, x_test

    def transform_to_numeric(self, x_train, x_test):
        # transformer CS_SEXO to numeric
        x_train["CS_SEXO"] = x_train["CS_SEXO"].map({"M": 0, "F": 1, "I": 2})
        x_test["CS_SEXO"] = x_test["CS_SEXO"].map({"M": 0, "F": 1, "I": 2})
        return x_train, x_test

    def ohe_categorical_columns(self, x_train, x_test):
        columns_to_ohe = self.params["columns_to_keep"]
        columns_to_ohe.remove("NU_IDADE_N")
        # one hot encoding
        ohe = make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                columns_to_ohe,
            ),
            remainder="passthrough",
        )
        x_train = ohe.fit_transform(x_train)
        x_test = ohe.transform(x_test)
        self.x_test_kaggle = ohe.transform(self.x_test_kaggle)

        return x_train, x_test

    def train_rf(self, x_train, y_train):
        rf = RandomForestClassifier(
            n_estimators=1000,
            random_state=42,
            n_jobs=10,
            class_weight={1: 2, 2: 2, 3: 2, 4: 1, 5: 1},
        )
        rf.fit(x_train, y_train.values.ravel())
        return rf

    def train_histogram_gradient_boosting(self, x_train, y_train):
        hgb = HistGradientBoostingClassifier(
            max_iter=2000,
            random_state=42,
            class_weight={0: 2, 1: 2, 2: 2, 3: 1, 4: 1},
            max_depth=75,
            l2_regularization=1.5,
            scoring="f1_macro",
        )
        hgb.fit(x_train, y_train.values.ravel())
        return hgb

    def train_lightgbm(self, x_train, y_train):
        lgbm = LGBMClassifier(
            n_estimators=1000, random_state=42, class_weight="balanced", n_jobs=10
        )
        lgbm.fit(x_train, y_train.values.ravel())
        return lgbm

    def train_xgboost(self, x_train, y_train):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        xgb = XGBClassifier(
            n_estimators=1000,
            random_state=42,
            class_weight="balanced",
            n_jobs=10,
        )
        xgb.fit(x_train.astype("float64"), y_train.ravel())
        return xgb

    def train_logistic_regression(self, x_train, y_train):
        lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            n_jobs=10,
        )
        lr.fit(x_train, y_train.values.ravel())
        return lr

    def train_catboost(self, x_train, y_train):
        cat = CatBoostClassifier(
            n_estimators=1000,
            random_state=42,
            # cat_features=[self.params["columns_to_keep"]]
            # class_weight="balanced",
            # n_jobs=10,
        )
        cat.fit(x_train.astype("float64"), y_train.values.ravel())
        return cat

    def evaluate_model(self, rf, x_test, y_test):
        # evaluating the model
        y_pred = rf.predict(x_test)
        classification_report_rf = classification_report(y_test, y_pred)
        print(classification_report_rf)

    def generate_submission_file(self, rf):
        # generate submission file
        test_data = self.x_test_kaggle.copy()
        test_data["CS_SEXO"] = test_data["CS_SEXO"].map({"M": 0, "F": 1, "I": 2})

        y_pred = rf.predict(test_data)

        submission = pd.DataFrame({"ID": self.id_test_kaggle, "CLASSI_FIN": y_pred})
        submission.to_csv("data/submission_v2.csv", index=False)
