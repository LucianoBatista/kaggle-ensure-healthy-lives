from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.tree import ExtraTreeClassifier
from unidecode import unidecode
from xgboost import XGBClassifier


@dataclass
class Params:
    columns_to_keep: list
    columns_to_drop: list
    categorical_features: list
    categorical_features_order_matters: list
    numerical_features: list
    target_column: str


# cleaning municipios names
def to_capitalize(x):
    nome = pl.Series([i.upper() for i in x])
    return nome


def remove_accents(x):
    nome = pl.Series([unidecode(i) for i in x])
    return nome


def cut_on_6_digits(x):
    digits = pl.Series([str(i)[:6] for i in x])
    digits = pl.Series([int(i) for i in digits])
    return digits


class Pipes:
    def __init__(self, train_data: Path, test_data: Path, params: Path):
        self.train_data_path = train_data
        self.test_data_path = test_data
        self.params_path = params
        self.params: Params = self._get_params()
        self.train_data, self.test_data = self._get_data()

    def _get_params(self):
        # load from yaml file
        with open(self.params_path) as f:
            params = yaml.safe_load(f)
        return params

    def _get_data(self):
        train_data = pd.read_csv(self.train_data_path, low_memory=False)
        test_data = pd.read_csv(self.test_data_path, low_memory=False)
        return train_data, test_data

    def _create_estado_municipio_lat_long(self, data):
        municipios = pl.read_csv("data/municipios.csv")
        estados = pl.read_csv("data/estados.csv")

        municipios_clean = (
            municipios.with_columns([pl.col("nome_m").map(to_capitalize)])
            .with_columns([pl.col("nome_m").map(remove_accents)])
            .with_columns([pl.col("Código IBGE").map(cut_on_6_digits)])
        )
        municipios_clean_pd = municipios_clean.to_pandas()
        data = data.merge(
            right=municipios_clean_pd,
            how="left",
            left_on=["ID_MUNICIP", "CO_MUN_NOT"],
            right_on=["nome_m", "Código IBGE"],
            suffixes=("_left", "_right"),
        )
        data = data.merge(
            right=estados.to_pandas(),
            how="left",
            left_on="SG_UF_NOT",
            right_on="uf",
            suffixes=("_left", "_right"),
        )
        data = data.drop(
            [
                "nome_m",
                "uf",
                "Código IBGE",
                "codigo_uf",
                "siafi_id",
                "ddd",
                "fuso_horario",
                "Densidade",
            ],
            axis=1,
        )
        return data

    def create_lat_long_others(self):
        self.train_data = self._create_estado_municipio_lat_long(self.train_data)
        self.test_data = self._create_estado_municipio_lat_long(self.test_data)

    def _create_preprocessor_best_score(self):
        # categorical preprocessor
        categorical_preprocessor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )

        # numerical preprocessor
        numerical_preprocessor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    numerical_preprocessor,
                    self.params["numerical_features"],
                ),
                (
                    "categorical",
                    categorical_preprocessor,
                    self.params["categorical_features"],
                ),
            ]
        )

        return preprocessor

    def _create_estimator_best_score(self):
        _ = HistGradientBoostingClassifier(
            max_iter=2000,
            random_state=42,
            class_weight={0: 2, 1: 2, 2: 2, 3: 1, 4: 1},
            max_depth=100,
            l2_regularization=1.5,
            max_bins=255,
            scoring="f1_macro",
        )
        clf_2 = LGBMClassifier(
            random_state=42,
            class_weight={0: 2, 1: 2, 2: 2, 3: 1, 4: 1},
            max_depth=100,
            n_estimators=2000,
            num_leaves=255,
            min_child_samples=2000,
            colsample_bytree=0.75,
            subsample_freq=7,
            reg_alpha=1.5,
            reg_lambda=1.5,
        )
        _ = ExtraTreeClassifier(
            random_state=42,
            class_weight={0: 2, 1: 2, 2: 2, 3: 1, 4: 1},
            max_depth=100,
            min_samples_leaf=10,
            min_samples_split=10,
            splitter="random",
        )
        _ = LogisticRegression(
            random_state=42,
            class_weight={0: 2, 1: 2, 2: 2, 3: 1, 4: 1},
            max_iter=10000,
            C=1.5,
            penalty="l2",
        )
        clf_5 = XGBClassifier(
            random_state=42,
            max_depth=100,
            n_estimators=2000,
            reg_alpha=1.5,
            reg_lambda=1.5,
            tree_method="gpu_hist",
            gpu_id=0,
        )

        _ = CatBoostClassifier(
            random_state=42,
            class_weights={0: 2, 1: 2, 2: 2, 3: 1, 4: 1},
            learning_rate=0.01,
            n_estimators=10000,
            loss_function="MultiClass",
            reg_lambda=1.5,
        )

        _ = RandomForestClassifier(
            random_state=42,
            class_weight={0: 2, 1: 2, 2: 2, 3: 1, 4: 1},
            max_depth=100,
            n_estimators=2000,
            max_features=0.5,
            min_samples_leaf=1,
            min_samples_split=2,
            criterion="entropy",
        )

        clf = VotingClassifier(
            estimators=[
                ("XGBoost", clf_5),
                ("LGBMClassifier", clf_2),
            ],
            voting="hard",
        )

        return clf

    def create_pipeline(self):
        preprocessor = self._create_preprocessor_best_score()
        clf = self._create_estimator_best_score()
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )
        return pipe

    def _create_train_test_split(self):
        columns_to_drop = [
            "ID_REGIONA",
            "ID_MUNICIP",
            "OUTRO_DES",
            "MORB_DESC",
            "OUT_AMOST",
            "PAC_COCBO",
            "PAC_DSCBO",
            "TOMO_OUT",
            "TP_IDADE",
            "DELTA_UTI",
            "DOSE_REF",
            "OUT_ANIM",
            "OBES_IMC",
            "M_AMAMENTA",
            "MAE_VAC",
            "ID",
            "CLASSI_FIN",
        ]
        X = self.train_data.drop(columns_to_drop, axis=1)
        y = self.train_data["CLASSI_FIN"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def create_grid_search(self, pipe):
        param_grid = {
            "classifier__max_depth": [25, 50, 75, 100],
            "classifier__l2_regularization": [0.5, 1.0, 1.5, 2.0],
            "classifier__class_weight": [{0: 2, 1: 2, 2: 2, 3: 1, 4: 1}],
            "classifier__learning_rate": [0.01, 0.1, 1, 10],
            "classifier__max_bins": [50, 100, 255],
            "classifier__max_iter": [500, 1000, 1500, 2000],
            "classifier__max_leaf_nodes": [3, 10, 30],
            "classifier__random_state": [42],
            "classifier__scoring": ["f1_macro"],
        }

        grid_search = GridSearchCV(
            pipe,
            param_grid,
            cv=5,
            n_jobs=5,
            verbose=1,
            scoring="f1_macro",
            return_train_score=True,
        )

        return grid_search

    def fit(self, pipe):
        X_train, X_test, y_train, y_test = self._create_train_test_split()
        pipe.fit(X_train, y_train)

        # evaluate
        y_pred = pipe.predict(X_test)
        print(classification_report(y_test, y_pred, digits=4))

        return pipe

    def fit_without_split(self, pipe):
        X = self.train_data.drop(["CLASSI_FIN", "ID"], axis=1)
        y = self.train_data["CLASSI_FIN"]
        pipe.fit(X, y)

        # evaluate
        y_pred = pipe.predict(X)
        print(classification_report(y, y_pred, digits=4))

        return pipe

    def create_submission_file(self, pipe, file_name: str):
        columns_to_drop_test = [
            "ID_REGIONA",
            "ID_MUNICIP",
            "OUTRO_DES",
            "MORB_DESC",
            "OUT_AMOST",
            "PAC_COCBO",
            "PAC_DSCBO",
            "TOMO_OUT",
            "TP_IDADE",
            "DELTA_UTI",
            "DOSE_REF",
            "OUT_ANIM",
            "OBES_IMC",
            "M_AMAMENTA",
            "MAE_VAC",
            "ID",
        ]
        x_test_id = self.test_data["ID"]
        x_test = self.test_data.drop(columns_to_drop_test, axis=1)
        y_pred = pipe.predict(x_test)

        submission = pd.DataFrame({"ID": x_test_id, "CLASSI_FIN": y_pred})
        submission.to_csv(file_name, index=False)
        print("Done!")
