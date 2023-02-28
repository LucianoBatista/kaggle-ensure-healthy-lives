from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from feature_engine.encoding import RareLabelEncoder
from feature_engine.outliers import Winsorizer
from imblearn.over_sampling import ADASYN, SMOTE

# from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


@dataclass
class Params:
    columns_to_keep: list
    columns_to_drop: list
    categorical_features: list
    categorical_features_order_matters: list
    numerical_features: list
    target_column: str


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

    def adjust_OBES_IMC(self):
        # remove comman from OBES_IMC column
        self.train_data["OBES_IMC"] = (
            self.train_data["OBES_IMC"].str.replace(",", ".").astype(float)
        )
        self.test_data["OBES_IMC"] = (
            self.test_data["OBES_IMC"].str.replace(",", ".").astype(float)
        )

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
                # (
                #     "winsorizer",
                #     Winsorizer(
                #         capping_method="gaussian",
                #         fold=3,
                #         tail="both",
                #     ),
                # ),
                ("scaler", StandardScaler()),
            ]
        )

        # high cardinality preprocessor
        _ = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "rare_label_encoder",
                    RareLabelEncoder(
                        tol=0.05,
                        n_categories=500,
                        ignore_format=True,
                    ),
                ),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )

        # preprocessor = make_pipeline(
        #     (
        #         SimpleImputer(
        #             strategy="most_frequent",
        #             variables=self.params["high_cardinality_features"],
        #         )
        #     ),
        #     (
        #         RareLabelEncoder(
        #             tol=0.05,
        #             n_categories=15,
        #             ignore_format=True,
        #             variables=self.params["high_cardinality_features"],
        #         )
        #     ),
        #     (
        #         OrdinalEncoder(
        #             handle_unknown="use_encoded_value",
        #             unknown_value=-1,
        #             variables=self.params["high_cardinality_features"],
        #         )
        #     ),
        #     (StandardScaler(variables=self.params["numerical_features"])),
        #     (
        #         OrdinalEncoder(
        #             handle_unknown="use_encoded_value",
        #             unknown_value=-1,
        #             variables=self.params["categorical_features"],
        #         )
        #     ),
        # )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    numerical_preprocessor,
                    self.params["numerical_features"],
                ),
                # (
                #     "high_cardinality",
                #     high_cardinality_preprocessor,
                #     self.params["high_cardinality_features"],
                # ),
                (
                    "categorical",
                    categorical_preprocessor,
                    self.params["categorical_features"],
                ),
            ]
        )

        return preprocessor

    def _create_estimator_best_score(self):
        clf_1 = HistGradientBoostingClassifier(
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
            reg_alpha=1.5,
            reg_lambda=1.5,
        )
        clf_3 = LogisticRegression(
            random_state=42,
            class_weight={0: 2, 1: 2, 2: 2, 3: 1, 4: 1},
            max_iter=2000,
            C=1.5,
            penalty="l2",
        )

        clf = VotingClassifier(
            estimators=[
                ("HistGradientBoostingClassifier", clf_1),
                ("LGBMClassifier", clf_2),
                # ("LogisticRegression", clf_3),
            ],
            voting="hard",
        )
        # rf = HistGradientBoostingClassifier()

        return clf

    def create_pipeline(self):
        preprocessor = self._create_preprocessor_best_score()
        clf = self._create_estimator_best_score()
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                # (
                #     "SMOTE",
                #     ADASYN(
                #         random_state=42,
                #         sampling_strategy="not majority",
                #     ),
                # ),
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
            "TP_ANTIVIR",
            # "CO_MUN_NOT",
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
        # X_train = X_train.astype({"CO_MUN_NOT": "category", "CO_REGIONA": "category"})
        # X_test = X_test.astype({"CO_MUN_NOT": "category", "CO_REGIONA": "category"})
        print(X_train.shape)
        print(X_test.shape)
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
            "TP_ANTIVIR",
            # "CO_MUN_NOT",
            "ID",
        ]
        x_test_id = self.test_data["ID"]
        x_test = self.test_data.drop(columns_to_drop_test, axis=1)
        y_pred = pipe.predict(x_test)

        submission = pd.DataFrame({"ID": x_test_id, "CLASSI_FIN": y_pred})
        submission.to_csv(file_name, index=False)
        print("Done!")
        print("Done!")