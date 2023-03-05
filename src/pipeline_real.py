from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
import yaml
from catboost import CatBoostClassifier
from feature_engine.encoding import RareLabelEncoder
from feature_engine.outliers import Winsorizer
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as imb_pipeline
from imblearn.under_sampling import RandomUnderSampler

# from imblearn.pipeline import Pipeline
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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
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

    def _handle_obes_imc(self, data: pd.DataFrame):
        # remove comman from OBES_IMC column
        data_pl = pl.from_pandas(data)
        data = data_pl.with_columns(
            pl.col("OBES_IMC").str.replace(",", ".").cast(pl.Float64).alias("OBES_IMC")
        ).with_columns(
            [
                pl.when((pl.col("OBESIDADE") == 1) & (pl.col("OBES_IMC") < 30))
                .then(30)
                .when((pl.col("OBESIDADE") == 1) & (pl.col("OBES_IMC") >= 100))
                .then(30)
                .when(pl.col("OBES_IMC").is_null())
                .then(24)
                .otherwise(pl.col("OBES_IMC"))
                .alias("OBES_IMC")
            ]
        )
        data = data.to_pandas()

        return data

    def _handle_idade(self, data: pd.DataFrame):
        data_pl = pl.from_pandas(data)
        data = data_pl.with_columns(
            [
                pl.when(pl.col("TP_IDADE").is_in([1, 2]))
                .then(0)
                .when(pl.col("TP_IDADE") == 3)
                .then(pl.col("NU_IDADE_N"))
                .otherwise(pl.col("NU_IDADE_N"))
                .alias("NU_IDADE_N")
            ]
        )
        data = data.to_pandas()

        return data

    def _handle_antiviral(self, data: pd.DataFrame):
        data_pl = pl.from_pandas(data)
        data = data_pl.with_columns(
            [
                pl.when(pl.col("ANTIVIRAL") == 1)
                .then(pl.col("TP_ANTIVIR").fill_null(1))
                .when(pl.col("TP_ANTIVIR").is_null())
                .then(0)
                .otherwise(pl.col("TP_ANTIVIR"))
                .alias("TP_ANTIVIR")
            ]
        )
        data = data.to_pandas()

        return data

    def _handle_tomo_res(self, data: pd.DataFrame):
        data_pl = pl.from_pandas(data)
        data = data_pl.with_columns([pl.col("TOMO_RES").fill_null(6).alias("TOMO_RES")])
        data = data.to_pandas()

        return data

    def _create_a_risk_factor_variable(self, data: pd.DataFrame):
        data_pl = pl.from_pandas(data)
        data_pl = (
            data_pl.with_columns(
                [
                    pl.when((pl.col("CARDIOPATI") == 2))
                    .then(0)
                    .otherwise(pl.col("CARDIOPATI"))
                    .alias("CARDIOPATI")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("HEMATOLOGI") == 2))
                    .then(0)
                    .otherwise(pl.col("HEMATOLOGI"))
                    .alias("HEMATOLOGI")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("SIND_DOWN") == 2))
                    .then(0)
                    .otherwise(pl.col("SIND_DOWN"))
                    .alias("SIND_DOWN")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("HEPATICA") == 2))
                    .then(0)
                    .otherwise(pl.col("HEPATICA"))
                    .alias("HEPATICA")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("ASMA") == 2))
                    .then(0)
                    .otherwise(pl.col("ASMA"))
                    .alias("ASMA")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("DIABETES") == 2))
                    .then(0)
                    .otherwise(pl.col("DIABETES"))
                    .alias("DIABETES")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("NEUROLOGIC") == 2))
                    .then(0)
                    .otherwise(pl.col("NEUROLOGIC"))
                    .alias("NEUROLOGIC")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("PNEUMOPATI") == 2))
                    .then(0)
                    .otherwise(pl.col("PNEUMOPATI"))
                    .alias("PNEUMOPATI")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("IMUNODEPRE") == 2))
                    .then(0)
                    .otherwise(pl.col("IMUNODEPRE"))
                    .alias("IMUNODEPRE")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("RENAL") == 2))
                    .then(0)
                    .otherwise(pl.col("RENAL"))
                    .alias("RENAL")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("OBESIDADE") == 2))
                    .then(0)
                    .otherwise(pl.col("OBESIDADE"))
                    .alias("OBESIDADE")
                ]
            )
            .with_columns(
                [
                    pl.sum(
                        [
                            "CARDIOPATI",
                            "HEMATOLOGI",
                            "SIND_DOWN",
                            "HEPATICA",
                            "ASMA",
                            "DIABETES",
                            "NEUROLOGIC",
                            "PNEUMOPATI",
                            "IMUNODEPRE",
                            "RENAL",
                            "OBESIDADE",
                        ]
                    ).alias("RISCK_FACTORS")
                ]
            )
        )
        data["RISCK_FACTORS"] = data_pl["RISCK_FACTORS"].to_numpy()
        data = data.drop(
            [
                "CARDIOPATI",
                "HEMATOLOGI",
                "SIND_DOWN",
                "HEPATICA",
                "ASMA",
                "DIABETES",
                "NEUROLOGIC",
                "PNEUMOPATI",
                "IMUNODEPRE",
                "RENAL",
                "OBESIDADE",
            ],
            axis=1,
        )

        return data

    def _create_sintoma_factor_variable(self, data: pd.DataFrame):
        data_pl = pl.from_pandas(data)
        data_pl = (
            data_pl.with_columns(
                [
                    pl.when((pl.col("NOSOCOMIAL") == 2))
                    .then(0)
                    .otherwise(pl.col("NOSOCOMIAL"))
                    .alias("NOSOCOMIAL")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("FEBRE") == 2))
                    .then(0)
                    .otherwise(pl.col("FEBRE"))
                    .alias("FEBRE")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("TOSSE") == 2))
                    .then(0)
                    .otherwise(pl.col("TOSSE"))
                    .alias("TOSSE")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("GARGANTA") == 2))
                    .then(0)
                    .otherwise(pl.col("GARGANTA"))
                    .alias("GARGANTA")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("DISPNEIA") == 2))
                    .then(0)
                    .otherwise(pl.col("DISPNEIA"))
                    .alias("DISPNEIA")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("DESC_RESP") == 2))
                    .then(0)
                    .otherwise(pl.col("DESC_RESP"))
                    .alias("DESC_RESP")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("SATURACAO") == 2))
                    .then(0)
                    .otherwise(pl.col("SATURACAO"))
                    .alias("SATURACAO")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("DIARREIA") == 2))
                    .then(0)
                    .otherwise(pl.col("DIARREIA"))
                    .alias("DIARREIA")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("VOMITO") == 2))
                    .then(0)
                    .otherwise(pl.col("VOMITO"))
                    .alias("VOMITO")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("OUTRO_SIN") == 2))
                    .then(0)
                    .otherwise(pl.col("OUTRO_SIN"))
                    .alias("OUTRO_SIN")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("PUERPERA") == 2))
                    .then(0)
                    .otherwise(pl.col("PUERPERA"))
                    .alias("PUERPERA")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("DOR_ABD") == 2))
                    .then(0)
                    .otherwise(pl.col("DOR_ABD"))
                    .alias("DOR_ABD")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("FADIGA") == 2))
                    .then(0)
                    .otherwise(pl.col("FADIGA"))
                    .alias("FADIGA")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("PERD_OLFT") == 2))
                    .then(0)
                    .otherwise(pl.col("PERD_OLFT"))
                    .alias("PERD_OLFT")
                ]
            )
            .with_columns(
                [
                    pl.when((pl.col("PERD_PALA") == 2))
                    .then(0)
                    .otherwise(pl.col("PERD_PALA"))
                    .alias("PERD_PALA")
                ]
            )
            .with_columns(
                [
                    pl.sum(
                        [
                            "NOSOCOMIAL",
                            "FEBRE",
                            "TOSSE",
                            "GARGANTA",
                            "DISPNEIA",
                            "DESC_RESP",
                            "SATURACAO",
                            "DIARREIA",
                            "VOMITO",
                            "OUTRO_SIN",
                            "PUERPERA",
                            "DOR_ABD",
                            "FADIGA",
                            "PERD_OLFT",
                            "PERD_PALA",
                        ]
                    ).alias("SINTOMAS_FACTORS")
                ]
            )
        )
        data["SINTOMAS_FACTORS"] = data_pl["SINTOMAS_FACTORS"].to_numpy()
        data = data.drop(
            [
                "NOSOCOMIAL",
                "FEBRE",
                "TOSSE",
                "GARGANTA",
                "DISPNEIA",
                "DESC_RESP",
                "SATURACAO",
                "DIARREIA",
                "VOMITO",
                "OUTRO_SIN",
                "PUERPERA",
                "DOR_ABD",
                "FADIGA",
                "PERD_OLFT",
                "PERD_PALA",
            ],
            axis=1,
        )
        return data

    def _create_estado_municipio_lat_long_correct(self, data):
        municipios = pl.read_csv("data/municipios.csv")
        estados = pl.read_csv("data/estados.csv")

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

        municipios_clean = (
            municipios.with_columns([pl.col("nome_m").map(to_capitalize)])
            .with_columns([pl.col("nome_m").map(remove_accents)])
            .with_columns([pl.col("codigo_ibge").map(cut_on_6_digits)])
        )
        municipios_clean_pd = municipios_clean.to_pandas()
        print(data.shape)
        data = data.merge(
            right=municipios_clean_pd,
            how="left",
            left_on=["ID_MUNICIP", "CO_MUN_NOT"],
            right_on=["nome_m", "codigo_ibge"],
            suffixes=("_left", "_right"),
        )
        data = data.merge(
            right=estados.to_pandas(),
            how="left",
            left_on="SG_UF_NOT",
            right_on="uf",
            suffixes=("_left", "_right"),
        )
        print(data.shape)
        data = data.drop(
            [
                "nome_m",
                "uf",
                "codigo_ibge",
                "codigo_uf",
                "siafi_id",
                "ddd",
                "fuso_horario",
            ],
            axis=1,
        )
        return data

    def _create_estado_municipio_lat_long_wrong(self, data):
        municipios = pl.read_csv("data/municipios.csv")
        estados = pl.read_csv("data/estados.csv")

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

        municipios_clean = municipios.with_columns(
            [pl.col("nome_m").map(to_capitalize)]
        ).with_columns([pl.col("nome_m").map(remove_accents)])
        municipios_clean_pd = municipios_clean.to_pandas()
        print(data.shape)
        data = data.merge(
            right=municipios_clean_pd,
            how="left",
            left_on="ID_MUNICIP",
            right_on="nome_m",
            suffixes=("_left", "_right"),
        )
        data = data.merge(
            right=estados.to_pandas(),
            how="left",
            left_on="SG_UF_NOT",
            right_on="uf",
            suffixes=("_left", "_right"),
        )
        print(data.shape)
        data = data.drop(
            [
                "nome_m",
                "uf",
                "codigo_ibge",
                "codigo_uf",
                "siafi_id",
                "ddd",
                "fuso_horario",
            ],
            axis=1,
        )
        return data

    def _oversampling_data(self, data):
        # smote_enn = SMOTEENN(random_state=0)
        # X_resampled, y_resampled = smote_enn.fit_resample(X, y)

        # smote_tomek = SMOTETomek(random_state=0)
        # X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

        # pipeline = imb_pipeline(steps=steps)
        # X, y = pipeline.fit_resample(
        #     data.drop(["CLASSI_FIN"], axis=1), data["CLASSI_FIN"]
        # )
        # data = pd.concat([X, y], axis=1)
        return data

    def adjust_OBES_IMC(self):
        # remove comman from OBES_IMC column
        self.train_data = self._handle_obes_imc(self.train_data)
        self.test_data = self._handle_obes_imc(self.test_data)

    def adjust_idade(self):
        self.train_data = self._handle_idade(self.train_data)
        self.test_data = self._handle_idade(self.test_data)

    def adjust_antiviral(self):
        self.train_data = self._handle_antiviral(self.train_data)
        self.test_data = self._handle_antiviral(self.test_data)

    def adjust_tomo_res(self):
        self.train_data = self._handle_tomo_res(self.train_data)
        self.test_data = self._handle_tomo_res(self.test_data)

    def create_risck_variable(self):
        self.train_data = self._create_a_risk_factor_variable(self.train_data)
        self.test_data = self._create_a_risk_factor_variable(self.test_data)

    def create_sintomas_factors(self):
        self.train_data = self._create_sintoma_factor_variable(self.train_data)
        self.test_data = self._create_sintoma_factor_variable(self.test_data)

    def create_lat_long_others(self):
        # self.train_data = self._create_estado_municipio_lat_long(self.train_data)
        self.train_data = self._create_estado_municipio_lat_long_correct(
            self.train_data
        )
        self.test_data = self._create_estado_municipio_lat_long_correct(self.test_data)

    def oversampling_data(self):
        self.train_data = self._oversampling_data(self.train_data)

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
                # (
                #     "one_hot_encoder",
                #     OneHotEncoder(
                #         handle_unknown="ignore",
                #         sparse=False,
                #         drop="if_binary",
                #     ),
                # ),
            ]
        )

        # numerical preprocessor
        numerical_preprocessor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
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
        high_cardinality = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "rare_label_encoder",
                    RareLabelEncoder(
                        n_categories=100,
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
                #     high_cardinality,
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
        clf_3 = ExtraTreeClassifier(
            random_state=42,
            class_weight={0: 2, 1: 2, 2: 2, 3: 1, 4: 1},
            max_depth=100,
            max_features=0.5,
            min_samples_leaf=1,
            min_samples_split=2,
            splitter="random",
        )
        clf_4 = LogisticRegression(
            random_state=42,
            class_weight={0: 2, 1: 2, 2: 2, 3: 1, 4: 1},
            max_iter=2000,
            C=1.5,
            penalty="l2",
        )
        clf_5 = XGBClassifier(
            random_state=42,
            # class_weight={0: 2, 1: 2, 2: 2, 3: 1, 4: 1},
            max_depth=100,
            n_estimators=2000,
            reg_alpha=1.5,
            reg_lambda=1.5,
        )

        clf_6 = CatBoostClassifier(
            random_state=42,
            # iterations=2000,
            class_weights={0: 2, 1: 2, 2: 2, 3: 1, 4: 1},
            # max_depth=100,
            learning_rate=0.01,
            n_estimators=10000,
            loss_function="MultiClass",
            reg_lambda=1.5,
        )

        clf_7 = RandomForestClassifier(
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
                ("HistGradientBoostingClassifier", clf_1),
                ("LGBMClassifier", clf_2),
                # ("ExtraTree", clf_3),
                # ("LogistiRegression", clf_4),
                # ("XGBoost", clf_5),
                # ("RandomForest", clf_7),
            ],
            voting="hard",
        )
        # rf = HistGradientBoostingClassifier()

        return clf

    def create_pipeline(self):
        preprocessor = self._create_preprocessor_best_score()
        clf = self._create_estimator_best_score()
        pipe = imb_pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smoteenn", SMOTEENN(random_state=0)),
                ("smotetomek", SMOTETomek(random_state=0)),
                ("classifier", clf),
            ]
        )
        return pipe

    def _create_train_test_split(self):
        columns_to_drop = [
            "ID_REGIONA",
            # "ID_MUNICIP",
            "OUTRO_DES",
            "MORB_DESC",
            "OUT_AMOST",
            "PAC_COCBO",
            "PAC_DSCBO",
            "TOMO_OUT",
            # "TP_IDADE",
            "DELTA_UTI",
            "DOSE_REF",
            "OUT_ANIM",
            # "OBES_IMC",
            "M_AMAMENTA",
            "MAE_VAC",
            # "TP_ANTIVIR",
            # "CO_MUN_NOT",
            # "CO_REGIONA",
            "SURTO_SG",
            "RAIOX_OUT",
            "FNT_IN_COV",
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
            # "ID_MUNICIP",
            "OUTRO_DES",
            "MORB_DESC",
            "OUT_AMOST",
            "PAC_COCBO",
            "PAC_DSCBO",
            "TOMO_OUT",
            # "TP_IDADE",
            "DELTA_UTI",
            "DOSE_REF",
            "OUT_ANIM",
            # "OBES_IMC",
            "M_AMAMENTA",
            "MAE_VAC",
            # "TP_ANTIVIR",
            # "CO_MUN_NOT",
            # "CO_REGIONA",
            "SURTO_SG",
            "RAIOX_OUT",
            "FNT_IN_COV",
            "ID",
        ]
        x_test_id = self.test_data["ID"]
        x_test = self.test_data.drop(columns_to_drop_test, axis=1)
        y_pred = pipe.predict(x_test)

        print(x_test.shape)

        submission = pd.DataFrame({"ID": x_test_id, "CLASSI_FIN": y_pred})
        submission.to_csv(file_name, index=False)
        print("Done!")
        print("Done!")
