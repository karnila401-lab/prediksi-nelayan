"""Training utilities for the Lhokseumawe fish-catch Random Forest project."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "hasil_kg"
CATEGORICAL_FEATURES = ["alat_tangkap", "musim"]
NUMERIC_FEATURES = [
    "gt",
    "lama_melaut_jam",
    "w_temperature_2m_mean",
    "w_precipitation_sum",
    "w_wind_speed_10m_max",
    "w_cloud_cover_mean",
    "w_relative_humidity_2m_mean",
    "woff_wind_speed_10m_max",
    "woff_precipitation_sum",
    "oc_wave_height_max",
    "oc_wave_period_max",
    "oc_ocean_current_velocity_mean",
    "oco_wave_height_max",
    "oco_ocean_current_velocity_mean",
    "sst_oisst_c",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_LABELS = {
    "gt": "Ukuran kapal (GT)",
    "lama_melaut_jam": "Lama melaut (jam)",
    "alat_tangkap": "Alat tangkap",
    "musim": "Musim",
    "w_temperature_2m_mean": "Suhu udara rata-rata (°C)",
    "w_precipitation_sum": "Curah hujan (mm)",
    "w_wind_speed_10m_max": "Kecepatan angin maksimum (m/s)",
    "w_cloud_cover_mean": "Tutupan awan rata-rata (%)",
    "w_relative_humidity_2m_mean": "Kelembapan rata-rata (%)",
    "woff_wind_speed_10m_max": "Kecepatan angin lepas pantai (m/s)",
    "woff_precipitation_sum": "Curah hujan lepas pantai (mm)",
    "oc_wave_height_max": "Tinggi gelombang maksimum (m)",
    "oc_wave_period_max": "Periode gelombang maksimum (s)",
    "oc_ocean_current_velocity_mean": "Kecepatan arus laut rata-rata (m/s)",
    "oco_wave_height_max": "Tinggi gelombang laut maksimum (m)",
    "oco_ocean_current_velocity_mean": "Kecepatan arus laut lepas pantai (m/s)",
    "sst_oisst_c": "Suhu permukaan laut (°C)",
}


@dataclass
class TrainingResult:
    model: Pipeline
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    train_rows: int
    test_rows: int
    test_start_date: str


def temporal_split(
    dataframe: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date so records from the same day cannot cross the boundary."""
    if "tanggal" not in dataframe.columns:
        raise ValueError("Dataset harus memiliki kolom 'tanggal'.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction harus berada di antara 0 dan 1.")

    dates = pd.to_datetime(dataframe["tanggal"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Kolom 'tanggal' berisi tanggal kosong atau tidak valid.")

    unique_dates = np.sort(dates.unique())
    if len(unique_dates) < 2:
        raise ValueError("Dataset membutuhkan setidaknya dua tanggal berbeda.")

    split_index = min(max(int(len(unique_dates) * (1 - test_fraction)), 1), len(unique_dates) - 1)
    test_start = unique_dates[split_index]
    train = dataframe.loc[dates < test_start].copy()
    test = dataframe.loc[dates >= test_start].copy()
    return train, test


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numerik", numeric_pipeline, NUMERIC_FEATURES),
            ("kategori", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(dataframe: pd.DataFrame) -> TrainingResult:
    required_columns = {"tanggal", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Kolom dataset tidak ditemukan: {', '.join(missing_columns)}")

    target = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    if target.isna().any():
        raise ValueError(f"Kolom '{TARGET_COLUMN}' memiliki nilai kosong atau tidak valid.")
    if (target < 0).any():
        raise ValueError(f"Kolom '{TARGET_COLUMN}' tidak boleh bernilai negatif.")

    train, test = temporal_split(dataframe)
    model = build_pipeline()
    model.fit(train[FEATURE_COLUMNS], target.loc[train.index])

    predictions = model.predict(test[FEATURE_COLUMNS])
    actual = target.loc[test.index]
    metrics = {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "r2": float(r2_score(actual, predictions)),
    }

    preprocessing = model.named_steps["preprocessing"]
    feature_names = preprocessing.get_feature_names_out()
    importances = model.named_steps["random_forest"].feature_importances_
    feature_importance = (
        pd.DataFrame({"fitur": feature_names, "kepentingan": importances})
        .assign(fitur=lambda frame: frame["fitur"].str.replace(r"^(numerik|kategori)__", "", regex=True))
        .sort_values("kepentingan", ascending=False)
        .reset_index(drop=True)
    )

    return TrainingResult(
        model=model,
        metrics=metrics,
        feature_importance=feature_importance,
        train_rows=len(train),
        test_rows=len(test),
        test_start_date=pd.Timestamp(test["tanggal"].min()).strftime("%Y-%m-%d"),
    )
