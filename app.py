from pathlib import Path

import pandas as pd
import streamlit as st

from model import FEATURE_COLUMNS, FEATURE_LABELS, NUMERIC_FEATURES, train_model


DATA_PATH = Path(__file__).parent / "dataset_final.csv"

st.set_page_config(
    page_title="Prediksi Hasil Tangkapan Ikan",
    page_icon="🐟",
    layout="wide",
)


@st.cache_data
def load_data(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path)


@st.cache_resource
def get_training_result(path: str, modified_time: float):
    del modified_time
    return train_model(pd.read_csv(path))


st.title("Prediksi Hasil Tangkapan Ikan Nelayan Lhokseumawe")
st.caption("Random Forest berdasarkan data cuaca, gelombang, dan arus laut")

if not DATA_PATH.exists():
    st.error(f"Dataset tidak ditemukan: {DATA_PATH}")
    st.stop()

try:
    modified_time = DATA_PATH.stat().st_mtime
    data = load_data(str(DATA_PATH), modified_time)
    result = get_training_result(str(DATA_PATH), modified_time)
except (OSError, ValueError, KeyError) as error:
    st.error(f"Data atau model gagal disiapkan: {error}")
    st.stop()

metric_columns = st.columns(4)
metric_columns[0].metric("Jumlah data", f"{len(data):,}")
metric_columns[1].metric("MAE", f"{result.metrics['mae']:.1f} kg")
metric_columns[2].metric("RMSE", f"{result.metrics['rmse']:.1f} kg")
metric_columns[3].metric("R²", f"{result.metrics['r2']:.3f}")

st.caption(
    f"Model dilatih pada {result.train_rows:,} data. "
    f"Evaluasi menggunakan {result.test_rows:,} data dari {result.test_start_date} "
    "hingga tanggal terakhir dataset."
)

st.subheader("Coba prediksi")
st.write(
    "Masukkan kondisi kapal, cuaca, dan laut. Prediksi menggunakan satuan kilogram "
    "per kegiatan melaut."
)

with st.form("prediction_form"):
    form_columns = st.columns(3)
    input_values = {}

    for index, feature in enumerate(FEATURE_COLUMNS):
        column = form_columns[index % len(form_columns)]
        label = FEATURE_LABELS[feature]
        if feature in NUMERIC_FEATURES:
            values = pd.to_numeric(data[feature], errors="coerce").dropna()
            if values.empty:
                st.error(f"Kolom '{feature}' tidak memiliki nilai numerik.")
                st.stop()
            minimum = float(values.min())
            maximum = float(values.max())
            default = float(values.median())
            if minimum == maximum:
                input_values[feature] = column.number_input(
                    label, value=default, disabled=True
                )
            else:
                input_values[feature] = column.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(default, minimum), maximum),
                    step=0.1,
                    format="%.2f",
                )
        else:
            choices = sorted(data[feature].dropna().astype(str).unique().tolist())
            if not choices:
                st.error(f"Kolom '{feature}' tidak memiliki pilihan.")
                st.stop()
            input_values[feature] = column.selectbox(label, choices)

    submitted = st.form_submit_button("Prediksi hasil tangkapan", type="primary")

if submitted:
    input_frame = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
    prediction = float(result.model.predict(input_frame)[0])
    st.success(f"Perkiraan hasil tangkapan: **{prediction:,.1f} kg**")

with st.expander("Faktor yang paling berpengaruh pada model"):
    top_features = result.feature_importance.head(10).copy()
    top_features["nama_fitur"] = top_features["fitur"].map(FEATURE_LABELS).fillna(
        top_features["fitur"]
    )
    st.bar_chart(top_features.set_index("nama_fitur")["kepentingan"])

with st.expander("Tentang data dan evaluasi"):
    st.write(
        "Target prediksi adalah `hasil_kg`. Data dibagi berdasarkan tanggal: "
        "data tanggal yang lebih baru ditahan sebagai data uji, sehingga baris "
        "dari tanggal yang sama tidak tersebar ke data latih dan data uji."
    )
    st.write(
        "MAE dan RMSE dinyatakan dalam kilogram. Nilai R² menunjukkan kecocokan "
        "model pada data uji dan dapat bernilai negatif jika model lebih buruk "
        "daripada prediksi rata-rata."
    )
