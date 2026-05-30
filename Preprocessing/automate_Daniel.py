"""
automate_Daniel-Dermawansyah.py
Script otomatisasi preprocessing dataset Diabetes Prediction.
Menjalankan seluruh pipeline preprocessing dari data mentah hingga data siap latih.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ─────────────────────────────────────────────
# KONFIGURASI PATH
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(
    BASE_DIR,
    "Dataset_raw",
    "diabetes_prediction_dataset.csv"
)
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes_preprocessing")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "diabetes_preprocessed.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    print(f"[1/6] Memuat dataset dari: {path}")
    df = pd.read_csv(path)
    print(f"      Shape awal: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani nilai kosong (missing values)."""
    print(f"[2/6] Menangani missing values...")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("      Tidak ada missing values ditemukan.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        print(f"      Kolom dengan missing values telah ditangani.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus data duplikat."""
    print(f"[3/6] Menghapus duplikat...")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"      Baris dihapus: {before - after} | Shape setelah: {df.shape}")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani outlier pada kolom numerik kontinu menggunakan metode IQR."""
    print(f"[4/6] Menangani outlier (metode IQR)...")
    continuous_cols = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    for col in continuous_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        removed = before - len(df)
        if removed > 0:
            print(f"      {col}: {removed} baris outlier dihapus (batas [{lower:.2f}, {upper:.2f}])")
    df = df.reset_index(drop=True)
    print(f"      Shape setelah handling outlier: {df.shape}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encoding fitur kategorikal."""
    print(f"[5/6] Encoding fitur kategorikal...")

    # Label Encoding untuk gender (binary + Other → 3 kelas)
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    print(f"      gender → Label Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # One-Hot Encoding untuk smoking_history (multi-kelas)
    df = pd.get_dummies(df, columns=['smoking_history'], drop_first=False, dtype=int)
    print(f"      smoking_history → One-Hot Encoding selesai.")
    print(f"      Kolom setelah encoding: {df.shape[1]} fitur")
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi fitur numerik kontinu menggunakan MinMaxScaler."""
    print(f"[6/6] Normalisasi fitur numerik...")
    scale_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    scale_cols = [c for c in scale_cols if c in df.columns]
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"      Kolom dinormalisasi: {scale_cols}")
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Menyimpan dataset yang sudah diproses."""
    df.to_csv(path, index=False)
    print(f"\n✅ Preprocessing selesai!")
    print(f"   Dataset tersimpan di: {path}")
    print(f"   Shape akhir         : {df.shape}")


def main():
    print("=" * 55)
    print("  Pipeline Preprocessing — Diabetes Prediction Dataset")
    print("  by Daniel Dermawansyah Putra Saragih")
    print("=" * 55)

    df = load_data(INPUT_PATH)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = encode_categorical(df)
    df = normalize_features(df)
    save_data(df, OUTPUT_PATH)

    print("\nPreview 5 baris pertama hasil preprocessing:")
    print(df.head())


if __name__ == "__main__":
    main()
