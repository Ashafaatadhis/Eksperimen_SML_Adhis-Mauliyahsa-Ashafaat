import pandas as pd
import numpy as np
import mlflow
import os
from sklearn.preprocessing import StandardScaler

# --- PERBAIKAN Wajib: Set Tracking URI untuk Runner Linux ---
# Mengatur MLflow agar menulis data tracking ke folder yang dapat diakses di runner
# Ini akan ditimpa oleh environment variable di GitHub Actions (MLFLOW_TRACKING_URI)
# Namun, baris ini menjaga kompatibilitas lokal.

# --- 2. Data Paths ---
# Path data mentah (relatif dari folder preprocessing/)
RAW_DATA_PATH = '../healthcare-dataset-stroke-data_raw.csv'

# PERBAIKAN: Nama file output sesuai struktur yang diminta. HANYA NAMA FILE.
CLEAN_DATA_FILE = "healthcare-dataset-stroke-data_preprocessing.csv"
# Path Artifact di dalam folder preprocessing/ (dibiarkan kosong karena disimpan langsung di CWD)
# ARTIFACT_DIR = "." # Dihapus/disederhanakan untuk menghindari konflik path


# --- 1. Persiapan MLflow ---
mlflow.set_experiment("Stroke_Preprocessing_Pipeline")

# --- 2. Data Loading ---

# PERBAIKAN: Hitung path data mentah secara absolut relatif terhadap lokasi script (__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Gabungkan path script dan path relatif data mentah
absolute_raw_data_path = os.path.join(script_dir, RAW_DATA_PATH)

try:
    # Memastikan script dapat dijalankan dari mana saja
    df = pd.read_csv(absolute_raw_data_path)
except FileNotFoundError:
    # Ini akan terjadi jika data mentah tidak berada di path yang ditentukan
    print(f"Error: Data mentah tidak ditemukan di {absolute_raw_data_path}. Harap periksa path.")
    exit(1)


# --- 3. Lakukan Semua Langkah Preprocessing ---

# Imputasi BMI dengan Median
bmi_median = df['bmi'].median()
df['bmi'] = df['bmi'].fillna(bmi_median)

# Penanganan Gender 'Other' dan Encoding
df = df[df['gender'] != 'Other']
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})

# One-Hot Encoding
df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True, dtype=int)

# Outlier Capping
outlier_cols = ['avg_glucose_level', 'bmi']
for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# --- 4. TAHAP AKHIR PREPROCESSING: SCALING ---
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Hapus kolom ID
df = df.drop('id', axis=1)

# --- 5. Simpan Data Bersih ke Disk ---

# PERBAIKAN: Gunakan os.path.join(script_dir, CLEAN_DATA_FILE) untuk path absolut
# Ini akan menyimpan file di path: /home/runner/.../preprocessing/healthcare-dataset-stroke-data_preprocessing.csv
clean_data_path = os.path.join(script_dir, CLEAN_DATA_FILE)

# Simpan DataFrame bersih ke CSV
df.to_csv(clean_data_path, index=False)


# --- 6. Log Artifact Menggunakan MLflow ---
# Karena script dijalankan dari 'preprocessing/', log_artifact akan menemukan CLEAN_DATA_FILE
with mlflow.start_run(run_name="Dataset_Final_Clean"):
    
    # Log Data Statistik Dasar sebagai informasi
    mlflow.log_param("Total_Rows", df.shape[0])
    mlflow.log_param("Features_Count", df.shape[1])
    
    # Log file CSV yang sudah bersih sebagai Artifact
    mlflow.log_artifact(local_path=clean_data_path, artifact_path="clean_data")
    
    print("\n✅ Preprocessing Selesai dan Data Bersih Berhasil Dicatat di MLflow!")
    print(f"Data Clean disimpan secara lokal di: {clean_data_path}")