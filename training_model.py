import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Baca dataset
df = pd.read_csv("xAPI-Edu-Data.csv")

# Buat target kolom 'Graduated' dari 'Class'
if 'Graduated' not in df.columns:
    if 'Class' in df.columns:
        # Anggap 'High' = lulus tepat waktu
        df['Graduated'] = df['Class'].apply(lambda x: 1 if x == 'H' or x == 'High' else 0)
        df = df.drop(columns=['Class'])
    else:
        raise ValueError("Kolom 'Class' tidak ditemukan. Tidak bisa membuat kolom 'Graduated'.")

# Pisahkan kolom target
target_column = 'Graduated'
X = df.drop(columns=[target_column])
y = df[target_column]

# Deteksi kolom kategorikal
categorical_columns = X.select_dtypes(include='object').columns

# Encode kolom kategorikal
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = X[col].fillna('Unknown')
    le.fit(X[col].tolist() + ['Unknown'])  # Tambahkan Unknown sebagai label aman
    X[col] = le.transform(X[col])
    encoders[col] = le

# Bagi data train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Simpan model dan encoder
joblib.dump(model, 'model_rf.pkl')
joblib.dump(encoders, 'label_encoder.pkl')

print("Model dan encoder berhasil disimpan.")
