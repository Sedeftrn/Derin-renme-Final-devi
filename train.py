import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, GlobalAveragePooling1D, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Varsayımsal Parametreler
N_SAMPLES = 50000  # Toplam balık örneği sayısı
T_STEPS = 100      # Zaman serisi uzunluğu (anlık ağırlık okumaları)
N_META_FEATURES = 5 # Ek özellik sayısı (K)
N_CLASSES = 4      # Çıkış Gate sayısı (C)

# 1. Zaman Serisi Verisi (X_TS): Rastgele gürültülü ağırlık profilleri
# (50000 örnek, 100 zaman adımı, 1 özellik)
X_TS = np.random.uniform(low=50, high=500, size=(N_SAMPLES, T_STEPS, 1)).astype('float32')

# 2. Ek Meta Veri (X_Meta): Rastgele özellik değerleri (Scale ID, Step Counter vb.)
# (50000 örnek, 5 özellik)
X_META = np.random.uniform(low=0, high=10, size=(N_SAMPLES, N_META_FEATURES)).astype('float32')
# Scale ID'yi kategorik (0 veya 1) olarak simüle edelim
X_META[:, 0] = np.random.randint(0, 2, N_SAMPLES)

# 3. Çıkış Etiketleri (Y): Rastgele Gate atamaları (0, 1, 2 veya 3)
Y_LABELS = np.random.randint(0, N_CLASSES, N_SAMPLES)

# One-Hot Encoding: Sınıflandırma için etiketleri one-hot formatına dönüştürme
Y_OHE = to_categorical(Y_LABELS, num_classes=N_CLASSES).astype('float32')

print(f"Zaman Serisi Veri Şekli (X_TS): {X_TS.shape}")
print(f"Meta Veri Şekli (X_META): {X_META.shape}")
print(f"Etiket Şekli (Y_OHE): {Y_OHE.shape}")
# Eğitim (80%) ve Test (20%) olarak ayırma
X_train_ts, X_test_ts, X_train_meta, X_test_meta, Y_train, Y_test = train_test_split(
    X_TS, X_META, Y_OHE, test_size=0.2, random_state=42
    )

# Eğitim kümesini de Eğitim (75%) ve Doğrulama (25%) olarak ayırma (Toplam %60 Eğitim, %20 Doğrulama)
X_train_ts, X_val_ts, X_train_meta, X_val_meta, Y_train, Y_val = train_test_split(X_train_ts, X_train_meta, Y_train, test_size=0.25, random_state=42)

print("-" * 30)
print(f"Eğitim (Training) Örnek Sayısı: {len(Y_train)}")
print(f"Doğrulama (Validation) Örnek Sayısı: {len(Y_val)}")
print(f"Test (Test) Örnek Sayısı: {len(Y_test)}")
# Gerekli kütüphaneleri yeniden içe aktardığınızdan emin olun
# (T_STEPS, N_META_FEATURES, N_CLASSES değişkenlerinin Adım 2'de tanımlı olduğunu varsayıyoruz)
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, GlobalAveragePooling1D, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model

# Adım 4: Model Mimarisi Oluşturma Fonksiyonu
def create_fish_sorter_model(ts_length, meta_features_count, num_classes):
    input_ts = Input(shape=(ts_length, 1), name='TS_Input')
    x = Conv1D(filters=16, kernel_size=5, activation='relu')(input_ts)
    x = MaxPool1D(pool_size=2)(x)
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = MaxPool1D(pool_size=2)(x)
    f_cnn = GlobalAveragePooling1D()(x)

    input_meta = Input(shape=(meta_features_count,), name='Meta_Input')
    f_meta = input_meta

    f_combined = Concatenate()([f_cnn, f_meta])

    y = Dense(64, activation='relu')(f_combined)
    y = Dropout(0.2)(y)
    output = Dense(num_classes, activation='softmax', name='Output_Gate')(y)

    model = Model(inputs=[input_ts, input_meta], outputs=output)
    return model

                                                        # Modelin oluşturulması
model = create_fish_sorter_model(T_STEPS, N_META_FEATURES, N_CLASSES)

                                                        # Modelin derlenmesi
model.compile(
          optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy']
             )

                                                                    # Çıktı almak için summary komutu
model.summary()
# Callback'ler:
# 1. Early Stopping: Doğrulama kaybı iyileşmezse eğitimi durdurur.
early_stopping = EarlyStopping(
monitor='val_loss',
patience=10, # 10 epoch bekler
restore_best_weights=True
            )

            # 2. Model Checkpoint: En iyi modeli kaydeder.
model_checkpoint = ModelCheckpoint(
'best_fish_sorter_model.keras',
monitor='val_loss',
save_best_only=True,
verbose=1
  )

                            # Eğitimin Başlatılması
EPOCHS = 50
BATCH_SIZE = 64

history = model.fit(
                                # Model iki giriş bekler: [Zaman Serisi, Meta Veri]
x=[X_train_ts, X_train_meta],
y=Y_train,
validation_data=([X_val_ts, X_val_meta], Y_val),
epochs=EPOCHS,
batch_size=BATCH_SIZE,
callbacks=[early_stopping, model_checkpoint],
verbose=1
)
# Kayıp ve Doğruluk Grafiği
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp Eğrisi')
plt.xlabel('Epok')
plt.ylabel('Kayıp')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk Eğrisi')
plt.xlabel('Epok')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# Test Kümesi Üzerinde Değerlendirme
print("\n--- Test Kümesi Değerlendirmesi ---")
loss, accuracy = model.evaluate(
    x=[X_test_ts, X_test_meta],
        y=Y_test,
            verbose=0
            )
print(f"Test Kaybı: {loss:.4f}")
print(f"Test Doğruluğu: {accuracy:.4f}")
# Gerekli kütüphaneleri yükleyin
try:
    import lightgbm as lgb
except ImportError:
        !pip install lightgbm
        import lightgbm as lgb

from sklearn.metrics import accuracy_score, classification_report
            # X_TS, X_META, Y_LABELS, X_train_meta, Y_train gibi değişkenlerin Adım 2 ve 3'te tanımlanmış olduğunu varsayıyoruz.

            # --- 1. Özellik Mühendisliği (Eğitim Seti) ---
            # Zaman serisinden istatistiksel özellikler çıkarılır
X_train_ts_mean = X_train_ts.mean(axis=1).squeeze()
X_train_ts_std = X_train_ts.std(axis=1).squeeze()
X_train_ts_min = X_train_ts.min(axis=1).squeeze()
X_train_ts_max = X_train_ts.max(axis=1).squeeze()

            # LightGBM için tek bir matris oluşturulur (Meta verilerle birleştirme)
X_train_lgbm = np.column_stack((
    X_train_meta,
    X_train_ts_mean,
    X_train_ts_std,
    X_train_ts_min,
    X_train_ts_max
  ))

                                # --- 2. Özellik Mühendisliği (Test Seti) ---
X_test_ts_mean = X_test_ts.mean(axis=1).squeeze()
X_test_ts_std = X_test_ts.std(axis=1).squeeze()
X_test_ts_min = X_test_ts.min(axis=1).squeeze()
X_test_ts_max = X_test_ts.max(axis=1).squeeze()

X_test_lgbm = np.column_stack((
    X_test_meta,
    X_test_ts_mean,
    X_test_ts_std,
    X_test_ts_min,
    X_test_ts_max
    ))

                                                    # LightGBM'i eğitmek için etiketleri tek boyutlu tam sayı dizisine (integer array) dönüştürme
Y_train_lgbm = np.argmax(Y_train, axis=1)
Y_test_lgbm = np.argmax(Y_test, axis=1)

print(f"LightGBM Eğitim Özellik Şekli: {X_train_lgbm.shape}")

                                                    # --- 3. LightGBM Model Eğitimi ---
lgbm_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=N_CLASSES,
    metric='multi_logloss',
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
               )
lgbm_model.fit(X_train_lgbm, Y_train_lgbm)
                                              # --- 4. Değerlendirme ---
Y_pred_lgbm = lgbm_model.predict(X_test_lgbm)
lgbm_accuracy = accuracy_score(Y_test_lgbm, Y_pred_lgbm)
print("\n--- LightGBM Kıyaslama Sonuçları (Rastgele Veri) ---")
print(f"LightGBM Test Doğruluğu: {lgbm_accuracy:.4f}")
print("Sınıflandırma Raporu:\n", classification_report(Y_test_lgbm, Y_pred_lgbm))
