import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, GlobalAveragePooling1D, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model

def create_fish_sorter_model(ts_length=100, meta_features_count=5, num_classes=4):
    """
    1D-CNN tabanlı hibrit sınıflandırıcı mimarisi.
    Ağırlık profili (Zaman Serisi) ve bağlamsal verileri (Meta Veri) birleştirir.
    """
    
    # --- 1. Zaman Serisi Kolu (1D-CNN) ---
    # Giriş: Anlık ağırlık profili (Örn: 100 zaman adımı, 1 kanal)
    input_ts = Input(shape=(ts_length, 1), name='input_ts') [cite: 86]
    
    # Katman 1: Yerel zaman serisi örüntülerini yakalar
    x = Conv1D(filters=16, kernel_size=5, activation='relu', name='conv1d_1')(input_ts) [cite: 80, 88]
    x = MaxPool1D(pool_size=2)(x)
    
    # Katman 2: Daha karmaşık zaman serisi örüntülerini yakalar
    x = Conv1D(filters=32, kernel_size=3, activation='relu', name='conv1d_2')(x) [cite: 80, 88]
    x = MaxPool1D(pool_size=2)(x)
    
    # CNN çıktısını sabit uzunlukta bir özellik vektörüne dönüştürür
    f_cnn = GlobalAveragePooling1D(name='global_avg_pool')(x) [cite: 80, 88]

    # --- 2. Meta Veri Kolu ---
    # Giriş: Scale ID, Step Counter gibi ek özellikler (K adet)
    input_meta = Input(shape=(meta_features_count,), name='input_meta') [cite: 80, 86]
    f_meta = input_meta

    # --- 3. Birleştirme ve Sınıflandırma Başlığı ---
    # Zaman serisi özellikleri ile meta verileri birleştirir
    f_combined = Concatenate(name='concatenate')([f_cnn, f_meta]) [cite: 80, 90]

    # Tam bağlantılı katmanlar (Fully Connected Head)
    y = Dense(64, activation='relu', name='dense_baslik')(f_combined) [cite: 80, 93]
    y = Dropout(0.2)(y) # Aşırı öğrenmeyi engellemek için
    
    # Çıkış: 4 Gate için Softmax olasılık dağılımı
    output = Dense(num_classes, activation='softmax', name='cikis_katmani')(y) [cite: 80, 93]

    # Modeli Giriş/Çıkış noktalarıyla tanımla
    model = Model(inputs=[input_ts, input_meta], outputs=output) [cite: 95]
    
    return model
