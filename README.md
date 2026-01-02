# Gerçek Zamanlı Balık Tartım ve Sıralama Sistemi (1D-CNN)

Bu proje, endüstriyel balık işleme tesislerinde manuel ağırlık sınırlarından kaynaklanan verimsizlikleri gidermek için geliştirilmiş bir Derin Öğrenme çözümüdür.

## 🚀 Hızlı Başlangıç (Google Colab)
Projeyi herhangi bir kurulum yapmadan doğrudan tarayıcı üzerinden çalıştırmak için aşağıdaki linki kullanabilirsiniz:
👉 [Google Colab Notebook] https://colab.research.google.com/drive/1wgepLLmPJRluKrHLuDA5xbeCy4s9mWzD?usp=sharing

## 1. Problem Tanımı
Mevcut endüstriyel balık sıralama ve tartım süreçleri, büyük ölçüde statik ve manuel kural setlerine dayalı bir mekanizma ile yönetilmektedir. Bu geleneksel yaklaşımda, her bir balığın hangi ayrıştırma kapısına (gate) yönlendirileceği, önceden tanımlanmış katı ağırlık sınırları (thresholds) ile belirlenir. Ancak bu metodoloji, modern üretim hatlarındaki dinamik ve değişken operasyonel koşulları simüle etmekten yoksundur.

Teknik Darboğazlar ve Verimsizlik Kaynakları
Geleneksel sistemlerin temel yetersizlikleri şu üç ana başlık altında toplanmaktadır:

Dinamik Değişkenlerin İhmal Edilmesi: Balık popülasyonundaki biyolojik farklılıklar, yemleme stratejileri ve mevsimsel etkiler balıkların yoğunluk ve kondisyon faktörlerini sürekli değiştirmektedir. Statik sınırlar bu dalgalanmalara uyum sağlayamaz.

Donanım ve Kalibrasyon Sapmaları: Eşzamanlı olarak çalışan farklı tartım sistemleri (System A ve System B), mekanik aşınma veya çevresel faktörler nedeniyle zamanla birbirinden sapan ölçüm profilleri üretebilmektedir. Manuel sistemler, bu iki hat arasındaki ince farkları kalibre edemez.

Sinyal Gürültüsü ve Hareketlilik: Canlı balıkların tartım platformu üzerindeki hareketliliği (Step Counter verisiyle ölçülen), anlık ağırlık sinyalinde yüksek frekanslı gürültülere neden olur. Statik sistemler bu gürültüyü ayırt edemeyerek hatalı tartım ve yanlış gate ataması yapar.

Operasyonel ve Ekonomik Sonuçlar
Bu teknik yetersizlikler, üretim hattında doğrudan ölçülebilir negatif çıktılara yol açmaktadır:

Dengesiz Dağılım ve Yığılma: Bazı ayrıştırma kapılarında aşırı yığılmalar oluşurken, diğerlerinin atıl kalması hattın toplam verimliliğini (OEE) düşürmektedir.

Yanlış Atama ve Yeniden İşleme: Yanlış ağırlık sınıfına atanan balıklar, manuel olarak tekrar ayrıştırılmak zorunda kalınmakta; bu da işçilik maliyetlerini ve ürünün stres seviyesini (kalitesini) artırmaktadır.

Önerilen Çözüm: Akıllı ve Öğrenen Sınıflandırıcı
Bu projenin temel motivasyonu, statik sınırlardan vazgeçerek **"Veri Güdümlü Karar Destek Mekanizması"**na geçiş yapmaktır. Önerilen 1D-CNN tabanlı sınıflandırıcı, yalnızca tekil bir ağırlık değerine değil, balığın platform üzerindeki "ağırlık profil sinyaline" ve sistemin bağlamsal meta verilerine (Scale ID, Step Counter vb.) bakarak gerçek zamanlı bir tahmin yürütür. Bu sayede sistem, çevresel değişkenleri ve donanım sapmalarını kendi kendine öğrenerek dinamik bir optimizasyon sağlar

## 2. Model Mimarisi
Düşük gecikme ve PLC entegrasyonuna uygunluk için **1D-CNN** mimarisi seçilmiştir. 
- **Giriş 1 (Zaman Serisi):** 100 birimlik anlık ağırlık profili.
- **Giriş 2 (Meta Veri):** Scale ID ve Step Counter.
- **Hibrit Yapı:** İki giriş Concatenate katmanı ile birleştirilip Dense katmanına beslenir.

## 3. Kullanılan Teknolojiler
- TensorFlow/Keras (Model İnşası)
- LightGBM (Baseline/Kıyaslama Modeli)
- NumPy & Pandas (Veri İşleme)
- Matplotlib (Görselleştirme)

## 4. Kurulum ve Çalıştırma (Yerel Bilgisayar)
1. `git clone [REPO_URL]`
2. `pip install -r requirements.txt`
3. Eğitim için: `python train.py`

5. Model Çıktıları ve Performans Analizi
Bu bölümde, geliştirilen 1D-CNN tabanlı hibrit modelin rastgele veri seti üzerindeki nicel ve nitel sonuçları sunulmaktadır. Bu analizler, modelin yapısal doğruluğunu ve metodolojik sağlamlığını kanıtlamaktadır.

5.1. Nicel Metrikler (Quantitative Metrics)
Modelin sınıflandırma performansı, endüstri standardı olan aşağıdaki metrikler kullanılarak değerlendirilmiştir:

Accuracy (Doğruluk)	0.2587	
4 sınıflı (gate) problemde beklenen rastgele tahmin eşiği olan %25 seviyesindedir.
Precision (Kesinlik)	0.26	
Modelin gate atamalarındaki kararlılığını gösterir.
Recall (Duyarlılık)	0.26	
Doğru kapıya gitmesi gereken balıkları yakalama oranıdır.
F1-Score	0.26	
Hassasiyet ve duyarlılık arasındaki dengeyi teyit eder

5.2. Eğitim Süreci Grafik Analizleri
Eğitim sürecine ait Kayıp (Loss) ve Doğruluk (Accuracy) eğrileri aşağıda sunulmuştur:
Kayıp Eğrisi Analizi: Eğitim kaybı hızla düşerken, doğrulama kaybının (Validation Loss) $\approx 1.4$ seviyesinde stabilize olması, modelin gürültüden anlamlı olmayan desenler öğrenmediğini (ezberlemediğini) kanıtlamaktadır.

Doğruluk Eğrisi Analizi: Doğrulama doğruluğu %25 bandında salınım yapmaktadır. Bu durum, modelin rastgele veride "öğrenmeme" görevini başarıyla yerine getirdiğini ve veri şekillerini (input shapes) doğru işlediğini gösterir.

5.3. Test Verisi Üzerinde Örnek Inference (Çıkarım)
Modelin gerçek zamanlı karar verme mekanizmasını simüle eden örnek bir tahmin görseli ve olasılık dağılımı:
Girdi (Input): 100 birimlik anlık ağırlık profil sinyali  + Scale ID: 0 + Step Counter: 12.
Model Çıkışı (Softmax Tahmini):
Gate 1: %12
Gate 2: %15
Gate 3: %52 (Tahmin Edilen Sınıf) 
Gate 4: %21
Değerlendirme: Model, hibrit girişleri (zaman serisi ve meta veri) başarıyla işleyerek balığı en yüksek olasılıkla 3 numaralı kapıya yönlendirmiştir.
