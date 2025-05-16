# Human Activity Recognition with Multiple Deep Learning Approaches

Bu proje, akıllı telefonların ivmeölçer ve jiroskop sensörlerinden toplanan veriler üzerinde insan aktivite tanıma (HAR) gerçekleştiren üç farklı derin öğrenme yaklaşımı içerir:

1. **CNN+Attention**: Convolutional Neural Network ve Spatial Attention mekanizmasının birleşimi (Focal Loss)
2. **Transformer+Contrastive**: TS-TCC benzeri Transformer encoder mimarisi ve Supervised Contrastive Learning (Contrastive+CE Loss)
3. **Multi-Branch CNN**: İvmeölçer ve jiroskop verilerini ayrı kollar üzerinden işleyip birleştiren CNN mimarisi (CE Loss)
4. **CNN+LSTM**: 1D CNN ve LSTM katmanlarını birleştirerek zaman serisi verilerini işleyen mimari (CE Loss)

Her dört yaklaşım da 6 farklı aktiviteyi (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) sınıflandırmak için kullanılır ve sonuçları karşılaştırılabilir.

## Veri Seti

Projede [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) kullanılmaktadır. Bu veri seti:
- 30 gönüllüden toplanan veriler
- 6 farklı aktivite (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
- 561 özellik vektörü
- 10299 örnek içerir

## Kurulum

1. Projeyi klonlayın:
```bash
git clone <repository_url>
cd <project_directory>
```

2. (Önerilen) Sanal ortam oluşturun ve aktive edin:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac için
# veya
.\venv\Scripts\activate  # Windows için
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. UCI HAR Dataset'ini indirin ve proje klasörüne çıkartın. Klasör yapısı şöyle olmalıdır:
```
project_root/
├── UCI HAR Dataset/
│   ├── train/
│   │   ├── X_train.txt
│   │   └── y_train.txt
│   └── test/
│       ├── X_test.txt
│       └── y_test.txt
├── model.py
├── train.py
├── dataloader.py
└── main.py
```

## Kullanım

### Tek Yöntem ile Eğitim

Komut satırından belirli bir yöntemi çalıştırmak için:

```bash
# CNN+Attention ile geliştirme modunda eğitim
python main.py --method cnn_attention --dev

# Transformer+Contrastive ile tam eğitim
python main.py --method transformer_contrastive

# Multi-Branch CNN ile tam eğitim
python main.py --method multi_branch_cnn

# CNN+LSTM ile tam eğitim
python main.py --method cnn_lstm
```

### Yöntemleri Karşılaştırma

Tüm yöntemleri çalıştırıp karşılaştırmak için:

```bash
# Geliştirme modunda karşılaştırma (daha hızlı)
python main.py --method compare --dev

# Tam veri setiyle karşılaştırma (daha doğru sonuçlar)
python main.py --method compare
```

Bu komut tüm yöntemleri sırayla eğitir, doğruluk oranlarını karşılaştırır ve bir karşılaştırma grafiği oluşturur.

## Model Mimarileri

### 1. CNN+Attention Modeli
- **CNN**: 3 konvolüsyon katmanı (32, 64, 128 filtre)
- **Attention**: Her konvolüsyon bloğundan sonra spatial attention
- **Dropout**: Kademeli dropout stratejisi (0.3, 0.42, 0.6)
- **Fully Connected**: 512 nöronlu ara katman
- **Çıktı**: 6 sınıf için softmax
- **Loss**: Focal Loss (γ=2.0)

### 2. Transformer+Contrastive Modeli
- **Input Projection**: Giriş verilerini embedding boyutuna dönüştürme
- **Positional Encoding**: Transformer için pozisyon bilgisi
- **Encoder**: 3 encoder katmanı (Multi-head Attention + Feed Forward)
- **Class Token**: Sınıflandırma için özel token
- **Projection Head**: Contrastive learning için projeksiyon
- **Loss**: Supervised Contrastive Loss + Cross Entropy Loss

### 3. Multi-Branch CNN Modeli
- **Dual Branch Architecture**:
  - **Branch 1**: ACC + GYRO (6 kanal) verilerini işleyen CNN kolu
  - **Branch 2**: TOTAL_ACC (3 kanal) verilerini işleyen CNN kolu
- **Her Kolda**: 3 konvolüsyon katmanı + batch normalization + pooling
- **Merge Layer**: İki kolu birleştiren katman
- **Fully Connected**: 512 nöronlu birleştirme katmanı
- **Loss**: Cross Entropy Loss (CE Loss)
- **Input Format**: (batch_size, 9, 128) - İlk 6 kanal acc+gyro, son 3 kanal total_acc

### 4. CNN+LSTM Modeli
- **CNN Frontend**: 1D CNN layers for feature extraction from raw signals
- **LSTM Core**: Bidirectional LSTM to capture temporal dependencies
- **Fully Connected**: Classification layers
- **Loss**: Cross Entropy Loss (CE Loss)
- **Input Format**: (batch_size, 9, 128) - Time series data

## Veri Önişleme

Her model, farklı veri formatlarına ihtiyaç duyar:

1. **CNN+Attention ve Transformer**: (batch_size, 1, 33, 17) şeklinde 2D tensör
2. **Multi-Branch CNN ve CNN+LSTM**: (batch_size, 9, 128) şeklinde 1D tensör
   - İlk 6 kanal (0-5): ivmeölçer ve jiroskop verileri (2×3 eksen)
   - Son 3 kanal (6-8): toplam ivmelenme verileri

## Çıktı Klasör Yapısı

Her eğitim çalışması için ayrı bir klasör oluşturulur:

```
output/
├── run_YYYYMMDD_HHMMSS_[dev/full]_[method]/  # Her çalıştırma için benzersiz klasör
│   ├── models/                                # Eğitilmiş model dosyaları
│   │   └── method_model_[dev/full].pth        # Model durumu ve parametreleri
│   ├── plots/                                 # Görselleştirmeler
│   │   ├── loss_plot_[method].png             # Eğitim ve validasyon loss grafikleri
│   │   ├── accuracy_plot_[method].png         # Test accuracy grafiği
│   │   ├── confusion_matrix_[method].png      # Confusion matrix görselleştirmesi
│   │   ├── classification_report_[method].txt  # Detaylı sınıflandırma raporu
│   │   └── metrics_[method].csv               # Detaylı metrikler
│   └── logs/                                  # Eğitim logları
│       └── training_log.txt                   # Her epoch için detaylı bilgiler
│
└── comparison_YYYYMMDD_HHMMSS/                # Karşılaştırma çalışması klasörü
    ├── method_comparison.png                  # Tüm yöntemlerin karşılaştırma grafiği 
    └── comparison_results.txt                 # Karşılaştırma sonuçları ve en iyi yöntem
```

## Geliştirme Notları

- Geliştirme modunda veri setinin %5'i kullanılır
- Epoch sayısı geliştirme modunda 10, tam eğitimde 50'dir
- Early stopping: Dev modunda 2 epoch, full modda 3 epoch
- Her çalışma aşağıdaki mekanizmaları içerir:
  - Learning rate scheduling (ReduceLROnPlateau veya OneCycleLR)
  - Weight decay (L2 regularization)
  - Dropout ayarlama
  - Stabilite optimizasyonları
  - Gradient clipping

## Performans Değerlendirme

Her yöntem için aşağıdaki metrikler hesaplanır ve karşılaştırılır:
- Accuracy (%)
- Confusion Matrix
- Precision, Recall, F1-Score (sınıf bazında)
- Eğitim/validasyon loss eğrileri

## Yöntem Karşılaştırması

Yöntemleri karşılaştırmak için oluşturulan çıktılar şunlardır:
- Bar grafiği: Her yöntemin doğruluk oranlarını görsel olarak karşılaştırma
- Metin raporu: Her yöntemin doğruluk oranları ve en iyi performans gösteren yöntem
- Ayrıntılı performans metrikleri: Her yöntem için confusion matrix ve sınıflandırma raporları

## Katkıda Bulunma

Yeni özellikler eklemek, hataları düzeltmek veya modelleri iyileştirmek için pull request gönderebilirsiniz.

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına bakın. 