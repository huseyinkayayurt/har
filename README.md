# Human Activity Recognition with CNN and Attention

Bu proje, akıllı telefonların ivmeölçer ve jiroskop sensörlerinden toplanan veriler üzerinde insan aktivite tanıma (HAR) gerçekleştiren bir derin öğrenme modeli içerir. Model, CNN (Convolutional Neural Network) ve Attention mekanizmasını birleştirerek 6 farklı aktiviteyi sınıflandırır.

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
└── dataloader.py
```

## Kullanım

### Geliştirme Modu

Hızlı denemeler için geliştirme modunda çalıştırın (veri setinin %10'u kullanılır):
```bash
python train.py
```

### Tam Eğitim

Tüm veri seti üzerinde eğitim için `train.py` dosyasında:
```python
if __name__ == '__main__':
    main(dev_mode=False)
```
şeklinde değiştirin ve çalıştırın.

## Model Mimarisi

- **CNN**: 3 konvolüsyon katmanı (32, 64, 128 filtre)
- **Attention**: Her konvolüsyon bloğundan sonra spatial attention
- **Fully Connected**: 512 nöronlu ara katman
- **Çıktı**: 6 sınıf için softmax
- **Loss**: Focal Loss

## Dosya Yapısı

### Kod Dosyaları
- `model.py`: CNN ve Attention mimarisi
- `dataloader.py`: Veri yükleme ve ön işleme
- `train.py`: Eğitim döngüsü ve değerlendirme
- `requirements.txt`: Gerekli Python paketleri

### Çıktı Klasör Yapısı
```
output/
└── run_YYYYMMDD_HHMMSS_[dev/full]/  # Her çalıştırma için benzersiz klasör
    ├── models/                       # Eğitilmiş model dosyaları
    │   └── model_[dev/full].pth     # Model durumu ve parametreleri
    ├── plots/                       # Görselleştirmeler
    │   ├── loss_plot.png           # Eğitim ve validasyon loss grafikleri
    │   ├── accuracy_plot.png       # Test accuracy grafiği
    │   └── metrics.csv             # Detaylı metrikler
    └── logs/                       # Eğitim logları
        └── training_log.txt        # Her epoch için detaylı bilgiler
```

## Eğitim Çıktıları

Her eğitim çalışması (run) için aşağıdaki çıktılar üretilir:

1. **Model Dosyası** (`models/model_[dev/full].pth`):
   - Model durumu
   - Optimizer durumu
   - Son epoch bilgileri
   - Train/val loss ve accuracy değerleri

2. **Grafikler** (`plots/`):
   - Training ve validation loss karşılaştırması
   - Test accuracy değişimi
   - CSV formatında detaylı metrikler

3. **Loglar** (`logs/training_log.txt`):
   - Her epoch için detaylı bilgiler
   - Overfitting/underfitting durumu
   - Learning rate değişimleri
   - Dropout ayarlamaları

## Geliştirme Notları

- Geliştirme modunda veri setinin %10'u kullanılır
- Epoch sayısı geliştirme modunda 10, tam eğitimde 50'dir
- Early stopping, learning rate scheduling ve dropout ayarlama mekanizmaları içerir
- Her çalıştırma için benzersiz bir output klasörü oluşturulur
- Training ve validation loss karşılaştırması ile overfitting durumu gözlemlenebilir

## Overfitting ve Underfitting Kontrolü

Model, eğitim sırasında aşağıdaki mekanizmalarla overfitting ve underfitting'i kontrol eder:

1. **Early Stopping**: Validation loss belirli bir süre iyileşmezse eğitimi durdurur
2. **Learning Rate Ayarlama**: Öğrenme yavaşladığında learning rate'i düşürür
3. **Dropout Ayarlama**: 
   - Overfitting durumunda dropout oranını artırır
   - Underfitting durumunda dropout oranını azaltır
4. **Model Kaydetme**: En iyi performans gösteren model durumunu kaydeder 