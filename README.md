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

- `model.py`: CNN ve Attention mimarisi
- `dataloader.py`: Veri yükleme ve ön işleme
- `train.py`: Eğitim döngüsü ve değerlendirme
- `requirements.txt`: Gerekli Python paketleri

## Sonuçlar

Eğitim sonuçları:
- Loss ve accuracy grafikleri `training_results.png` dosyasına kaydedilir
- Eğitilmiş model `har_cnn_attention_model.pth` (tam eğitim) veya `har_cnn_attention_model_dev.pth` (geliştirme modu) olarak kaydedilir

## Geliştirme Notları

- Geliştirme modunda veri setinin %10'u kullanılır
- Epoch sayısı geliştirme modunda 10, tam eğitimde 50'dir
- Model performansı ve eğitim süreci matplotlib grafikleri ile görselleştirilir 