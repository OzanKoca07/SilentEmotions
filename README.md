# SilentEmotions
SilentEmotions, dudak hareketlerinden kelimeleri ve yüz ifadelerinden duyguları aynı anda analiz edebilen çok modlu (multimodal) bir yapay zeka sistemidir. Proje, sessiz videolar üzerinden bireyin hem ne söylediğini hem de nasıl hissettiğini tespit etmeyi amaçlamaktadır.

# 🎭 SilentEmotions

**SilentEmotions**, sessiz video girişlerinden bireylerin hem **ne söylediklerini** hem de **ne hissettiklerini** anlamayı amaçlayan çok modlu (multimodal) bir yapay zeka projesidir. Bu sistem, dudak okuma ve duygu tanıma modellerini birleştirerek sessiz iletişimi analiz edebilir hale getirir.

## 📌 Proje Özeti

Bu proje, özellikle işitme engelli bireyler için sessiz iletişimi kolaylaştırmak, insan-bilgisayar etkileşimini artırmak ve duygu analizinde yeni bir bakış açısı sunmak amacıyla geliştirilmiştir. Ana bileşenleri:
- Dudak okuma (Lip Reading)
- Duygu tanıma (Emotion Recognition)
- Multimodal fusion (özelliklerin birleşimi)
- Mobil uygulama üzerinden gerçek zamanlı tahmin sistemi

---

## 🚀 Özellikler

- 🎬 **Video tabanlı giriş**: Mobil uygulama ile video yakalama ve gönderme.
- 👄 **Lip Reading**: 3D CNN + Transformer Encoder ile kelime tahmini.
- 😊 **Emotion Recognition**: CNN mimarisi ile 7 temel duygu sınıflandırması.
- 🧩 **Fusion Model**: Hadamard çarpımı ve attention fusion ile özellik birleştirme.
- 📱 **Android Entegrasyonu**: Kotlin, CameraX ve Retrofit ile video gönderimi.
- ⚡ **FastAPI Backend**: RESTful API ile model servisleri ve tahmin işlemleri.

---

## 🛠️ Kullanılan Teknolojiler

### Backend
- Python 3.10+
- PyTorch
- FastAPI
- MediaPipe (Face Mesh, Lip Landmark Extraction)
- OpenCV
- Torchvision

### Mobile App
- Kotlin
- CameraX
- Retrofit
- ViewModel Architecture

---

## 🧪 Model Yapısı

- `LipReadingModel`: 3D CNN + Transformer katmanları
- `EmotionModel`: CNN tabanlı duygusal özellik çıkarımı
- `FusionModel`: İki modelin çıktılarının Hadamard çarpımı ve attention fusion ile birleştirilmesi

---

## 📲 Mobil Uygulama

Mobil uygulama, kullanıcının yüzünü kamera ile kaydedip arka planda çalışan FastAPI backend’ine gönderir. Backend bu videoyu işleyerek:
- Dudak hareketlerinden kelime tahmini yapar.
- Yüz ifadelerinden duygu analizini çıkarır.

Sonuçlar kullanıcı arayüzünde gösterilir.

---

## 🧰 Kurulum

### 1. Backend

```bash
git clone https://github.com/kullanici-adi/SilentEmotions.git
cd SilentEmotions/backend
pip install -r requirements.txt
uvicorn main:app --reload
