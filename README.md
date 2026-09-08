# ⚽ TactiVision — Yapay Zeka Destekli Futbol Maç Spikeri ve Görsel Analiz Platformu

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-FF6F00?logo=ultralytics)
![Gemini AI](https://img.shields.io/badge/Gemini_AI-2.5_Flash-8E44AD?logo=google)
![Edge TTS](https://img.shields.io/badge/TTS-Edge_TTS-0078D4)
![License](https://img.shields.io/badge/Lisans-T%C3%BCm_Haklar%C4%B1_Sakl%C4%B1d%C4%B1r-red)

**TactiVision**, ham futbol maçı yayın videolarını bilgisayarlı görü (Computer Vision) ve üretken yapay zeka (Generative AI) teknolojilerini birleştirerek analiz eden, maçtaki olaylara (paslar, zilyetlik, aksiyonlar) duyarlı coşkulu bir **Türkçe yapay zeka maç spikeri** oluşturan ve bunu gerçekçi insan sesiyle videoya gömen uçtan uca bir analiz platformudur.

---

## 💡 Proje Hakkında

Geleneksel maç analizleri ve spiker anlatımları insan gücüne dayalı ve zaman alıcı süreçlerdir. **TactiVision**, bir futbol maçı videosu yüklendiğinde otomatik olarak:
1. Oyuncuları, hakemleri ve topu tespit eder.
2. Form renklerine göre takımları kümeleyip pas aksiyonlarını ve topa sahip olma oranlarını hesaplar.
3. Tespit edilen maç event'lerini (olaylarını) **Google Gemini LLM** modeline besleyerek coşkulu, dinamik bir spiker anlatım metni yazdırır.
4. Bu anlatımı **Microsoft Edge TTS** ile doğal Türkçe insan sesine dönüştürür.
5. Son aşamada ses ile videoyu **FFmpeg** kullanarak senkronize bir şekilde birleştirip izlenmeye hazır spikerli video üretir.

---

## 🔄 Sistem Akış Şeması

```
                                  +-----------------------+
                                  |   Kullanıcı Videosu   |
                                  +-----------+-----------+
                                              |
                                              v
+-----------------------------------------------------------------------------------+
|                              FastAPI Backend (:8000)                               |
|                                                                                   |
|  [Adım 1: VisionEngine]   -->  YOLOv11 + ByteTrack + Form Kümeleme + Pas Tespiti    |
|                                └─> Üretilen: Pas & Zilyetlik Event'leri (JSONL)   |
|                                                                                   |
|  [Adım 2: CommentaryAI]   -->  Gemini 2.5 Flash LLM                               |
|                                └─> Üretilen: Coşkulu Türkçe Spiker Anlatım Metni  |
|                                                                                   |
|  [Adım 3: Edge TTS]       -->  Microsoft Neural Voice (tr-TR-AhmetNeural)         |
|                                └─> Üretilen: Spiker Ses Dosyası (MP3)              |
|                                                                                   |
|  [Adım 4: FFmpeg Merge]   -->  Ses + Video Birleştirme                             |
|                                └─> Üretilen: Spikerli Final Video (MP4)            |
+---------------------------------------------+-------------------------------------+
                                              |
                                              v
                                 +-------------------------+
                                 |  React Web UI (:5173 /  |
                                 |   http://localhost:8000)|
                                 +-------------------------+
```

---

## ✨ Öne Çıkan Özellikler

- 🎯 **Gelişmiş Görsel Analiz (VisionEngine)**
  - **Nesne Tespiti & Takip**: YOLOv11 modeli ve ByteTrack algoritması ile oyuncu, hakem ve top takibi.
  - **ROI Recovery & Ball Extrapolator**: Görüş alanından çıkan oyuncu/top takibini kayıpsız sürdürme.
  - **Otomatik Takım Ayrımı**: Oyuncu formalarındaki dominant renkleri K-Means kümeleme yöntemiyle analiz ederek A Takımı, B Takımı ve Hakem ayrımı yapabilme.
  - **Pas Durum Makinesi (Pass Detector)**: Oyuncular arası top transferini, pas mesafesini, zilyetlik değişimini ve pas serilerini milisaniye hassasiyetinde tespit etme.

- 🎙️ **Yapay Zeka Maç Spikeri (CommentaryAI)**
  - **Gemini 2.5 Flash Entegrasyonu**: Maç olaylarının zaman damgası ve oyuncu/takım bilgisiyle LLM prompt'una dönüştürülmesi.
  - **Dinamik Anlatım Bütçesi**: Anlatım metninin video süresine tam sığması için hedef süre optimizasyonu.
  - **Şablon Fallback Sistemi**: API anahtarı olmaması veya kota aşımı durumunda kesintisiz çalışmayı sağlayan kural tabanlı anlatım jeneratörü.

- 🔊 **Gerçekçi Ses Sentezi ve Birleştirme (TTS & Multiplexing)**
  - **Türkçe Nöral Sesler**: Microsoft Edge TTS altyapısı ile `tr-TR-AhmetNeural` (Erkek) veya `tr-TR-EmelNeural` (Kadın) sesleri.
  - **Hız ve Tempo Ayarı**: Spiker heyecanını yansıtacak özelleştirilebilir konuşma hızı (`+12%`).
  - **FFmpeg Multiplexing**: Üretilen MP3 sesini video akışına pürüzsüz biçimde gömme.

- 💻 **Modern Web Arayüzü (WebFrontEnd)**
  - React + Vite tabanlı hızlı ve tepkisel arayüz.
  - Sürükle-bırak video yükleme, canlı aşama takibi (Progress Bar & Stages), zaman çizelgeli pas event listesi ve entegre video/ses oynatıcı.

- ⚡ **Akıllı Mock Modu**
  - Ekstra GPU veya PyTorch/YOLO kütüphanelerine ihtiyaç duymadan sistemi saniyeler içinde test etmeyi sağlayan **Mock Pipeline** seçeneği (`TACTIVISION_MOCK=1`).

---

## 📁 Proje Dizin Yapısı

```
TactiVision/
├── docs/                      # Mimari ve Web Dokümantasyonları (WEB.md vb.)
├── scripts/                   # Test ve benchmark senaryoları (smoke_api.py, HF clip araçları)
├── src/
│   ├── BackendAPI/            # FastAPI Rest API Sunucusu
│   │   ├── Controllers/       # API Uç Nokta Yöneticileri (Video & Job Kontrolörleri)
│   │   ├── Infrastructure/    # Ortam yapılandırması ve Job Deposu (JobStore)
│   │   ├── Models/            # Pydantic Veri Modelleri ve Enum'lar
│   │   └── Services/          # Pipeline Orkestrasyonu, Vision & Video Servisleri
│   ├── CommentaryAI/          # LLM Spiker ve Ses Sentez Modülü
│   │   ├── prompts/           # LLM Sistem Prompt Şablonları
│   │   ├── synthesis/         # Edge TTS ve Ses Üretim Kütüphanesi
│   │   └── llm_client.py      # Google Gemini API İstemcisi
│   ├── VisionEngine/          # Görsel Analiz ve Pas Tespiti Çekirdeği
│   │   ├── clustering/        # Takım Forması Renk Kümeleme (K-Means)
│   │   ├── detection/         # YOLOv11 Model Yükleme ve Çıkarım (Inference)
│   │   ├── pipeline/          # Video İşleme ve Kare Analiz Akışı
│   │   ├── state_machine/     # Pas ve Zilyetlik Durum Makinesi
│   │   └── tracking/          # ByteTrack ve Nesne Takip Algoritmaları
│   ├── WebFrontEnd/           # React + Vite Kullanıcı Arayüzü
│   │   ├── components/        # UI Bileşenleri (Upload, Progress, CommentaryViewer)
│   │   └── services/          # Frontend API İstemcisi
│   └── main.py                # Görsel Analiz Çekirdeği CLI Giriş Noktası
├── .env.example               # Örnek Konfigürasyon Dosyası
├── pyproject.toml             # Python Paket Yapılandırması
├── requirements.txt           # Görsel Analiz (YOLO/PyTorch/OpenCV) Bağımlılıkları
├── requirements-api.txt       # Hafif API & Web (FastAPI/Edge-TTS) Bağımlılıkları
├── run_server.py              # Cross-Platform Sunucu Başlatıcı
└── start.ps1                  # Windows PowerShell Tek-Komut Başlatıcı
```

---

## 🚀 Hızlı Başlangıç

### Ön Gereksinimler
- **Python**: 3.10 veya üzeri
- **Node.js**: 18.0 veya üzeri (Frontend için)
- **FFmpeg**: Sisteminizde yüklü olmalı veya `imageio-ffmpeg` Python paketi kurulmalıdır.

---

### 🟢 1. Yöntem: Tek Komutla Başlatma (Windows PowerShell — Önerilen)

Projeyi tüm bağımlılıklarını otomatik kurup derleyerek tek hamlede çalıştırmak için:

```powershell
.\start.ps1
```

Bu komut:
1. Python API bağımlılıklarını otomatik kurar.
2. React frontend uygulamasını ilk çalıştırmada derler (`dist/`).
3. Web sunucusunu **http://localhost:8000** adresinde başlatır.

---

### 🟡 2. Yöntem: Python Başlatıcı ile Çalıştırma

Tüm platformlarda (Windows / macOS / Linux) doğrudan Python script'i ile başlatabilirsiniz:

```bash
python run_server.py
```

Sunucu başladığında otomatik olarak varsayılan tarayıcınızda **http://localhost:8000** açılacaktır.

---

### 🔵 3. Yöntem: Geliştirici Modu (Dev Mode — İki Terminal)

Frontend veya backend üzerinde kod geliştirirken canlı yeniden yükleme (hot-reload) için iki ayrı terminal kullanabilirsiniz:

**Terminal 1 — Backend (FastAPI):**
```bash
cd src
python -m uvicorn BackendAPI.app:app --reload --port 8000
```

**Terminal 2 — Frontend (React + Vite):**
```bash
cd src/WebFrontEnd
npm install
npm run dev
```

Geliştirici arayüzü **http://localhost:5173** adresinde açılır (Vite, API isteklerini otomatik 8000 portuna yönlendirir).

---

## ⚙️ Yapılandırma ve Ortam Değişkenleri

Proje kök dizininde `.env.example` dosyasını kopyalayarak bir `.env` dosyası oluşturun:

```bash
cp .env.example .env
```

### Konfigürasyon Seçenekleri

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `TACTIVISION_MOCK` | `0` | `1`: Örnek event üretir (Hızlı demo). `0`: Gerçek YOLO pas tespiti çalıştırır. |
| `TACTIVISION_ALLOW_MOCK_FALLBACK` | `0` | `1`: Gerçek analiz hata verirse otomatik Mock moduna düşer. |
| `TACTIVISION_YOLO_MODEL` | `yolo11n.pt` | Kullanılacak YOLO model ağırlık dosyası. |
| `TACTIVISION_LLM` | `gemini` | Spiker metni sağlayıcısı (`gemini` veya `template`). |
| `GEMINI_API_KEY` | — | Google AI Studio API Anahtarı ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)). |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Anlatım metni üretecek Gemini modeli. |
| `TACTIVISION_TTS_VOICE` | `tr-TR-AhmetNeural` | Spiker Türkçe sesi (`tr-TR-AhmetNeural`: Erkek, `tr-TR-EmelNeural`: Kadın). |
| `TACTIVISION_TTS_RATE` | `+12%` | Spiker konuşma temposu / hızı. |

---

## 🌐 API Uç Noktaları (REST API Reference)

FastAPI arka plan servislerinin sunduğu temel uç noktalar:

| Metot | Uç Nokta (Endpoint) | Açıklama |
|-------|--------------------|----------|
| `POST` | `/api/videos` | Video dosyasını yükler ve analiz işini başlatır. |
| `GET` | `/api/jobs/{job_id}` | İşin güncel durumunu, ilerleme yüzdesini (`progress`) ve aşamasını getirir. |
| `GET` | `/api/jobs/{job_id}/events` | Tespit edilen pas ve zilyetlik olaylarının (events) listesini döner. |
| `GET` | `/api/jobs/{job_id}/commentary` | Üretilen spiker metnini ve ses URL'ini döner. |
| `GET` | `/api/jobs/{job_id}/audio` | Üretilen spiker MP3 ses dosyasını indirir / akış olarak sunar. |
| `GET` | `/api/health` | Sistem sağlık durumunu ve aktif LLM/TTS konfigürasyonunu sorgular. |

> Swagger API Dokümantasyonuna sunucu çalışırken **http://localhost:8000/docs** adresinden erişebilirsiniz.

---

## 🔬 Gerçek Pas Tespiti ve YOLO Kullanımı (Full CV Pipeline)

Gerçek Bilgisayarlı Görü (Computer Vision) pipeline'ını çalıştırmak için PyTorch ve OpenCV bağımlılıklarının kurulu olması gerekir:

1. **Görsel Analiz Bağımlılıklarını Kurun:**
   ```bash
   pip install -r requirements.txt
   ```
2. **YOLO Ağırlık Dosyasını Hazırlayın:**
   - Varsayılan `yolo11n.pt` dosyası otomatik indirilir veya custom eğitilmiş model yolu `.env` içinde `TACTIVISION_YOLO_MODEL` olarak ayarlanır.
3. **Mock Modunu Kapatın:**
   `.env` dosyasında `TACTIVISION_MOCK=0` ayarlayın.

---

## 💻 CLI (Komut Satırı) Kullanımı

Sadece görsel analiz motorunu (VisionEngine) doğrudan komut satırından tek bir video üzerinde çalıştırmak için:

```bash
python src/main.py --video data/input_match.mp4 --output outputs/analyzed_match.mp4 --model yolo11n.pt
```

### Öne Çıkan CLI Parametreleri:
- `--video <path>`: Analiz edilecek video dosyasının yolu.
- `--output <path>`: İşlenmiş ve görsel efektlerle süslenmiş çıktı video yolu.
- `--conf 0.22`: YOLO nesne tespiti güven eşiği.
- `--imgsz 1280`: Görsel çıkarım çözünürlüğü (Büyük değerler küçük top tespitini iyileştirir).
- `--roi-recovery`: Görüşten çıkan oyuncuların ID'lerini korumaya yönelik gelişmiş takip algoritması.

---

## 🧪 Test ve Doğrulama

API ve entegrasyon testini hızlıca koşturmak için:

```bash
python scripts/smoke_api.py
```

---

## 📄 Lisans

Özel Proje — Tüm Hakları Saklıdır.  
© 2026 TactiVision Team.
