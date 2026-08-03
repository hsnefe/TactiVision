# ⚽ TactiVision — AI Maç Spikeri

Futbol maçı videosu yükle → yapay zeka **oyuncu ve pas tespiti** yapsın → **coşkulu bir spiker** gibi maçı anlatsın → bu anlatım **gerçek bir insan sesine** dönüşüp **videonun üzerine eklensin**.

```
Web (React)  →  FastAPI  →  VisionEngine (YOLO + pas tespiti)
                        →  CommentaryAI (Gemini LLM spiker metni)
                        →  Edge TTS (Türkçe spiker sesi)
                        →  ffmpeg (sesi videoya gömme)
                        →  Spikerli video + anlatım + event listesi
```

## Bileşenler

| Klasör | Açıklama |
|--------|----------|
| `src/VisionEngine` | YOLO tabanlı oyuncu/top takibi ve pas tespiti (mevcut çekirdek). |
| `src/BackendAPI` | FastAPI: yükleme, iş takibi, orkestrasyon, video birleştirme. |
| `src/CommentaryAI` | LLM spiker promptu + Gemini sağlayıcı + TTS. |
| `src/WebFrontEnd` | React + Vite arayüzü. |

## Hızlı Başlangıç (tek komut — Windows)

```powershell
cd TactiVision-pass-tracking
.\start.ps1
```

Sonra tarayıcıda **http://localhost:8000** adresini aç. Hepsi bu.

> `start.ps1` ilk seferde Python bağımlılıklarını kurar, frontend'i derler ve sunucuyu açar.

### Geliştirici modu (canlı yeniden yükleme, iki sunucu)

```powershell
# Terminal A
cd src
python -m uvicorn BackendAPI.app:app --reload --port 8000
# Terminal B
cd src/WebFrontEnd
npm install
npm run dev    # http://localhost:5173
```

## Yapılandırma

`.env.example` dosyasını kopyalayıp doldur. Öne çıkanlar:

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `TACTIVISION_MOCK` | `1` | `1`: örnek event (demo). `0`: gerçek YOLO pas tespiti. |
| `TACTIVISION_LLM` | `gemini` | `gemini` \| `template`. |
| `GEMINI_API_KEY` | — | Google AI Studio anahtarı. Girilince gerçek coşkulu anlatım üretilir. |
| `TACTIVISION_TTS_VOICE` | `tr-TR-AhmetNeural` | Spiker sesi. |

Detaylı mimari ve API uçları: [`docs/WEB.md`](docs/WEB.md).

## Gerçek pas tespitine geçiş

1. `python -m pip install -r requirements.txt` (opencv/ultralytics/torch).
2. YOLO ağırlığı sağla (`yolo11n.pt`).
3. `TACTIVISION_MOCK=0` yap.

## Lisans

Özel proje — tüm hakları saklıdır.
