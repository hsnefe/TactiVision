# TactiVision Web Platformu

Video yükle → **VisionEngine** pas tespiti → **CommentaryAI** (LLM spiker) → **TTS** sesi.

```
Tarayıcı (React/Vite :5173)
      │  POST /api/videos
      ▼
FastAPI Backend (:8000)
      │  1) VisionEngine  → pas event JSONL   (BackendAPI/Services/vision_service.py)
      │  2) CommentaryAI  → coşkulu spiker metni (CommentaryAI/llm_client.py)
      │  3) Edge TTS      → spiker sesi (mp3)   (CommentaryAI/synthesis/voice.py)
      ▼
İş (job) durumu + event'ler + metin + ses  →  arayüzde gösterilir
```

## Mimari

| Katman | Konum | Görev |
|--------|-------|-------|
| Frontend | `src/WebFrontEnd` | React + Vite arayüzü (yükleme, ilerleme, event listesi, ses oynatıcı) |
| API | `src/BackendAPI` | FastAPI; yükleme, iş takibi, orkestrasyon |
| Vision | `src/VisionEngine` | Mevcut YOLO + pas tespiti pipeline'ı (CLI) |
| Spiker | `src/CommentaryAI` | LLM prompt + sağlayıcı + TTS |

İş akışı `BackendAPI/Services/pipeline_service.py` içinde arka planda 3 adımda yürür ve durum
`/api/jobs/{id}` ile yoklanır (polling).

## Kurulum

### 1) Backend (Python 3.10+)
```powershell
cd TactiVision-pass-tracking
python -m pip install -r requirements-api.txt
# (Gerçek pas tespiti için ayrıca: python -m pip install -r requirements.txt)
```

### 2) Frontend (Node 18+)
```powershell
cd src/WebFrontEnd
npm install
```

## Çalıştırma

**Terminal A — backend:**
```powershell
cd TactiVision-pass-tracking/src
python -m uvicorn BackendAPI.app:app --reload --port 8000
```

**Terminal B — frontend:**
```powershell
cd TactiVision-pass-tracking/src/WebFrontEnd
npm run dev
```

Tarayıcı: **http://localhost:5173** (Vite, `/api`'yi otomatik 8000'e proxy'ler.)

## Yapılandırma (.env veya ortam değişkenleri)

`.env.example` dosyasını kopyalayıp doldurun. Önemli anahtarlar:

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `TACTIVISION_MOCK` | `1` | `1`: örnek event (hızlı demo). `0`: gerçek YOLO pas tespiti. |
| `TACTIVISION_LLM` | `gemini` | `gemini` \| `template` |
| `GEMINI_API_KEY` | — | Google AI Studio anahtarı. Boşsa şablona düşer. |
| `GEMINI_API_KEY` | — | Google Gemini anahtarı. |
| `TACTIVISION_TTS_VOICE` | `tr-TR-AhmetNeural` | Edge TTS sesi (alternatif: `tr-TR-EmelNeural`). |
| `TACTIVISION_TTS_RATE` | `+12%` | Spiker tempo/hızı. |

PowerShell'de geçici örnek:
```powershell
$env:TACTIVISION_LLM = "gemini"; $env:GEMINI_API_KEY = "AIza..."
```

## Gerçek pas tespitine geçiş

1. `python -m pip install -r requirements.txt` (opencv/ultralytics/torch).
2. YOLO ağırlık dosyasını sağla (`yolo11n.pt` veya `TACTIVISION_YOLO_MODEL`).
3. `TACTIVISION_MOCK=0` ayarla.

Backend yüklenen videoyu `src/main.py --pass-detection` ile çalıştırır, üretilen
`*_passes.jsonl` okunur. Pipeline hata verir/eksik bağımlılık olursa otomatik olarak
mock'a düşer (demo bozulmaz).

## API uçları

| Metot | Uç | Açıklama |
|-------|----|----------|
| `POST` | `/api/videos` | Video yükle, işi başlat → `{job_id, stage, ...}` |
| `GET` | `/api/jobs/{id}` | İş durumu (stage, progress, message) |
| `GET` | `/api/jobs/{id}/events` | Tespit edilen pas event'leri |
| `GET` | `/api/jobs/{id}/commentary` | Spiker metni + ses URL'i |
| `GET` | `/api/jobs/{id}/audio` | Spiker sesi (mp3) |
| `GET` | `/api/health` | Aktif sağlayıcı/mod bilgisi |

## Hızlı test (sunucusuz)
```powershell
cd TactiVision-pass-tracking
python scripts/smoke_api.py
```
