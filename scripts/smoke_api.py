"""Backend'i uçtan uca dener (mock vision + template/LLM + TTS)."""
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from BackendAPI.app import app  # noqa: E402

client = TestClient(app)

print("health:", client.get("/api/health").json())

files = {"file": ("mac.mp4", io.BytesIO(b"fake video bytes"), "video/mp4")}
r = client.post("/api/videos", files=files)
print("upload:", r.status_code, r.json())
job_id = r.json()["job_id"]

for _ in range(30):
    j = client.get(f"/api/jobs/{job_id}").json()
    print("  stage:", j["stage"], j["progress"], j["message"])
    if j["stage"] in ("done", "failed"):
        break
    time.sleep(0.5)

ev = client.get(f"/api/jobs/{job_id}/events").json()
print("events:", len(ev["events"]), "ör:", ev["events"][:2])

c = client.get(f"/api/jobs/{job_id}/commentary").json()
print("provider:", c["provider"], "audio_url:", c["audio_url"])
print("commentary:\n", c["text"][:600])

if c["audio_url"]:
    a = client.get(c["audio_url"])
    print("audio bytes:", len(a.content), "content-type:", a.headers.get("content-type"))
