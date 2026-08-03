#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PORT = 8000


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _open_browser_later() -> None:
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()


def main() -> None:
    _load_dotenv()
    sys.path.insert(0, str(SRC))

    try:
        import uvicorn
    except ModuleNotFoundError:
        print("\n[!] Gerekli paketler kurulu değil. Bir kez şunu çalıştır:")
        print('    python -m pip install -r requirements-api.txt imageio-ffmpeg\n')
        raise

    print(f"\n  TactiVision basliyor ->  http://localhost:{PORT}\n")
    _open_browser_later()

    from BackendAPI.app import app
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
