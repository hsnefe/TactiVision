# TactiVision Project Documentation

This project contains multiple components including Backend API, Vision Engines, Event Handlers, and a Web Frontend.

## Pipeline modes

When you launch `python src/main.py --video <path>` the program asks you to
pick a mode (you can also pass `--mode {normal,field}` to skip the prompt):

```
Choose pipeline mode:
  1) Normal
  2) Use pre-trained field detection model
Enter choice [1/2]:
```

- **Normal (`1`)** — the default pipeline (YOLO tracking, ball, teams,
  possession). Identical behavior to previous releases.
- **Field detection (`2`)** — additionally runs the Roboflow
  [`football-field-detection-f07vi`](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)
  YOLOv8 keypoint model. Its 32 pitch landmarks are used to:
  - build a polygon of the pitch and **filter out** any player / referee
    detections that fall outside it (crowd, dugouts, ad boards), and
  - compute an image -> pitch **homography** that fills in the
    `FieldMapper` for top-down mapping and metric distances.

### Setting up the field detection model

1. Get a Roboflow API key from <https://app.roboflow.com/settings/api>.
2. Export it once (the env var name is configurable via
   `Settings.roboflow_api_key_env`, default `ROBOFLOW_API_KEY`):

   ```powershell
   $env:ROBOFLOW_API_KEY="..."        # PowerShell
   ```

3. Download the weights for offline use:

   ```bash
   python scripts/download_field_model.py
   ```

   This populates `models/field-keypoints.pt`. The runtime detector tries
   that file first, and only falls back to the hosted Roboflow Inference
   API (slower, requires internet) if the file is missing.