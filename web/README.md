# Webcam Person Segmentation (YOLO11n-seg ONNX)

This adds real-time person segmentation from your webcam and background replacement directly in the browser using onnxruntime-web and your `best.onnx` model.

## How to run

Browsers require HTTPS or localhost with a server for webcam access and to fetch ONNX/WASM files. Open a terminal in this folder and run one of the static servers below, then navigate to the shown URL and open `indexV2.html`.

### Python 3 (Linux/macOS/Windows)
```bash
python3 -m http.server 8000
# Then open: http://localhost:8000/indexV2.html
```

### Node.js (if you have npx)
```bash
npx serve -l 8000 .
# Then open: http://localhost:8000/indexV2.html
```

## Usage

1. Wait for the model status to show "Модель: загружена".
2. Click "Старт вебкамеры" to begin.
3. Optionally choose a background image with the file picker.
4. The segmented foreground (person) will be composited over the selected background in the canvas.
5. Click "Стоп" to stop the camera.

## Notes

- The pipeline assumes an input size of 640x640 for the YOLO11n-seg model. If your ONNX uses a different input size, adjust `INPUT_SIZE` in `seg.js`.
- The code attempts to handle both `[1, N, D]` and `[1, D, N]` output layouts for detections. If your `best.onnx` differs significantly, share the output tensor shapes and we can adapt the parser.
- Performance depends on your device. onnxruntime-web is configured for the WASM backend by default.
- Background image is fit with `cover` semantics to the output canvas.
