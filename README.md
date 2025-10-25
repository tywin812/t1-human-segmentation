# Инструкция по запуску проекта

## Предварительные требования
- Python 3.12 или выше
- Вебкамера (для использования видеопотока)

### На Linux/macOS:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

uv sync
source .venv/bin/activate

#Для MODNet
git clone https://github.com/ZHKKKang/MODNet.git
```

## Запуск программы

```bash
python src/yolo_segmentation.py

#Для MODNet
python src/modnet_seg.py

#Для комбинированных YOLO
python src/yolo_combined.py
```
