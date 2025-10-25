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

### Python версия (локальное видеопотребление)

```bash
# YOLO сегментация
python src/yolo_segmentation.py

# MODNet сегментация
python src/modnet_seg.py

# Комбинированная YOLO (тело + руки)
python src/yolo_combined.py
```

### Веб-версия (npm)

1. **Установка зависимостей:**
```bash
cd web
npm install
```

2. **Запуск локального сервера:**
```bash
npm start
```

