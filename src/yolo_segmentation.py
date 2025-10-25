"""
Простая программа для запуска сегментации с веб-камеры используя YOLO модель.
Без diff алгоритма, без рамок - только сегментация.
"""

import cv2
from ultralytics import YOLO
import numpy as np
import time


def run_simple_segmentation(model_path='models/yolo11n-human-seg.pt', camera_id=0):
    """
    Простая сегментация с веб-камеры
    
    Args:
        model_path: путь к обученной модели YOLO
        camera_id: ID веб-камеры
    """
    print(f"Загрузка модели из {model_path}...")
    model = YOLO(model_path)
    print("Модель загружена успешно!")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Ошибка: не удалось открыть веб-камеру")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Нажмите 'q' для выхода")
    
    fps = 0
    fps_frame_count = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Ошибка: не удалось получить кадр")
            break
        
        results = model(frame, verbose=False)
        
        result_frame = frame.copy()
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            
            for mask in masks:
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_uint8 = (mask_resized * 255).astype(np.uint8)
                
                mask_3ch = cv2.merge([mask_uint8, mask_uint8, mask_uint8])
                mask_normalized = mask_3ch.astype(np.float32) / 255.0
                
                blue_bg = np.full_like(frame, (255, 200, 100), dtype=np.uint8)
                result_frame = (frame.astype(np.float32) * mask_normalized + 
                               blue_bg.astype(np.float32) * (1 - mask_normalized)).astype(np.uint8)
        
        fps_frame_count += 1
        elapsed_time = time.time() - fps_start_time
        
        if elapsed_time > 0:
            fps = fps_frame_count / elapsed_time
        
        cv2.putText(
            result_frame,
            f'FPS: {fps:.2f}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        if elapsed_time >= 1.0:
            fps_frame_count = 0
            fps_start_time = time.time()
        
        cv2.imshow('YOLO Simple Segmentation', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Программа завершена")


if __name__ == "__main__":
    run_simple_segmentation(
        model_path='models/yolo11n-human-seg.pt',
        camera_id=0
    )
