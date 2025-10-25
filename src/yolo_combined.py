"""
Программа для запуска сегментации с веб-камеры используя ДВЕ обученные YOLO модели
с интеграцией diff для определения области движения.
Модель для тела + модель для рук = полная сегментация человека.
Оптимизированная версия для увеличения FPS.
"""

import cv2
import time
from ultralytics import YOLO
import numpy as np


def expand_rect(rect, x, y, w, h):
    """Расширяет прямоугольник, объединяя его с новым"""
    rx, ry, rw, rh = rect
    nx1 = min(rx, x)
    ny1 = min(ry, y)
    nx2 = max(rx + rw, x + w)
    ny2 = max(ry + rh, y + h)
    return (nx1, ny1, nx2 - nx1, ny2 - ny1)


def run_webcam_segmentation_dual_models(
    body_model_path='yolo_best_body.pt',
    arms_model_path='yolo_best_arms.pt',
    camera_id=0
):
    """
    Запускает сегментацию с веб-камеры используя две модели:
    одну для тела, другую для рук
    
    Args:
        body_model_path: путь к модели для сегментации тела
        arms_model_path: путь к модели для сегментации рук
        camera_id: ID веб-камеры (обычно 0 для встроенной камеры)
    """
    # Загрузка моделей
    print(f"Загрузка модели тела из {body_model_path}...")
    model_body = YOLO(body_model_path)
    print("Модель тела загружена успешно!")
    
    print(f"Загрузка модели рук из {arms_model_path}...")
    model_arms = YOLO(arms_model_path)
    print("Модель рук загружена успешно!")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Ошибка: не удалось открыть веб-камеру")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prev_frame = None
    threshold = 20
    max_rect = None
    pending_rect = None
    pending_start = None
    pending_motion = False
    frame_counter = 0
    update_interval = 300
    diff_skip_frames = 2
    max_yolo_size = 416
    fps = 0
    fps_frame_count = 0
    fps_start_time = time.time()
    last_crop_result = None
    last_rect_coords = None
    
    print("Нажмите 'q' для выхода, '+'/'-' для изменения чувствительности")
    print(f"Текущий threshold: {threshold}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Ошибка: не удалось получить кадр")
            break
        
        frame_h, frame_w = frame.shape[:2]
        
        blue_background = np.full_like(frame, (255, 200, 100), dtype=np.uint8)
        result_frame = blue_background.copy()
        
        process_diff = (frame_counter % diff_skip_frames == 0)
        
        if process_diff:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(frame_gray, prev_frame)
                _, diff_thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_TOZERO)
                diff_blur = cv2.GaussianBlur(diff_thresh, (5, 5), 0)
                diff_mask = (diff_blur > 0).astype(np.uint8) * 255
                kernel = np.ones((5, 5), np.uint8)
                diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
                
                contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w * h > 500:
                        if max_rect is None:
                            max_rect = (x, y, w, h)
                        else:
                            new_rect = expand_rect(max_rect, x, y, w, h)
                            nx, ny, nw, nh = new_rect
                            new_w = int(nw * 1.1)
                            new_h = int(nh * 1.1)
                            if new_w <= frame_w and new_h <= frame_h:
                                max_rect = new_rect
                
                if pending_rect is None and (frame_counter % update_interval == 0):
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w * h > 500:
                            pending_rect = (x, y, w, h)
                            pending_start = frame_counter
                            pending_motion = True
                            break
                    else:
                        pending_rect = None
                        pending_start = None
                        pending_motion = False
                
                if pending_rect is not None:
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w * h > 500:
                            new_rect = expand_rect(pending_rect, x, y, w, h)
                            nx, ny, nw, nh = new_rect
                            new_w = int(nw * 1.1)
                            new_h = int(nh * 1.1)
                            if new_w <= frame_w and new_h <= frame_h:
                                pending_rect = new_rect
                            pending_motion = True
                    
                    if pending_start is not None and (frame_counter - pending_start >= update_interval):
                        if pending_motion:
                            max_rect = pending_rect
                            print(f"Рамка обновлена на кадре {frame_counter}")
                        pending_rect = None
                        pending_start = None
                        pending_motion = False
            
            prev_frame = frame_gray.copy()
        
        if max_rect is not None:
            x, y, w, h = max_rect
            
            center_x = x + w // 2
            center_y = y + h // 2
            new_w = int(w * 1.1)
            new_h = int(h * 1.1)
            
            rect_x = center_x - new_w // 2
            rect_y = center_y - new_h // 2
            
            if rect_x < 0:
                rect_x = 0
            if rect_y < 0:
                rect_y = 0
            if rect_x + new_w > frame_w:
                rect_x = frame_w - new_w
            if rect_y + new_h > frame_h:
                rect_y = frame_h - new_h
            rect_x = max(0, rect_x)
            rect_y = max(0, rect_y)
            
            if rect_x >= 0 and rect_y >= 0 and rect_x + new_w <= frame_w and rect_y + new_h <= frame_h and new_w > 0 and new_h > 0:
                crop_frame = frame[rect_y:rect_y+new_h, rect_x:rect_x+new_w].copy()
                
                scale = 1.0
                if max(new_w, new_h) > max_yolo_size:
                    scale = max_yolo_size / max(new_w, new_h)
                    resized_w = int(new_w * scale)
                    resized_h = int(new_h * scale)
                    crop_frame_resized = cv2.resize(crop_frame, (resized_w, resized_h))
                else:
                    crop_frame_resized = crop_frame
                
                results_body = model_body(crop_frame_resized, verbose=False, half=True)
                
                results_arms = model_arms(crop_frame_resized, verbose=False, half=True)
                
                combined_mask = np.zeros((crop_frame_resized.shape[0], crop_frame_resized.shape[1]), dtype=np.uint8)
                
                if results_body[0].masks is not None:
                    masks_body = results_body[0].masks.data.cpu().numpy()
                    
                    mask_areas_body = []
                    for mask in masks_body:
                        mask_resized = cv2.resize(mask, (crop_frame_resized.shape[1], crop_frame_resized.shape[0]))
                        mask_uint8 = (mask_resized * 255).astype(np.uint8)
                        area = np.sum(mask_uint8 > 127)
                        mask_areas_body.append((area, mask_uint8))
                    
                    mask_areas_body.sort(reverse=True, key=lambda x: x[0])
                    for i in range(min(2, len(mask_areas_body))):
                        combined_mask = cv2.bitwise_or(combined_mask, mask_areas_body[i][1])
                
                if results_arms[0].masks is not None:
                    masks_arms = results_arms[0].masks.data.cpu().numpy()
                    
                    for mask in masks_arms:
                        mask_resized = cv2.resize(mask, (crop_frame_resized.shape[1], crop_frame_resized.shape[0]))
                        mask_uint8 = (mask_resized * 255).astype(np.uint8)
                        combined_mask = cv2.bitwise_or(combined_mask, mask_uint8)
                
                kernel = np.ones((3, 3), np.uint8)
                combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
                combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
                
                if scale < 1.0:
                    combined_mask = cv2.resize(combined_mask, (new_w, new_h))
                    crop_frame_resized = cv2.resize(crop_frame_resized, (new_w, new_h))
                
                crop_mask_normalized = combined_mask.astype(np.float32) / 255.0
                crop_mask_3ch = cv2.merge([crop_mask_normalized, crop_mask_normalized, crop_mask_normalized])
                
                crop_blue_bg = np.full_like(crop_frame, (255, 200, 100), dtype=np.uint8)
                crop_result = (crop_frame.astype(np.float32) * crop_mask_3ch + 
                             crop_blue_bg.astype(np.float32) * (1 - crop_mask_3ch)).astype(np.uint8)
                
                last_crop_result = crop_result
                last_rect_coords = (rect_x, rect_y, new_w, new_h)
        
        if last_crop_result is not None and last_rect_coords is not None:
            rect_x, rect_y, new_w, new_h = last_rect_coords
            result_frame[rect_y:rect_y+new_h, rect_x:rect_x+new_w] = last_crop_result
            
            cv2.rectangle(result_frame, (rect_x, rect_y),
                        (rect_x+new_w, rect_y+new_h), (0, 0, 255), 2)
        
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
        
        cv2.putText(
            result_frame,
            'Body + Arms Models',
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        if elapsed_time >= 1.0:
            fps_frame_count = 0
            fps_start_time = time.time()
        
        cv2.imshow('Dual YOLO Segmentation (Body + Arms)', result_frame)
        
        frame_counter += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            threshold = min(255, threshold + 5)
            print(f"Threshold увеличен до: {threshold}")
        elif key == ord('-') or key == ord('_'):
            threshold = max(0, threshold - 5)
            print(f"Threshold уменьшен до: {threshold}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Программа завершена")


if __name__ == "__main__":
    run_webcam_segmentation_dual_models(
        body_model_path='models/yolo_best_body.pt',
        arms_model_path='models/yolo_best_arms.pt',
        camera_id=0
    )
