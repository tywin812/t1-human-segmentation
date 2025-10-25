"""
Программа для запуска сегментации с веб-камеры используя обученную MODNet модель
с интеграцией diff для определения области движения.
MODNet сегментирует только область, определённую diff алгоритмом.
Оптимизированная версия для увеличения FPS.
"""

import cv2
import time
import torch
import sys
import os
import numpy as np

modnet_imported = False
try:
    from MODNet.src.models.modnet import MODNet
    modnet_imported = True
except ImportError:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MODNet', 'src'))
        from models.modnet import MODNet
        modnet_imported = True
    except ImportError:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MODNet-master', 'src'))
            from models.modnet import MODNet
            modnet_imported = True
        except ImportError:
            print("Ошибка: не удалось найти MODNet!")
            print("Убедитесь, что репозиторий MODNet находится в одной из папок:")
            print("  - ../MODNet/src")
            print("  - ../MODNet-master/src")
            sys.exit(1)

def expand_rect(rect, x, y, w, h):
    """Расширяет прямоугольник, объединяя его с новым"""
    rx, ry, rw, rh = rect
    nx1 = min(rx, x)
    ny1 = min(ry, y)
    nx2 = max(rx + rw, x + w)
    ny2 = max(ry + rh, y + h)
    return (nx1, ny1, nx2 - nx1, ny2 - ny1)


def run_webcam_segmentation_with_diff(model_path='checkpoints/best_model.ckpt', camera_id=0):
    """
    Запускает сегментацию с веб-камеры с использованием diff для оптимизации
    
    Args:
        model_path: путь к обученной модели MODNet
        camera_id: ID веб-камеры (обычно 0 для встроенной камеры)
    """
    print(f"Загрузка модели MODNet из {model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")
    
    modnet = MODNet(backbone_pretrained=False)
    modnet = torch.nn.DataParallel(modnet)
    modnet.load_state_dict(torch.load(model_path, map_location=device))
    modnet.to(device)
    modnet.eval()
    print("Модель загружена успешно!")
    
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
    modnet_size = 512
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
                
                img_resized = cv2.resize(crop_frame, (modnet_size, modnet_size))
                img_rgb = img_resized[:, :, ::-1]
                img_normalized = img_rgb.astype(np.float32) / 255.0
                
                img_tensor = torch.from_numpy(img_normalized).float().permute(2, 0, 1).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    _, _, matte = modnet(img_tensor, True)
                
                matte_np = matte[0][0].cpu().numpy()
                
                crop_mask = cv2.resize(matte_np, (new_w, new_h))
                
                kernel = np.ones((3, 3), np.uint8)
                crop_mask_uint8 = (crop_mask * 255).astype(np.uint8)
                crop_mask_uint8 = cv2.dilate(crop_mask_uint8, kernel, iterations=2)
                crop_mask_uint8 = cv2.GaussianBlur(crop_mask_uint8, (5, 5), 0)
                crop_mask = crop_mask_uint8.astype(np.float32) / 255.0
                
                crop_mask_3ch = cv2.merge([crop_mask, crop_mask, crop_mask])
                
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
        
        if elapsed_time >= 1.0:
            fps_frame_count = 0
            fps_start_time = time.time()
        
        cv2.imshow('MODNet Segmentation with Diff Optimization', result_frame)
        
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
    run_webcam_segmentation_with_diff(
        model_path='models/modnet-human-seg.ckpt',
        camera_id=0
    )
