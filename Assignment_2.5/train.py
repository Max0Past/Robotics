from ultralytics import YOLO
import torch

def train_rps():
    # 1. Завантажуємо базову модель (Nano версія)
    # Вона важить всього ~6МБ, ідеально для старту
    print("Initializing YOLOv8 Nano...")
    model = YOLO('yolov8n.pt') 

    # 2. Запуск тренування
    # Ми явно вказуємо параметри для економії VRAM
    print("Starting training on GPU...")
    
    try:
        results = model.train(
            # ШЛЯХ ДО ДАНИХ (перевір назву папки!)
            data='./rock-paper-scissors-14/data.yaml', 
            
            # ГІПЕРПАРАМЕТРИ
            epochs=30,           # 30 епох вистачить для першого тесту
            imgsz=416,           # Зменшуємо з 640 до 416, щоб влізти в 4GB VRAM
            batch=16,            # Кількість картинок за раз. Якщо вилетить помилка - став 8
            
            # ТЕХНІЧНІ НАЛАШТУВАННЯ
            device=0,            # Використовуємо твою RTX 3050
            workers=2,           # На Windows краще ставити мало воркерів (0, 1 або 2)
            
            # ЗБЕРЕЖЕННЯ
            project='runs/detect',  # Куди зберігати
            name='rps_3050_v1',     # Ім'я папки з результатами
            exist_ok=True,          # Перезаписувати, якщо папка вже є
            amp=True                # Mixed Precision (швидше і менше пам'яті)
        )
        
        print("Training finished successfully!")
        
        # 3. Валідація (перевірка точності)
        metrics = model.val()
        print(f"Final mAP50: {metrics.box.map50}")
        
        metrics = model.val()
    
        # Витягуємо числа
        map50 = metrics.box.map50    # mAP при IoU=0.5
        map50_95 = metrics.box.map   # mAP при IoU=0.5:0.95
        precision = metrics.box.mp   # Середня точність (Mean Precision)
        recall = metrics.box.mr      # Середня повнота (Mean Recall)

        print("\n" + "="*40)
        print("       ПІДСУМКОВІ МЕТРИКИ МОДЕЛІ")
        print("="*40)
        print(f"Precision (Точність):  {precision:.4f}")
        print(f"Recall (Повнота):      {recall:.4f}")
        print(f"mAP@50 (Основна):      {map50:.4f}")
        print(f"mAP@50-95 (Сувора):    {map50_95:.4f}")
        print("="*40)
        
    except Exception as e:
        print(f"Сталася помилка: {e}")
        print("ПОРАДА: Якщо помилка 'CUDA out of memory', зменши batch до 8 або 4.")

if __name__ == '__main__':
    train_rps()