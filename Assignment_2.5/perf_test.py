from ultralytics import YOLO

def main():
    # 1. Завантажуємо модель. 
    # 'yolov8n.pt' завантажиться автоматично з інтернету при першому запуску (близько 6 МБ)
    model = YOLO('yolov8n.pt')  

    # 2. Запускаємо тренування
    # data='coco8.yaml' — це вбудований в пакет мікро-датасет (4 картинки для тесту)
    # epochs=3 — достатньо, щоб побачити, що лосс падає
    # imgsz=640 — стандартний розмір
    # device=0 — явно вказуємо твою GPU
    
    print("Starting training...")
    results = model.train(
        data='coco8.yaml', 
        epochs=3, 
        imgsz=640, 
        device=0, 
        batch=4,      # Маленький батч для безпеки
        workers=2,    # Оптимально для Windows
        name='test_run' # Назва папки, куди збережуться результати
    )
    
    print("Training finished successfully!")

if __name__ == '__main__':
    # На Windows обов'язково треба загортати код у if __name__ == '__main__'
    # через особливості мультипроцесингу (spawn method)
    main()