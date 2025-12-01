import cv2
import time
import random
from ultralytics import YOLO

# --- НАЛАШТУВАННЯ ---
MODEL_PATH = 'Assignment_2.5\\best.pt'       
CONF_THRESHOLD = 0.5        # Поріг впевненості (50%)
GAME_DURATION = 3           # Час на таймері (секунди)

CLASS_MAP = {
    0: "Paper",
    1: "Rock",
    2: "Scissors"
}

def determine_winner(player, ai):
    """
    Визначає переможця і повертає текст та колір для виводу.
    """
    # Нічия (Жовтий)
    if player == ai: 
        return "TIE", (0, 255, 255) 
    
    # Перемога гравця (Зелений)
    if (player == "Rock" and ai == "Scissors") or \
       (player == "Scissors" and ai == "Paper") or \
       (player == "Paper" and ai == "Rock"):
        return "YOU WIN!", (0, 255, 0) 
        
    # Перемога AI (Червоний)
    return "AI WINS!", (0, 0, 255) 

def main():
    # 1. Завантаження моделі
    print(f"Завантаження нейромережі з {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"ПОМИЛКА: Не знайдено файл моделі! Перевір шлях. Деталі: {e}")
        return

    # 2. Налаштування камери
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280) # Ширина
    cap.set(4, 720)  # Висота
    
    print(f"Камера відкрита: {cap.isOpened()}")

    # Змінні стану гри
    state = "WAITING" # Можливі стани: WAITING, COUNTDOWN, RESULT
    timer_start = 0
    
    ai_move = "???"
    player_move = "???"
    result_text = ""
    result_color = (255, 255, 255)

    print("Гра готова! Натисніть 'q' для виходу.")

    while True:
        success, img = cap.read()
        if not success:
            print("Не вдалося отримати кадр з камери.")
            print(f"Статус камери: {cap.isOpened()}")
            break
        
        # Віддзеркалення
        img = cv2.flip(img, 1)
        
        # --- 3. ДЕТЕКЦІЯ (YOLO) ---
        # verbose=False, щоб не спамити в консоль
        results = model(img, stream=True, verbose=False)
        
        # Список для збереження всіх рук, знайдених у поточному кадрі
        detected_hands_in_frame = [] 
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                
                # Фільтруємо слабкі детекції
                if conf > CONF_THRESHOLD:
                    cls_id = int(box.cls[0])
                    class_name = CLASS_MAP.get(cls_id, "Unknown")
                    
                    # Додаємо в список для подальшої логіки
                    detected_hands_in_frame.append(class_name)
                    
                    # --- Візуалізація ---
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Рамка навколо руки (Фіолетова)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    
                    # Текст над рамкою
                    label = f"{class_name} {int(conf*100)}%"
                    cv2.putText(img, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # --- 4. ЛОГІКА ГРИ (STATE MACHINE) ---
        
        # СТАН 1: ОЧІКУВАННЯ
        if state == "WAITING":
            # Малюємо підказку по центру
            text = "Press SPACE to Start"
            # Отримуємо розмір тексту, щоб відцентрувати (приблизно)
            cv2.putText(img, text, (350, 650), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255), 2)

        # СТАН 2: ЗВОРОТНИЙ ВІДЛІК
        elif state == "COUNTDOWN":
            elapsed = time.time() - timer_start
            time_left = GAME_DURATION - elapsed
            
            if time_left > 0:
                # Виводимо великі цифри: 3... 2... 1...
                display_num = str(int(time_left) + 1)
                cv2.putText(img, display_num, (580, 400), 
                            cv2.FONT_HERSHEY_TRIPLEX, 6, (0, 165, 255), 10)
            else:
                # ЧАС ВИЙШОВ! Переходимо до аналізу результатів
                state = "RESULT"
                
                # 1. Перевірка: Скільки рук ми бачимо?
                hand_count = len(detected_hands_in_frame)
                
                if hand_count == 0:
                    # Рук немає
                    player_move = "Nothing"
                    ai_move = "???"
                    result_text = "HAND NOT FOUND"
                    result_color = (0, 0, 255) # Червоний
                    
                elif hand_count > 1:
                    # Забагато рук (чітерство або помилка)
                    player_move = "Too Many!"
                    ai_move = "???"
                    result_text = "ONE HAND ONLY!"
                    result_color = (0, 0, 255) # Червоний
                    
                else:
                    # Рівно одна рука - ГРАЄМО!
                    player_move = detected_hands_in_frame[0]
                    
                    # Хід штучного інтелекту
                    possible_moves = ["Rock", "Paper", "Scissors"]
                    ai_move = random.choice(possible_moves)
                    
                    # Визначаємо переможця
                    result_text, result_color = determine_winner(player_move, ai_move)

        # СТАН 3: ПОКАЗ РЕЗУЛЬТАТУ
        elif state == "RESULT":
            # Інформація про ходи зверху
            cv2.putText(img, f"You: {player_move}", (50, 100), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(img, f"AI: {ai_move}", (900, 100), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 2)
            
            # Результат по центру (WIN / LOSE / ERROR)
            # Трохи центруємо текст
            text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_TRIPLEX, 2.5, 5)[0]
            text_x = (1280 - text_size[0]) // 2
            
            cv2.putText(img, result_text, (text_x, 400), 
                        cv2.FONT_HERSHEY_TRIPLEX, 2.5, result_color, 5)
            
            # Підказка про рестарт знизу
            cv2.putText(img, "Press SPACE to Restart", (420, 650), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), 2)

        # --- 5. ОБРОБКА КЛАВІШ ---
        key = cv2.waitKey(1)
        
        if key == ord('q') or key == 1081 or key == 27: 
            print("Вихід користувача...")
            break
        
        if key == 32: # Пробіл (SPACE)
            # Запускаємо таймер тільки якщо ми чекаємо або гра вже закінчилась
            if state == "WAITING" or state == "RESULT":
                state = "COUNTDOWN"
                timer_start = time.time()
        
        # Показуємо фінальну картинку
        cv2.imshow("Rock Paper Scissors AI", img)
        
        # Перевіряємо, чи закрито вікно (має бути ПІСЛЯ imshow)
        if cv2.getWindowProperty("Rock Paper Scissors AI", cv2.WND_PROP_VISIBLE) < 1:
            print("Вікно закрито.")
            break

    # Очистка ресурсів після виходу
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()