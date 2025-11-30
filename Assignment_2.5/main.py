import cv2
import time
import random
import math
from ultralytics import YOLO

# --- НАЛАШТУВАННЯ ---
MODEL_PATH = 'Assignment_2.5\\best.pt'      # Переконайся, що файл лежить поруч
CONF_THRESHOLD = 0.5        # Поріг впевненості (50%)
GAME_DURATION = 3           # Час на таймері

# ВАЖЛИВО: Перевір порядок класів у своєму data.yaml!
# Зазвичай у цьому датасеті: 0=Paper, 1=Rock, 2=Scissors
# Але іноді буває інакше. Якщо гра буде плутати - поміняй місцями назви тут.
CLASS_MAP = {
    0: "Paper",
    1: "Rock",
    2: "Scissors"
}

def determine_winner(player, ai):
    if player == ai: return "TIE", (255, 255, 0) # Yellow
    
    if (player == "Rock" and ai == "Scissors") or \
       (player == "Scissors" and ai == "Paper") or \
       (player == "Paper" and ai == "Rock"):
        return "YOU WIN", (0, 255, 0) # Green
        
    return "AI WINS", (0, 0, 255) # Red

def main():
    # 1. Завантаження моделі
    print("Завантаження нейромережі...")
    model = YOLO(MODEL_PATH)

    # 2. Камера
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280) # Ширина
    cap.set(4, 720)  # Висота

    # Змінні стану гри
    state = "WAITING" # WAITING -> COUNTDOWN -> RESULT
    timer_start = 0
    ai_move = ""
    player_move = ""
    result_text = ""
    result_color = (255, 255, 255)

    print("Гра почалася! Тисни ПРОБІЛ.")

    while True:
        success, img = cap.read()
        if not success: break
        
        # Віддзеркалення (дзеркальний ефект)
        img = cv2.flip(img, 1)
        
        # Копія для "чистого" аналізу, але малюємо на img
        # (YOLOv8 робить це сам, але про всяк випадок)
        
        # --- AI INFERENCE ---
        # verbose=False щоб не засмічувати консоль
        results = model(img, stream=True, verbose=False)
        
        current_detected_class = None
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Впевненість
                conf = float(box.conf[0])
                if conf > CONF_THRESHOLD:
                    # Клас
                    cls_id = int(box.cls[0])
                    current_detected_class = CLASS_MAP.get(cls_id, "Unknown")
                    
                    # Координати для малювання
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Малюємо рамку
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(img, f"{current_detected_class} {int(conf*100)}%", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 
                                0.8, (255, 0, 255), 2)

        # --- GAME LOGIC ---
        
        # 1. СТАН ОЧІКУВАННЯ
        if state == "WAITING":
            cv2.putText(img, "Press SPACE to Start", (400, 360), 
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

        # 2. СТАН ТАЙМЕРА (3..2..1)
        elif state == "COUNTDOWN":
            elapsed = time.time() - timer_start
            time_left = GAME_DURATION - elapsed
            
            if time_left > 0:
                cv2.putText(img, str(int(time_left)+1), (600, 360), 
                            cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 10)
            else:
                # Час вийшов! Фіксуємо хід
                state = "RESULT"
                
                # Хід AI (рандом)
                moves = ["Rock", "Paper", "Scissors"]
                ai_move = random.choice(moves)
                
                # Хід гравця
                if current_detected_class:
                    player_move = current_detected_class
                    result_text, result_color = determine_winner(player_move, ai_move)
                else:
                    player_move = "Nothing"
                    result_text = "HAND NOT FOUND"
                    result_color = (0, 0, 255)

        # 3. СТАН РЕЗУЛЬТАТУ
        elif state == "RESULT":
            # Показуємо хто що вибрав
            cv2.putText(img, f"You: {player_move}", (50, 100), 
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 3)
            cv2.putText(img, f"AI: {ai_move}", (900, 100), 
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 3)
            
            # Хто переміг
            cv2.putText(img, result_text, (350, 360), 
                        cv2.FONT_HERSHEY_PLAIN, 4, result_color, 5)
            
            cv2.putText(img, "Press SPACE to Restart", (400, 600), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), 2)

        # Керування клавішами
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == 32: # SPACE
            if state == "WAITING" or state == "RESULT":
                state = "COUNTDOWN"
                timer_start = time.time()

        cv2.imshow("Rock Paper Scissors AI", img)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()