import cv2
import mediapipe as mp
import threading
import pyttsx3
import time
from deepface import DeepFace
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Initialize text-to-speech
engine = pyttsx3.init()

gesture_data = {
    "fist": [[1.0, 1.0, 1.0, 1.0, 1.0], [0.9, 0.9, 0.9, 0.9, 0.9]],
    "index_up": [[0.2, 1.0, 1.0, 1.0, 1.0], [0.3, 0.9, 0.9, 0.9, 0.9]],
    "open_palm": [[0.2, 0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3, 0.3]],
    "peace_sign": [[0.2, 0.2, 1.0, 1.0, 1.0], [0.3, 0.3, 0.9, 0.9, 0.9]],
    "thumbs_up": [[0.2, 1.0, 1.0, 1.0, 1.0], [0.3, 0.9, 0.9, 0.9, 0.9]],
    "thumbs_down": [[1.8, 1.0, 1.0, 1.0, 1.0], [1.7, 0.9, 0.9, 0.9, 0.9]],
    "ok_sign": [[0.2, 0.2, 1.0, 1.0, 1.0], [0.3, 0.3, 0.9, 0.9, 0.9]],
    "call_me": [[0.2, 1.0, 1.0, 1.0, 0.2], [0.3, 0.9, 0.9, 0.9, 0.3]],
    "rock_on": [[0.2, 1.0, 1.0, 0.2, 0.2], [0.3, 0.9, 0.9, 0.3, 0.3]],
    "three": [[0.2, 0.2, 0.2, 1.0, 1.0], [0.3, 0.3, 0.3, 0.9, 0.9]],
    "four": [[0.2, 0.2, 0.2, 0.2, 1.0], [0.3, 0.3, 0.3, 0.3, 0.9]],
    "gun": [[1.0, 0.2, 0.2, 1.0, 1.0], [0.9, 0.3, 0.3, 0.9, 0.9]],
    "pinch": [[0.2, 0.2, 1.0, 1.0, 1.0], [0.3, 0.3, 0.9, 0.9, 0.9]],
    "spider_man": [[0.2, 0.2, 1.0, 1.0, 0.2], [0.3, 0.3, 0.9, 0.9, 0.3]],
    "crossed_fingers": [[1.0, 0.2, 0.2, 1.0, 1.0], [0.9, 0.3, 0.3, 0.9, 0.9]],
    "finger_heart": [[0.2, 0.2, 1.0, 1.0, 1.0], [0.3, 0.3, 0.9, 0.9, 0.9]],
    "middle_finger": [[1.0, 1.0, 0.2, 1.0, 1.0], [0.9, 0.9, 0.3, 0.9, 0.9]],
    "vulcan_salute": [[0.2, 0.2, 0.2, 1.0, 0.2], [0.3, 0.3, 0.3, 0.9, 0.3]],
    "writing": [[0.2, 0.2, 0.2, 1.0, 1.0], [0.3, 0.3, 0.3, 0.9, 0.9]],
    "phone": [[0.2, 1.0, 1.0, 1.0, 0.2], [0.3, 0.9, 0.9, 0.9, 0.3]]
}



# Convert to DataFrame and train KNN
data_list = []
for gesture, positions in gesture_data.items():
    for pos in positions:
        data_list.append({
            'gesture': gesture,
            'thumb': pos[0],
            'index': pos[1],
            'middle': pos[2],
            'ring': pos[3],
            'pinky': pos[4]
        })

gesture_df = pd.DataFrame(data_list)
X = gesture_df.drop('gesture', axis=1)
y = gesture_df['gesture']
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

gesture_texts = {
    "fist": "Please give me water",
    "index_up": "I am hungry",
    "open_palm": "Hi",
    "thumbs_up": "Good job!",
    "thumbs_down": "I don't like this.",
    "peace_sign": "Peace!",
    "ok_sign": "Okay!",
    "call_me": "Call me, please!",
    "rock_on": "Rock on!",
    "three": "I want to bath",
    "four": "Headache",
    "gun": "Finger gun",
    "pinch": "A little bit",
    "spider_man": "Spider-Man!",
    "crossed_fingers": "Wish me luck!",
    "finger_heart": "I love you",
    "middle_finger": "I am angry",
    "vulcan_salute": "Live long and prosper",
    "writing": "I need to write something",
    "phone": "Let's talk on the phone"
}

expression_texts = {
    "angry": "I am feeling angry!",
    "happy": "I am happy today!",
    "sad": "I am feeling sad.",
    "surprise": "Wow, I am surprised!",
    "neutral": "I am okay."
}

last_audio_time = time.time() - 3

def play_audio(text):
    global last_audio_time
    if time.time() - last_audio_time >= 3:
        threading.Thread(target=lambda: (engine.say(text), engine.runAndWait())).start()
        last_audio_time = time.time()

def detect_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    
    # Extract key points
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

    # Create feature vector
    features = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    
    # Try ML-based prediction first
    try:
        prediction = knn.predict([features])
        if prediction[0]:
            return prediction[0]
    except:
        pass

    # Fallback to rule-based detection
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]

    if all(landmark.y > index_mcp.y for landmark in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "fist"
    if index_tip.y < index_mcp.y and all(landmark.y > index_mcp.y for landmark in [middle_tip, ring_tip, pinky_tip]):
        return "index_up"
    if all(landmark.y < index_mcp.y for landmark in [index_tip, middle_tip, ring_tip, pinky_tip]) and thumb_tip < index_mcp.y:
        return "open_palm"
    if thumb_tip < index_tip and pinky_tip < index_tip:
        return "peace_sign"
    if index_tip < thumb_tip and pinky_tip < thumb_tip and middle_tip > index_tip:
        return "call_me"
    if all(landmark.y < index_mcp.y for landmark in [index_tip, middle_tip, ring_tip]) and pinky_tip > index_mcp.y:
        return "thumbs_up"
    if all(landmark.y > index_mcp.y for landmark in [index_tip, middle_tip, ring_tip]) and pinky_tip < index_mcp.y:
        return "thumbs_down"
    if thumb_tip < index_tip and middle_tip > index_tip and ring_tip > index_tip:
        return "ok_sign"
    
    return None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    gesture_detected = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            gesture_detected = detect_gesture(hand_landmarks)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]

        emotion = analysis.get('dominant_emotion', 'neutral')
        emotion_text = expression_texts.get(emotion, "I don't know what to say.")
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except:
        emotion_text = "I don't know what to say."

    text = gesture_texts.get(gesture_detected, emotion_text)
    cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    play_audio(text)

    cv2.imshow("Live Gesture & Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()