from flask import Flask, render_template, jsonify
import threading
import cv2
import mediapipe as mp
import pyttsx3
import time
from deepface import DeepFace
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Global variables
camera_running = False
camera_thread = None
cap = None
current_emotion = "neutral"
emotion_lock = threading.Lock()

print("\n" + "="*60)
print("🔄 INITIALIZING ALL MODELS...")
print("="*60)

# Pre-initialize MediaPipe
print("⏳ Loading MediaPipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils
print("✓ MediaPipe loaded!")

# Pre-initialize text-to-speech
print("⏳ Loading Text-to-Speech...")
engine = pyttsx3.init()
print("✓ Text-to-Speech loaded!")

# Pre-load DeepFace models in background thread
def preload_deepface():
    print("⏳ Pre-loading DeepFace models in background...")
    try:
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False, silent=True)
        print("✓ DeepFace models pre-loaded!")
    except Exception as e:
        print(f"⚠ DeepFace pre-load: {e}")

# Start DeepFace loading in background
preload_thread = threading.Thread(target=preload_deepface, daemon=True)
preload_thread.start()

# Gesture data and KNN model
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

print("✓ KNN model trained!")

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
    "neutral": "I am okay.",
    "fear": "I am scared.",
    "disgust": "I don't like this."
}

last_audio_time = time.time() - 3

def play_audio(text):
    global last_audio_time
    if time.time() - last_audio_time >= 3:
        threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()), daemon=True).start()
        last_audio_time = time.time()

def detect_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

    features = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    
    try:
        prediction = knn.predict([features])
        if prediction[0]:
            return prediction[0]
    except:
        pass

    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

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

# Separate thread for emotion detection
def emotion_detector_thread():
    global current_emotion, camera_running
    frame_buffer = None
    
    while camera_running:
        if frame_buffer is not None:
            try:
                analysis = DeepFace.analyze(frame_buffer, actions=['emotion'], enforce_detection=False, silent=True)
                if isinstance(analysis, list):
                    analysis = analysis[0]
                
                with emotion_lock:
                    current_emotion = analysis.get('dominant_emotion', 'neutral')
            except:
                with emotion_lock:
                    current_emotion = "neutral"
        
        time.sleep(0.5)  # Check every 0.5 seconds

def run_camera():
    global camera_running, cap, current_emotion
    
    # Open camera immediately with optimized settings
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✓ Camera opened instantly!")
    
    # Start emotion detection in separate thread
    emotion_thread = threading.Thread(target=emotion_detector_thread, daemon=True)
    emotion_thread.start()
    
    frame_count = 0
    frame_for_emotion = None
    
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Gesture detection (fast)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        gesture_detected = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                gesture_detected = detect_gesture(hand_landmarks)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Update frame for emotion detection every 20 frames
        frame_count += 1
        if frame_count % 20 == 0:
            frame_for_emotion = frame.copy()

        # Get current emotion safely
        with emotion_lock:
            emotion = current_emotion
        
        emotion_text = expression_texts.get(emotion, "I am okay.")
        
        # Display on screen
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        text = gesture_texts.get(gesture_detected, emotion_text)
        cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        if gesture_detected:
            play_audio(text)

        cv2.imshow("Live Gesture & Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera_running = False
            break

    camera_running = False
    cv2.destroyAllWindows()
    print("✓ Camera window closed")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_camera')
def start_camera():
    global camera_running, camera_thread
    
    if camera_running:
        return jsonify({
            'status': 'already_running',
            'message': 'Camera is already running!'
        })
    
    try:
        camera_running = True
        camera_thread = threading.Thread(target=run_camera, daemon=True)
        camera_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Camera started successfully!'
        })
    except Exception as e:
        camera_running = False
        return jsonify({
            'status': 'error',
            'message': f'Error starting camera: {str(e)}'
        }), 500

@app.route('/stop_camera')
def stop_camera():
    global camera_running
    camera_running = False
    return jsonify({
        'status': 'success',
        'message': 'Camera stopped'
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SERVER READY at http://localhost:5000")
    print("⚡ Camera opens INSTANTLY now!")
    print("💡 Press 'q' in camera window to close it")
    print("="*60 + "\n")
    app.run(debug=True, use_reloader=False, threaded=True)