import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# Initialize pygame for sound
pygame.mixer.init()

# Load drum samples - make sure these files exist
sounds = {
    "snare": pygame.mixer.Sound("snare.wav"),
    "kick": pygame.mixer.Sound("kick.wav"),
    "hihat": pygame.mixer.Sound("hihat.wav"),
    "crash": pygame.mixer.Sound("crash.wav")
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Drum pad positions (relative coordinates 0-1)
drum_pads = {
    "snare": (0.3, 0.6),
    "kick": (0.5, 0.8),
    "hihat": (0.7, 0.6),
    "crash": (0.5, 0.4)
}

# Initialize webcam
cap = cv2.VideoCapture(0)
width, height = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Cooldown tracking for each drum pad
last_hit_times = {pad_name: 0 for pad_name in drum_pads}
HIT_COOLDOWN = 0.2  # seconds

def draw_drum_pads(frame):
    """Draw virtual drum pads on the frame"""
    for pad_name, (x, y) in drum_pads.items():
        center_x = int(x * width)
        center_y = int(y * height)
        cv2.circle(frame, (center_x, center_y), 60, (100, 100, 100), 2)
        cv2.putText(frame, pad_name, (center_x-30, center_y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def is_palm_open(hand_landmarks):
    """Check if hand is open (palm)"""
    # Check if all fingers are extended
    fingers_open = 0
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:  # Finger tip and pip joints
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers_open += 1
    return fingers_open >= 3  # At least 3 fingers extended

def check_drum_hit(hand_landmarks):
    """Check if palm is hitting a drum pad"""
    wrist = hand_landmarks.landmark[0]
    wrist_x, wrist_y = wrist.x, wrist.y
    
    for pad_name, (pad_x, pad_y) in drum_pads.items():
        distance = np.sqrt((wrist_x - pad_x)**2 + (wrist_y - pad_y)**2)
        
        if distance < 0.1 and (time.time() - last_hit_times[pad_name]) > HIT_COOLDOWN:
            if is_palm_open(hand_landmarks):
                sounds[pad_name].play()
                last_hit_times[pad_name] = time.time()
                return True, pad_name
    
    return False, None

# Main game loop
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw drum pads
        draw_drum_pads(frame)
        
        # Display instructions
        cv2.putText(frame, "Air Drum Kit - Use open palm to play", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Both hands can play simultaneously", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press Q to quit", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Process hands
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
                
                # Check drum hits
                hit, pad_name = check_drum_hit(hand_landmarks)
                if hit:
                    center_x = int(drum_pads[pad_name][0] * width)
                    center_y = int(drum_pads[pad_name][1] * height)
                    cv2.circle(frame, (center_x, center_y), 70, (0, 255, 255), 3)
                    cv2.putText(frame, pad_name.upper(), (center_x-30, center_y-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        cv2.imshow('Air Drum Kit', frame)
        
        # Check for quit key
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):  # Optional: Add a reset key
            last_hit_times = {pad_name: 0 for pad_name in drum_pads}

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()