import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Cursor movement settings
cursor_speed = 0.8  # Reduce speed multiplier for more control
smooth_factor = 5  # Higher value = smoother but slower response

# Open webcam
cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0  # Store previous cursor position

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)  # Flip for natural movement
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    result = hands.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract **Thumb Tip** for movement
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            x = int(thumb.x * frame_w)
            y = int(thumb.y * frame_h)

            # Map coordinates to screen size
            screen_x = np.interp(thumb.x, [0.2, 0.8], [0, screen_w])
            screen_y = np.interp(thumb.y, [0.2, 0.8], [0, screen_h])

            # Apply smoothing for better control
            new_x = prev_x + (screen_x - prev_x) / smooth_factor
            new_y = prev_y + (screen_y - prev_y) / smooth_factor

            pyautogui.moveTo(new_x, new_y)
            prev_x, prev_y = new_x, new_y  # Update previous position

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract **Index Finger Tip** for clicking
            index_x = int(index_finger.x * frame_w)
            index_y = int(index_finger.y * frame_h)

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # Thumb (Green)
            cv2.circle(frame, (index_x, index_y), 10, (0, 0, 255), -1)  # Index (Red)

            # Check for pinch gesture (Thumb & Index close together)
            if abs(x - index_x) < 30 and abs(y - index_y) < 30:
                pyautogui.click()
                pyautogui.sleep(0.2)  # Small delay to prevent multiple clicks

    cv2.imshow("Thumb Cursor", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
