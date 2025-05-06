import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open Camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Create Blank Canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Colors & Settings
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]  # Red, Green, Blue, Yellow, Pink
color_index = 0
thickness = 10
eraser_mode = False

prev_x, prev_y = None, None  # Store Previous Cursor Position

# Opacity Slider
opacity = 1.0

# Create Trackbar Window
cv2.namedWindow("Settings")

def update_opacity(val):
    global opacity
    opacity = val / 100  # Convert from 0-100 scale to 0-1.0

cv2.createTrackbar("Opacity", "Settings", int(opacity * 100), 100, update_opacity)

# Button Positions
eraser_button = (1100, 50, 150, 50)  # (x, y, width, height)
brush_button = (900, 50, 150, 50)

def draw_buttons(frame):
    cv2.rectangle(frame, (brush_button[0], brush_button[1]), (brush_button[0] + brush_button[2], brush_button[1] + brush_button[3]), (255, 255, 255), -1)
    cv2.putText(frame, "Brush", (brush_button[0] + 40, brush_button[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.rectangle(frame, (eraser_button[0], eraser_button[1]), (eraser_button[0] + eraser_button[2], eraser_button[1] + eraser_button[3]), (200, 200, 200), -1)
    cv2.putText(frame, "Eraser", (eraser_button[0] + 35, eraser_button[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_x = int(hand_landmarks.landmark[8].x * 1280)
            index_y = int(hand_landmarks.landmark[8].y * 720)
            thumb_x = int(hand_landmarks.landmark[4].x * 1280)
            thumb_y = int(hand_landmarks.landmark[4].y * 720)

            # Detect pinch gesture (Thumb & Index close together) -> Draw
            distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            if distance < 50:  # Pinch detection for drawing
                if prev_x is not None and prev_y is not None:
                    num_steps = max(abs(index_x - prev_x), abs(index_y - prev_y)) // 5
                    for i in range(1, num_steps + 1):
                        inter_x = int(prev_x + (index_x - prev_x) * (i / num_steps))
                        inter_y = int(prev_y + (index_y - prev_y) * (i / num_steps))
                        if eraser_mode:
                            cv2.circle(canvas, (inter_x, inter_y), thickness + 10, (0, 0, 0), -1)
                        else:
                            cv2.circle(canvas, (inter_x, inter_y), thickness, colors[color_index], -1)

                prev_x, prev_y = index_x, index_y
            else:
                prev_x, prev_y = None, None

            # Check if user clicks buttons
            if 50 < index_y < 100:  # Button Click Area
                if 900 < index_x < 1050:  # Brush Button Clicked
                    eraser_mode = False
                elif 1100 < index_x < 1250:  # Eraser Button Clicked
                    eraser_mode = True

            # Draw Hand Landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Blend canvas with frame using adjustable opacity
    color_mask = cv2.addWeighted(canvas, opacity, np.zeros_like(canvas), 1 - opacity, 0)
    frame = cv2.addWeighted(frame, 1, color_mask, 1, 0)

    # Draw Buttons
    draw_buttons(frame)

    # Show Output
    cv2.imshow("Virtual Painter", frame)
    cv2.imshow("Settings", np.zeros((100, 300, 3), dtype=np.uint8))  # Keep slider window open

    # Keyboard Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Clear canvas
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    elif key == ord('q'):  # Quit
        break
    elif key in [ord(str(i)) for i in range(len(colors))]:  # Change Colors
        color_index = int(chr(key))

cap.release()
cv2.destroyAllWindows()
