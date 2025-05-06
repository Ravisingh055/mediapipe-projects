import cv2
import mediapipe as mp
import pyautogui

# Initialize webcam and FaceMesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

while True:
    success, frame = cam.read()
    if not success:
        print("⚠️ Webcam not working! Check your camera settings.")
        break  # Exit if no frame is captured

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)

    # Debug: Print if a face is detected
    face_detected = output.multi_face_landmarks is not None
    print("Face detected:", face_detected)

    if not face_detected:
        print("⚠️ No face landmarks detected.")
        continue  # Skip processing if no face is detected

    landmark_points = output.multi_face_landmarks

    # 🔥 Debugging: Print the landmark_points structure
    print("landmark_points:", landmark_points)

    if not landmark_points:
        print("⚠️ landmark_points is None or empty. Skipping frame.")
        continue

    # Try accessing the first detected face
    first_face = landmark_points[0]
    print("First face landmarks object:", first_face)  # Debugging

    try:
        landmarks = first_face.landmark  # ✅ Use `.landmark`, NOT `.landmarks`
        print(f"Total landmarks detected: {len(landmarks)}")
    except AttributeError as e:
        print("❌ Error: landmarks attribute is missing! Skipping frame.")
        print(e)
        continue  # Skip this frame if landmarks are not found

    frame_h, frame_w, _ = frame.shape

    # Eye tracking for cursor movement
    for id, landmark in enumerate(landmarks[474:478]):
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        if id == 1:
            screen_x = screen_w * landmark.x
            screen_y = screen_h * landmark.y
            pyautogui.moveTo(screen_x, screen_y)

    # Eye blinking detection for clicking
    left_eye = [landmarks[145], landmarks[159]]
    for landmark in left_eye:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

    if (left_eye[0].y - left_eye[1].y) < 0.01:
        pyautogui.click()
        pyautogui.sleep(1)

    cv2.imshow("Eye Cursor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cam.release()
cv2.destroyAllWindows()
