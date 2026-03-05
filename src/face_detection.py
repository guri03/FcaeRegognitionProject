import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def run_face_detection():
    cascade_path = "../assets/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)

    model_path = "../assets/hand_landmarker.task"
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
    )

    detector = HandLandmarker.create_from_options(options)

    selected_button = None
    hover_start = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # Back button
        back_btn = (50, 50, 300, 150)
        cv2.rectangle(frame, back_btn[:2], back_btn[2:], (0, 255, 0), 2)
        cv2.putText(frame, "Back", (back_btn[0] + 20, back_btn[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hand detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            finger_x, finger_y = lm_list[8]
            cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 255), -1)

            # Hover detection
            if back_btn[0] < finger_x < back_btn[2] and back_btn[1] < finger_y < back_btn[3]:
                if selected_button != "back":
                    selected_button = "back"
                    hover_start = time.time()
                elif time.time() - hover_start > 1:
                    cap.release()
                    cv2.destroyAllWindows()
                    return  # <-- FIX: return to app.py
            else:
                selected_button = None
                hover_start = 0

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w2, h2) in faces:
            cv2.rectangle(frame, (x, y), (x + w2, y + h2), (0, 255, 0), 2)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
