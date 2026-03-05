import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def run_hand_tracking():
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

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Timestamp needed for VIDEO mode
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Run detection
        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            # Only using the first detected hand
            hand_landmarks = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name 

            h, w, _ = frame.shape

            # Convert normalized landmarks to pixel coordinates
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            fingers = []

            # Thumb logic depends on left/right hand
            if handedness == "Right":
                if lm_list[4][0] < lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:  # Left hand
                if lm_list[4][0] > lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Other 4 fingers (same logic for both hands)
            tips = [8, 12, 16, 20]
            for tip in tips:
                if lm_list[tip][1] < lm_list[tip - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)

            # Display result
            cv2.putText(frame, f"{handedness} Hand", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(frame, f"Fingers: {total_fingers}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Hand Tracking (MediaPipe Tasks API)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
