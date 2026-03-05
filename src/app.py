import cv2
import mediapipe as mp
import time
from face_detection import run_face_detection
from hand_tracking import run_hand_tracking
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def run_menu():
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

    hover_start = 0
    selected_button = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # Draw buttons
        face_btn = (50, 50, 300, 150)
        hand_btn = (350, 50, 600, 150)

        cv2.rectangle(frame, face_btn[:2], face_btn[2:], (0, 255, 0), 2)
        cv2.putText(frame, "Face Detection", (face_btn[0] + 20, face_btn[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(frame, hand_btn[:2], hand_btn[2:], (255, 0, 0), 2)
        cv2.putText(frame, "Finger Counting", (hand_btn[0] + 20, hand_btn[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # MediaPipe detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            finger_x, finger_y = lm_list[8]  # index fingertip

            cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 255), -1)

            # Check if finger is inside a button
            if face_btn[0] < finger_x < face_btn[2] and face_btn[1] < finger_y < face_btn[3]:
                if selected_button != "face":
                    selected_button = "face"
                    hover_start = time.time()
                elif time.time() - hover_start > 1:
                    cap.release()
                    cv2.destroyAllWindows()
                    run_face_detection()
                    run_menu()  # Return to menu after face detection
                    return

            elif hand_btn[0] < finger_x < hand_btn[2] and hand_btn[1] < finger_y < hand_btn[3]:
                if selected_button != "hand":
                    selected_button = "hand"
                    hover_start = time.time()
                elif time.time() - hover_start > 1:
                    cap.release()
                    cv2.destroyAllWindows()
                    run_hand_tracking()
                    run_menu()  # Return to menu after hand tracking
                    return

            else:
                selected_button = None
                hover_start = 0

        cv2.imshow("Gesture Menu", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_menu()
