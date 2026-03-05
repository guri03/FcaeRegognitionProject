import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def count_fingers(lm_list, handedness):
    """
    Count raised fingers.

    MediaPipe labels handedness from the *camera's* perspective (mirrored),
    so "Right" in the label = your physical LEFT hand when facing the camera.

    Finger tips (index–pinky): up when tip.y < pip.y  — same for both hands.
    Thumb: up/down determined by X position, direction depends on the
           (mirrored) handedness label.
    """
    fingers = []

    # --- Thumb ---
    # For MediaPipe's mirrored "Right" label: thumb tip should be to the RIGHT of knuckle
    # For MediaPipe's mirrored "Left"  label: thumb tip should be to the LEFT of knuckle
    if handedness == "Right":
        # Physical left hand (facing camera): thumb opens to the right
        fingers.append(1 if lm_list[4][0] > lm_list[3][0] else 0)
    else:
        # Physical right hand (facing camera): thumb opens to the left
        fingers.append(1 if lm_list[4][0] < lm_list[3][0] else 0)

    # --- Index, Middle, Ring, Pinky ---
    # A finger is UP when its tip is HIGHER on screen than its PIP joint.
    # "Higher" = smaller Y value. This rule is the SAME for both hands.
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(1 if lm_list[tip][1] < lm_list[tip - 2][1] else 0)

    return fingers


def run_hand_tracking():
    model_path = "../assets/hand_landmarker.task"

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,  # allow up to 2 hands for robustness
    )

    detector = HandLandmarker.create_from_options(options)

    selected_button = None
    hover_start = 0

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Instruction label (fixed: was broken across two lines with implicit concat)
        text = "Use front hand"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(frame, text,
                    (w - text_w - 20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Back button
        btn_width, btn_height = 250, 100
        back_btn = (20, h - btn_height - 20, 20 + btn_width, h - 20)
        cv2.rectangle(frame, back_btn[:2], back_btn[2:], (0, 255, 0), 2)
        cv2.putText(frame, "Back", (back_btn[0] + 20, back_btn[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hand detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            for idx in range(len(result.hand_landmarks)):
                hand_landmarks = result.hand_landmarks[idx]
                handedness = result.handedness[idx][0].category_name  # "Left" or "Right"

                lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                # Index fingertip indicator
                finger_x, finger_y = lm_list[8]
                cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 255), -1)

                # Back button hover
                if back_btn[0] < finger_x < back_btn[2] and back_btn[1] < finger_y < back_btn[3]:
                    if selected_button != "back":
                        selected_button = "back"
                        hover_start = time.time()
                    elif time.time() - hover_start > 1:
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                else:
                    selected_button = None
                    hover_start = 0

                # Count fingers using fixed logic
                fingers = count_fingers(lm_list, handedness)
                total_fingers = fingers.count(1)

                # Display (offset per hand so two hands don't overlap)
                y_offset = 40 + idx * 110
                cv2.putText(frame, f"{handedness} Hand: {total_fingers} fingers",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Hand Tracking (MediaPipe Tasks API)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
