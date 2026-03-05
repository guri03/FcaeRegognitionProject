from face_detection import run_face_detection
from hand_tracking import run_hand_tracking

choice = input("Choose mode: (F) Face Detection  (H) Hand Tracking: ")

if choice == "F":
    run_face_detection()
elif choice == "H":
    run_hand_tracking()
else:
    print("Invalid choice")
