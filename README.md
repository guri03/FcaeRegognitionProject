# FaceRecognitionProject

A real-time computer vision toolkit built with Python, OpenCV, and MediaPipe.  
Includes two modules:
- Face detection using Haar Cascades  
- Hand tracking + real-time finger counting using the MediaPipe Tasks API  

---

# Features
- Live webcam capture  
- Real-time face detection with bounding boxes  
- Real-time hand tracking  
- Accurate left/right hand finger counting  
- Modular design (choose mode from `app.py`)  

---

# Tech Stack
- Python  
- OpenCV  
- MediaPipe

---

# Setup & Installation

### 1. Clone the repository
1. git clone https://github.com/guri03/FcaeRegognitionProject.git
2. cd FaceRecognitionProject

# Create and activate a virtual environment
1. python -m venv venv
2. venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the project
1. cd src
2. python app.py

Press q at any time to close the webcam window.

# Future Enhancements
1. Face recognition (OpenCV DNN or FaceNet)
2. Gesture recognition (thumbs up, peace sign, stop gesture)
