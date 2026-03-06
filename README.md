# RoadSense AI

RoadSense AI is a real-time road hazard detection system designed to improve rider safety. The system uses deep learning object detection models to identify hazards on the road using a live camera feed.

The system detects objects such as pedestrians, animals, vehicles, and other obstacles that may cause danger to riders.

---

## Technologies Used

- Python
- FastAPI
- YOLOv8n
- YOLOv12
- OpenCV
- MongoDB
- HTML
- CSS
- JavaScript
- WebSocket

---

## Features

- Real-time hazard detection using camera input
- Detection using YOLO deep learning models
- Live alerts when hazards are detected
- Map interface to view reported hazards
- Web-based interface for testing

---

## How to Run the Project

### Clone the repository


git clone https://github.com/alwar123/roadsense_ai.git

cd roadsense_ai


### Create virtual environment


python -m venv .venv


### Activate environment


.venv\Scripts\activate


### Install dependencies


pip install -r requirements.txt


### Run backend server


cd backend
python -m uvicorn app.main:app --reload


