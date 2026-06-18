# ACCIDENT-DETECTION-ALERT-SYSTEM

An AI-powered real-time accident detection system that uses YOLOv4-Tiny and OpenCV to identify vehicle collisions from video streams or webcams. When a potential accident is detected, the system triggers an alert, saves evidence images, and sends emergency notifications via SMS using Twilio. The project utilizes computer vision techniques to enhance road safety and support faster emergency response.

Features
Real-time vehicle detection using YOLOv4-Tiny
Collision detection based on vehicle overlap analysis
Automatic accident alerts with sound notifications
SMS alerts through Twilio integration
Accident image capture and storage
Works with webcam and recorded videos
Lightweight and efficient for real-time monitoring

Tech Stack
Python
OpenCV
YOLOv4-Tiny
NumPy
Twilio API
Computer Vision

Project Structure
accident-detection/
│
├── accident_images/
├── detected_accidents/
├── archive/
│   └── data/
│       ├── train/
│       ├── test/
│       └── val/
│
├── accident_detection.py
├── coco.names
├── yolov4-tiny.cfg
├── yolov4-tiny.weights
├── requirements.txt
└── README.md

How It Works
Detects vehicles in each frame using YOLOv4-Tiny.
Tracks vehicle bounding boxes.
Identifies possible collisions through bounding box overlap.
Triggers an alarm when an accident is detected.
Saves accident snapshots for evidence.
Sends SMS notifications to emergency contacts.

Installation
git clone https://github.com/your-username/accident-detection.git
cd accident-detection

pip install -r requirements.txt
python accident_detection.py

Future Enhancements
GPS location sharing
Emergency service integration
Deep learning-based accident severity prediction
Cloud dashboard for monitoring
Live CCTV camera support

Author
Akshata Patil
B.Tech Artificial Intelligence
St. Vincent Pallotti College of Engineering & Technology, Nagpur

⭐ I you find this project useful, consider giving it a star on GitHub!
