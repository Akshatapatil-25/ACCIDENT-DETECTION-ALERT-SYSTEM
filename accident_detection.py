import cv2
import numpy as np
import os
import winsound
from datetime import datetime
from twilio.rest import Client

# Load YOLO model
cfg_path = r"C:\Projects\accident detection\yolov4-tiny.cfg"
weights_path = r"C:\Projects\accident detection\yolov4-tiny.weights"
names_path = r"C:\Projects\accident detection\coco.names"

# Load YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

vehicle_classes = ['car', 'motorbike', 'bus', 'truck']

# Create folder for detected accidents
output_folder = "detected_accidents"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize Video Capture from webcam (real-time)
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

accident_counter = 0
accident_detected = False
sms_sent = False  # To avoid sending multiple SMS for one accident

# Twilio setup
account_sid = "AC5260a6888606cbfd1b823b279ab6fa96"
auth_token = "a706cde49238da1f0e8f186109736e05"
client = Client(account_sid, auth_token)

def send_sms_alert():
    message = client.messages.create(
        body="🚨 Accident Detected! Immediate attention required.",
        from_="+15177600819",
        to="+918999687108"
    )
    print("SMS sent:", message.sid)

print("[INFO] Starting Accident Detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Failed to grab frame")
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    height, width, _ = frame.shape

    # Object Detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in vehicle_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    current_boxes = []

    for i in range(len(boxes)):
        if i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            current_boxes.append((x, y, w, h))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    accident_now = False

    # Collision detection
    for i in range(len(current_boxes)):
        for j in range(i + 1, len(current_boxes)):
            x1, y1, w1, h1 = current_boxes[i]
            x2, y2, w2, h2 = current_boxes[j]
            if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
                accident_now = True

    if accident_now and not accident_detected:
        print("🚨 Accident Detected!")
        winsound.Beep(1000, 1000)  # Beep sound
        accident_detected = True
        accident_counter += 1

        # Save accident frame
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        accident_image_path = os.path.join(output_folder, f"accident_{timestamp}.jpg")
        cv2.imwrite(accident_image_path, frame)

        # Send SMS once
        if not sms_sent:
            send_sms_alert()
            sms_sent = True

    if not accident_now:
        accident_detected = False
        sms_sent = False  # Reset for next accident

    # Display
    cv2.putText(frame, f"Accidents Detected: {accident_counter}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()
