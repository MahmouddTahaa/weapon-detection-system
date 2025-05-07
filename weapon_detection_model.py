import cv2
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

net = cv2.dnn.readNet("./config/yolov3_training_2000.weights", "./config/yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = [(0, 0, 255)]

def detect_weapons(frame):
    """Process a single frame and detect weapons"""
    if frame is None:
        return None, "No frame provided"
    
    height, width, _ = frame.shape
    
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layer_names)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
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
    
    status = "No weapons detected"
    
    if len(indexes) > 0:
        status = f"{len(indexes)} weapon(s) detected!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]} ({confidences[i]:.2f})"
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), font, 0.7, color, 2)
                print(f"Detection: {label} at ({x}, {y}, {w}, {h})")  
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame_rgb, status