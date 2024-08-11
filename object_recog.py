import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

# Load YOLOv3 configuration and weights
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the COCO dataset labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the layer names of the YOLO network
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Set up the main application window
root = tk.Tk()
root.title("YOLOv3 Real-Time Object Detection")
root.geometry("800x600")

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)

# Function to perform object detection on a frame
def detect_objects(frame):
    height, width, _ = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Apply a confidence threshold to filter out weak detections
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

    # Apply non-maxima suppression to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # List to store detected object names
    detected_objects = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.append(label)
            
            # Drawing bounding boxes and labeling detected objects
            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, detected_objects

# Function to update the frame in the GUI
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame, detected_objects = detect_objects(frame)

        # Convert the frame to PhotoImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Update the image label
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Update the text box with detected objects
        objects_text.delete(1.0, tk.END)
        objects_text.insert(tk.END, "\n".join(detected_objects))

    # Schedule the next frame update
    root.after(10, update_frame)

# Image label for displaying the video
image_label = tk.Label(root)
image_label.pack(pady=20)

# Text box for detected objects
objects_text = tk.Text(root, height=10, width=50)
objects_text.pack(pady=20)

# Start the real-time video capture and detection
update_frame()

# Start the GUI event loop
root.mainloop()

# Release the video capture when done
cap.release()
