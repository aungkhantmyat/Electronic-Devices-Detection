import random
import time
import math
import os
import cv2
import json
import shutil
from ultralytics import YOLO
import keyboard

start_time = 0
end_time = 0
prev_state = "No Electronic Device Detected"
flag = False
cap = cv2.VideoCapture(0)
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video = str(random.randint(1, 50000)) + "EDViolation.avi"
writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"XVID"), 10, (width,height))

#ED Related
my_file = open("utils/coco.txt", "r") # opening the file in read mode
data = my_file.read() # reading the file
class_list = data.split("\n") # replacing end splitting the text | when newline ('\n') is seen.
my_file.close()
detected_things = []
detection_colors = [] # Generate random colors for class list
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))
model = YOLO("yolov8n.pt", "v8") # load a pretrained YOLOv8n model
EDFlag = False

def EDD_record_duration(text, img):
    global start_time, end_time, prev_state, flag, writer,recorded_Images,EDD_Duration, video
    if text == "Electronic Device Detected" and prev_state == "No Electronic Device Detected":
        start_time = time.time()
        writer.write(img)
    elif text == "Electronic Device Detected" and str(text) == prev_state and (time.time() - start_time) > 4:
        flag = True
        writer.write(img)
    elif text == "Electronic Device Detected" and str(text) == prev_state and (time.time() - start_time) <= 4:
        flag = False
        writer.write(img)
    else:
        if prev_state == "Electronic Device Detected":
            writer.release()
            end_time = time.time()
            duration = math.ceil((end_time - start_time)/10)
            EDViolation = {
                "Name": prev_state,
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                "Duration": str(duration) + " seconds",
                "Mark": (2 * (duration)),
                "Link": video
            }
            if flag:
                write_json(EDViolation)
                move_file_to_output_images(video)
            else:
                os.remove(video)
            video= str(random.randint(1, 50000)) + "EDViolation.avi"
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"XVID"), 10 , (width,height))
            flag = False

    prev_state=text

def move_file_to_output_images(file_name):
    # Get the current working directory (project folder)
    current_directory = os.getcwd()
    # Define the paths for the source file and destination folder
    source_path = os.path.join(current_directory, file_name)
    destination_path = os.path.join(current_directory,'OutputVideos', file_name)
    try:
        # Use 'shutil.move' to move the file to the destination folder
        shutil.move(source_path, destination_path)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found in the project folder.")
    except shutil.Error as e:
        print(f"Error: Failed to move the file. {e}")

# function to add to JSON
def write_json(new_data, filename='violation.json'):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data.append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

def electronicDevicesDetection(frame):
    global model, EDFlag
    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    for result in detect_params:  # iterate results
        boxes = result.boxes.cpu().numpy()  # get boxes on cpu in numpy
        for box in boxes:  # iterate boxes
            r = box.xyxy[0].astype(int)  # get corner points as int
            detected_obj = result.names[int(box.cls[0])]
            if (detected_obj == 'cell phone' or detected_obj == 'remote' or detected_obj == 'laptop' or detected_obj == 'laptop,book'): EDFlag = True
    # Display the resulting frame
    if EDFlag:
        text = 'Electronic Device Detected'
    else:
        text = "No Electronic Device Detected"
    EDD_record_duration(text, frame)
    print(text)
    EDFlag = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    electronicDevicesDetection(frame)
    if keyboard.is_pressed('q'):
        break


# When everything done, release the capture
cap.release()
