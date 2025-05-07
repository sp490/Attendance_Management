import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime
import os
import tkinter as tk
from tkinter import ttk
from threading import Thread
import queue

# Directory for known face images
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = f"attendance_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

known_face_encodings = []
known_face_names = []
attendance_marked = set()
running = False
attendance_queue = queue.Queue()

def load_known_faces():
    for name in os.listdir(KNOWN_FACES_DIR):
        for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
            image_path = f"{KNOWN_FACES_DIR}/{name}/{filename}"
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                print(f"Loaded face for {name} from {filename}")
            else:
                print(f"Warning: No face detected in {image_path}. Skipping.")
    print("Known faces loaded:", known_face_names)

def mark_attendance(name):
    if name not in attendance_marked:
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if os.stat(ATTENDANCE_FILE).st_size == 0:
                writer.writerow(["Name", "Time"])
            time_now = datetime.now().strftime("%H:%M:%S")
            writer.writerow([name, time_now])
            attendance_marked.add(name)
            attendance_queue.put((name, time_now))
            print(f"Attendance marked for {name} at {time_now}")

def is_blinking(landmarks, frame_count, prev_eye_state):
    # Get eye landmarks (left and right eye)
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    
    # Calculate Eye Aspect Ratio (EAR) for both eyes
    def eye_aspect_ratio(eye):
        vertical_1 = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        vertical_2 = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        horizontal = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (vertical_1 + vertical_2) / (2.0 * horizontal)
    
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Threshold for blinking (adjustable)
    EAR_THRESHOLD = 0.25
    current_state = avg_ear < EAR_THRESHOLD  # True if eyes closed
    
    # Detect blink if state changes from open to closed
    if frame_count > 1 and prev_eye_state.get('open', True) and current_state:
        return True, {'open': False}
    return False, {'open': current_state}

def face_recognition_thread(root):
    global running
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0
    prev_eye_states = {}  # Track eye state per face

    while running:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)

        frame_count += 1

        for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # Check for blinking
            face_id = f"{top}_{right}_{bottom}_{left}"  # Unique ID for this face in frame
            blinked, new_state = is_blinking(landmarks, frame_count, prev_eye_states.get(face_id, {}))
            prev_eye_states[face_id] = new_state

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = name if blinked else f"{name} (Blink to confirm)"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if name != "Unknown" and blinked:
                mark_attendance(name)

        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    running = False
    root.quit()

def update_attendance_list(root, listbox):
    try:
        while not attendance_queue.empty():
            name, time = attendance_queue.get()
            listbox.insert(tk.END, f"{name} - {time}")
    except queue.Empty:
        pass
    root.after(100, update_attendance_list, root, listbox)

def start_recognition(root):
    global running
    if not running:
        running = True
        Thread(target=face_recognition_thread, args=(root,), daemon=True).start()
        status_label.config(text="Status: Running", foreground="green")

def stop_recognition():
    global running
    running = False
    status_label.config(text="Status: Stopped", foreground="red")

# GUI Setup
root = tk.Tk()
root.title("Attendance Management System")
root.geometry("400x500")
root.configure(bg="#2C3E50")

title_label = tk.Label(root, text="Attendance System", font=("Helvetica", 20, "bold"), fg="white", bg="#2C3E50")
title_label.pack(pady=20)

status_label = tk.Label(root, text="Status: Stopped", font=("Helvetica", 12), fg="red", bg="#2C3E50")
status_label.pack(pady=10)

button_frame = tk.Frame(root, bg="#2C3E50")
button_frame.pack(pady=10)
start_button = ttk.Button(button_frame, text="Start", command=lambda: start_recognition(root))
start_button.grid(row=0, column=0, padx=10)
stop_button = ttk.Button(button_frame, text="Stop", command=stop_recognition)
stop_button.grid(row=0, column=1, padx=10)

listbox_frame = tk.Frame(root, bg="#2C3E50")
listbox_frame.pack(pady=20, fill=tk.BOTH, expand=True)
scrollbar = tk.Scrollbar(listbox_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
attendance_listbox = tk.Listbox(listbox_frame, height=15, width=40, font=("Helvetica", 12), bg="#ECF0F1", fg="#2C3E50", yscrollcommand=scrollbar.set)
attendance_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.config(command=attendance_listbox.yview)

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)
    print(f"Created directory '{KNOWN_FACES_DIR}'. Add face images there.")
load_known_faces()

root.after(100, update_attendance_list, root, attendance_listbox)
root.mainloop()