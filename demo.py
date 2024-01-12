import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time
import tkinter as tk
from tkinter import ttk

# Load the pre-trained facial landmark predictor
predictor_path = "C:/Users/Atif Traders/Music/eye trace usa/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Initialize the video capture from the webcam (0 corresponds to the default camera)
cap = cv2.VideoCapture(0)

# Define the eye landmarks indices
left_eye_indices = list(range(42, 48))
right_eye_indices = list(range(36, 42))

# Initialize variables for metrics calculation
blink_start_time = time.time()
pupil_size, blink_rate, blink_duration = 0.0, 0.0, 0.0
gaze_direction, fixation_duration, saccade_speed, pupil_light_response = 0.0, 0.0, 0.0, 0.0
fixation_start_time, saccade_detected, saccade_start_time = None, False, None
last_gaze_point = None
left_eye_micro_intensity, right_eye_micro_intensity = 0.0, 0.0

# Create the main window
root = tk.Tk()
root.title("Eye Tracking Metrics")

# Create labels for metrics
pupil_size_label = ttk.Label(root, text="Pupil Size: 0.0")
pupil_size_label.pack()

blink_rate_label = ttk.Label(root, text="Blink Rate: 0.0 blinks per minute")
blink_rate_label.pack()

blink_duration_label = ttk.Label(root, text="Blink Duration: 0.0 seconds")
blink_duration_label.pack()

gaze_direction_label = ttk.Label(root, text="Gaze Direction: 0.0 pixels")
gaze_direction_label.pack()

fixation_duration_label = ttk.Label(root, text="Fixation Duration: 0.0 seconds")
fixation_duration_label.pack()

saccade_speed_label = ttk.Label(root, text="Saccade Speed: 0.0 pixels/second")
saccade_speed_label.pack()

pupil_light_response_label = ttk.Label(root, text="Pupil Dilation Response to Light: 0.0")
pupil_light_response_label.pack()

left_eye_micro_label = ttk.Label(root, text="Left Eye Micro-Expression Intensity: 0.0")
left_eye_micro_label.pack()

right_eye_micro_label = ttk.Label(root, text="Right Eye Micro-Expression Intensity: 0.0")
right_eye_micro_label.pack()

# Function to update metrics and display them in the GUI
def update_metrics():
    global pupil_size, blink_rate, blink_duration, gaze_direction, fixation_duration, saccade_speed, pupil_light_response
    global left_eye_micro_intensity, right_eye_micro_intensity

    # Calculate eye metrics
    pupil_size, blink_rate, blink_duration, gaze_direction, fixation_duration, saccade_speed, pupil_light_response = calculate_eye_metrics(left_eye_landmarks, right_eye_landmarks)

    # Simulate micro-expressions around the eyes
    left_eye_micro_intensity = np.random.rand()
    right_eye_micro_intensity = np.random.rand()

    # Update GUI labels
    pupil_size_label.config(text=f"Pupil Size: {pupil_size:.2f}")
    blink_rate_label.config(text=f"Blink Rate: {blink_rate:.2f} blinks per minute")
    blink_duration_label.config(text=f"Blink Duration: {blink_duration:.2f} seconds")
    gaze_direction_label.config(text=f"Gaze Direction: {gaze_direction:.2f} pixels")
    fixation_duration_label.config(text=f"Fixation Duration: {fixation_duration:.2f} seconds")
    saccade_speed_label.config(text=f"Saccade Speed: {saccade_speed:.2f} pixels/second")
    pupil_light_response_label.config(text=f"Pupil Dilation Response to Light: {pupil_light_response:.2f}")
    left_eye_micro_label.config(text=f"Left Eye Micro-Expression Intensity: {left_eye_micro_intensity:.2f}")
    right_eye_micro_label.config(text=f"Right Eye Micro-Expression Intensity: {right_eye_micro_intensity:.2f}")

    # Schedule the function to be called after 100 milliseconds
    root.after(100, update_metrics)

# Function to calculate eye metrics
def calculate_eye_metrics(left_eye_landmarks, right_eye_landmarks):
    global pupil_size, blink_rate, blink_duration, gaze_direction, fixation_duration, saccade_speed, pupil_light_response
    global fixation_start_time, saccade_detected, saccade_start_time, last_gaze_point

    # Calculate pupil size
    left_pupil_size = distance.euclidean(left_eye_landmarks[0], left_eye_landmarks[3])
    right_pupil_size = distance.euclidean(right_eye_landmarks[0], right_eye_landmarks[3])
    pupil_size = (left_pupil_size + right_pupil_size) / 2

    # Calculate blink duration and rate
    blink_duration = time.time() - blink_start_time
    blink_rate = 60 / (blink_duration + 1e-8)  # Avoid division by zero

    # Calculate gaze direction
    current_gaze_point = np.mean(np.concatenate([left_eye_landmarks, right_eye_landmarks], axis=0), axis=0)
    if last_gaze_point is not None:
        gaze_direction = distance.euclidean(last_gaze_point, current_gaze_point)

    # Detect fixation
    if gaze_direction < 5:  # Adjust threshold as needed
        if fixation_start_time is None:
            fixation_start_time = time.time()
    else:
        if fixation_start_time is not None:
            fixation_duration = time.time() - fixation_start_time
            fixation_start_time = None

    # Detect saccades
    if not saccade_detected and gaze_direction >= 20:  # Adjust threshold as needed
        saccade_detected = True
        saccade_start_time = time.time()
    elif saccade_detected and gaze_direction < 5:
        saccade_duration = time.time() - saccade_start_time
        saccade_detected = False
        saccade_speed = gaze_direction / saccade_duration

    # Simulate micro-expressions around the eyes
    left_eye_micro_intensity = np.random.rand()  # Simulate micro-expression intensity (0 to 1)
    right_eye_micro_intensity = np.random.rand()

    # Simulate pupil dilation response to light
    pupil_light_response = np.random.rand()

    # Print and return the calculated metrics
    print(f"Pupil Size: {pupil_size:.2f}")
    print(f"Blink Rate: {blink_rate:.2f} blinks per minute")
    print(f"Blink Duration: {blink_duration:.2f} seconds")
    print(f"Gaze Direction: {gaze_direction:.2f} pixels")
    print(f"Fixation Duration: {fixation_duration:.2f} seconds")
    print(f"Saccade Speed: {saccade_speed:.2f} pixels/second")
    print(f"Pupil Dilation Response to Light: {pupil_light_response:.2f}")
    print(f"Left Eye Micro-Expression Intensity: {left_eye_micro_intensity:.2f}")
    print(f"Right Eye Micro-Expression Intensity: {right_eye_micro_intensity:.2f}")

    return pupil_size, blink_rate, blink_duration, gaze_direction, fixation_duration, saccade_speed, pupil_light_response

# Function to update landmarks and metrics
def update_landmarks_metrics():
    global left_eye_landmarks, right_eye_landmarks, last_gaze_point

    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a face detector (e.g., Haarcascades or a deep learning model) to detect faces
    # For simplicity, we use dlib's frontal face detector here
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    for face in faces:
        # Use facial landmarks to determine eye positions
        landmarks = predictor(gray, face)

        # Extract eye landmarks
        left_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_indices])
        right_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_indices])

        # Draw landmarks on the frame for visualization
        for landmark in landmarks.parts():
            cv2.circle(frame, (landmark.x, landmark.y), 2, (0, 255, 0), -1)

        # Update last gaze point
        last_gaze_point = np.mean(np.concatenate([left_eye_landmarks, right_eye_landmarks], axis=0), axis=0)

    # Display the frame with landmarks
    cv2.imshow('Webcam Feed', frame)

    # Schedule the function to be called after 100 milliseconds
    root.after(100, update_landmarks_metrics)

# Schedule the functions to be called periodically
root.after(100, update_landmarks_metrics)
root.after(100, update_metrics)

# Run the main loop
root.mainloop()

# Release the video capture, destroy the named window, and close all windows
cap.release()
cv2.destroyAllWindows()
