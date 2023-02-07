import cv2
import mediapipe as mp
import os

import numpy as np
import simpleaudio
import keyboard
import time
from utils.file_io import yaml_load, get_sound_object
from utils.vector_operations import update_positions_and_play_sounds

'''
'LEFT_ANKLE', 'LEFT_EAR', 'LEFT_ELBOW', 'LEFT_EYE', 'LEFT_EYE_INNER', 'LEFT_EYE_OUTER', 
'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_HIP', 'LEFT_INDEX', 'LEFT_KNEE', 'LEFT_PINKY', 'LEFT_SHOULDER', 'LEFT_THUMB',
 'LEFT_WRIST', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'NOSE', 'RIGHT_ANKLE', 'RIGHT_EAR', 'RIGHT_ELBOW', 'RIGHT_EYE', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER', 
'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_HIP', 'RIGHT_INDEX', 'RIGHT_KNEE', 'RIGHT_PINKY', 'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_WRIST'
'''

def draw_landmarks():
    #TODO Draw drums or other cues on the screen
    pass

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Load from config and setup
config = yaml_load(os.path.abspath(os.path.dirname(os.curdir)) + '/configs/sound_mapping.yaml')
frame_lag = config.frame_lag
# parts_tracked = config.parts_tracked or {}
# sound_objects = load_sound_mappings(parts_tracked)
# position_stacks, motion_vectors = get_stacks(frame_lag, parts_tracked)

moves = []

all_parts = ['LEFT_ANKLE', 'LEFT_ELBOW', 'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_HIP', 'LEFT_INDEX', 'LEFT_KNEE', 'LEFT_SHOULDER',
 'LEFT_WRIST', 'RIGHT_ANKLE', 'RIGHT_ELBOW', 'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_SHOULDER', 'RIGHT_WRIST']

all_parts_indexed = {str(idx): part for idx, part in enumerate(all_parts)}
positions = {part: np.zeros((frame_lag,2), dtype=float) - np.ones((frame_lag, 2), dtype=float) for part in all_parts}


if "background_music" in config:
    base_path = os.path.abspath(os.path.dirname(os.curdir)) + "/sounds/"
    background_sound_object = simpleaudio.WaveObject.from_wave_file(base_path + config.background_music)
    background_sound_object.play()

record_mode = False
cap = cv2.VideoCapture(0)
recorded_movement = []
landmark = None
sound_file = None
min_frame_rate = 1000000

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    fps_time = time.time()
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if keyboard.is_pressed('q'):
        break
    if keyboard.is_pressed('r'):
        print("Entering record mode...")
        done = False
        while not done:
            try:
                print("Enter idx of parts to track separated by commas: " + str(all_parts_indexed))
                rec_body_parts = input("Here: ").strip()
                if rec_body_parts[0] == 'r':
                    rec_body_parts = rec_body_parts[1:]
                idxs = rec_body_parts.replace(' ', '').split(',')
                recorded_parts = [all_parts_indexed[i] for i in idxs]
                sound_file = input("Enter name of .wav file: ").strip()
                if sound_file[-4:] != '.wav':
                    sound_file += '.wav'
                recorded_sound_object = get_sound_object(sound_file)
                done = True

            except Exception as e:
                print("Error with opening sound file or mapping body part.. ")

        record_mode = True
        count_down_start_time = time.time()
        record_start_time=None

    elif record_mode and count_down_start_time and abs(time.time() - count_down_start_time) > 5:
        record_start_time = time.time()
        count_down_start_time = None
    elif record_mode and record_start_time and abs(time.time() - record_start_time) > 1 and positions:
        record_mode = False
        record_start_time = None
        motion_array = np.array([positions[body_part] for body_part in recorded_parts], dtype=float)
        if -1 in motion_array[:, :]:
            print('Some part of move not found in look back window')
        else:
            move = {}
            move['tracked_parts'] = recorded_parts
            move['motion_array'] = motion_array
            move['sound_object'] = recorded_sound_object
            moves.append(move)
            print(move)
    if record_mode and record_start_time:
        cv2.circle(image, (10, 20), 10, (255, 0, 0), -1)
    elif record_mode and count_down_start_time:
        cv2.putText(image, str(int(time.time()-count_down_start_time)), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    if results and results.pose_landmarks:
        landmark = results.pose_landmarks.landmark
        if landmark:
            positions = update_positions_and_play_sounds(landmark, positions, all_parts, frame_lag, moves)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    cv2.putText(image,
                "FPS: %f" % (1.0 / (time.time() - fps_time)),
                (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    min_frame_rate = min((1.0 / (time.time() - fps_time), min_frame_rate))

    # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    cv2.imshow("Jacks Pose", image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
print("FRAME RATE MINIMUM")
print(min_frame_rate)
cap.release()