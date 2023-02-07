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

all_parts_indexed = {idx: part for idx, part in enumerate(all_parts)}
positions = {part: np.zeros(frame_lag) for part in all_parts}


if "background_music" in config:
    base_path = os.path.abspath(os.path.dirname(os.curdir)) + "/sounds/"
    background_sound_object = simpleaudio.WaveObject.from_wave_file(base_path + config.background_music)
    background_sound_object.play()

record_mode = False
cap = cv2.VideoCapture(0)
recorded_movement = []
landmark = None
sound_file = None


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

    if keyboard.is_pressed('r'):
        print("Entering record mode...")
        done = False
        while not done:
            try:
                print("Enter idx of parts to track separated by commas: " + str(enumerate(all_parts)))
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
        record_start_time = time.time()

    if record_mode and abs(time.time() - record_start_time) > 5:
        record_mode = False
        move = {}
        move['tracked_parts'] = recorded_parts
        move['motion_array'] = np.array([positions[body_part] for body_part in recorded_parts], dtype=float)
        move['sound_object'] = recorded_sound_object
        moves += move

    # if keyboard.is_pressed('space'):
    #     record_mode = False
    #     if not recorded_movement:
    #         print('No recorded movement..')
    #         print("Exiting Record Mode...")
    #     else:
    #         try:
    #             # TODO add move to moves here
    #
    #             # Add part to stacks
    #             # position_stacks[rec_body_part] = [[-1, -1] for i in range(frame_lag)]
    #             # motion_vectors[rec_body_part] = [0, 0]
    #             # parts_tracked[rec_body_part] = {'sound_file': file, 'vector': [recorded_movement[-1][0] - recorded_movement[0][0], recorded_movement[-1][1] - recorded_movement[0][1]]}
    #             # sound_objects = load_sound_mappings(parts_tracked, sound_objects)
    #         except Exception as e:
    #             print(e)
    #             print("Failed to add recorded movement")
    #         print(recorded_movement)

    if results and results.pose_landmarks:
        landmark = results.pose_landmarks.landmark
        if landmark:
            positions = update_positions_and_play_sounds(landmark, positions, all_parts, frame_lag, moves)
            # This method plays sounds based on movements defined in the config file
            #TODO seperate updating position stacks from the sounds
            #if parts_tracked:
                #position_stacks, motion_vectors = play_sounds(landmark, sound_objects, position_stacks, motion_vectors, parts_tracked, frame_lag)

            if record_mode:
                # if landmark[mp_pose.PoseLandmark[rec_body_part]].visibility > .5:
                #     recorded_movement = [[landmark[mp_pose.PoseLandmark[rec_body_part]].x,
                #                           landmark[mp_pose.PoseLandmark[rec_body_part]].y]] + recorded_movement
                cv2.circle(image, (10, 20), 10, (0, 0, 255), -1)

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

    # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    cv2.imshow("Jacks Pose", image)

    # Frames per second
    # print("FPS: %f" % (1.0 / (time.time() - fps_time)))
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()