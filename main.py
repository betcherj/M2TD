import cv2
import mediapipe as mp
import os
import simpleaudio
import keyboard
import yaml
import numpy
import time
from utils.file_io import yaml_load, load_sound_mappings
from utils.vector_operations import compare_vectors, get_stacks, play_sounds
from scipy.spatial import distance

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
#Todo map more than one move per part
parts_tracked = config.parts_tracked or {}
sound_objects = load_sound_mappings(parts_tracked)
position_stacks, motion_vectors = get_stacks(frame_lag, parts_tracked)

if "background_music" in config:
    base_path = os.path.abspath(os.path.dirname(os.curdir)) + "/sounds/"
    background_sound_object = simpleaudio.WaveObject.from_wave_file(base_path + config.background_music)
    background_sound_object.play()

record_mode = False
cap = cv2.VideoCapture(0)
recorded_movement = []
landmark = None

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
                rec_body_part = input("Enter body part to track: ").strip()
                file = input("Enter name of .wav file: ").strip()
                if file[-4:] != '.wav':
                    file += '.wav'
                done = True
            except Exception as e:
                print("Error with opening sound file or mapping body part.. ")
        recorded_movement = []
        record_mode = True

    if keyboard.is_pressed('space'):
        record_mode = False
        if not recorded_movement:
            print('No recorded movement..')
            print("Exiting Record Mode...")
        else:
            try:
                # Add part to stacks
                position_stacks[rec_body_part] = [[-1, -1] for i in range(frame_lag)]
                motion_vectors[rec_body_part] = [0, 0]

                parts_tracked[rec_body_part] = {'sound_file': file, 'vector': [recorded_movement[-1][0]-recorded_movement[0][0], recorded_movement[-1][1]-recorded_movement[0][1]]}
                sound_objects = load_sound_mappings(parts_tracked)
            except Exception as e:
                print(e)
                print("Failed to add recorded movement")
            print(recorded_movement)

    if results and results.pose_landmarks:
        landmark = results.pose_landmarks.landmark

        if landmark:
            # This method plays sounds based on movements defined in the config file
            #TODO seperate updating position stacks from the sounds
            if parts_tracked:
                position_stacks, motion_vectors = play_sounds(landmark, sound_objects, position_stacks, motion_vectors, parts_tracked, frame_lag)

            if record_mode:
                if landmark[mp_pose.PoseLandmark[rec_body_part]].visibility > .5:
                    recorded_movement = [[landmark[mp_pose.PoseLandmark[rec_body_part]].x,
                                          landmark[mp_pose.PoseLandmark[rec_body_part]].y]] + recorded_movement
                cv2.circle(image, (10, 20), 10, (0, 0, 255), -1)
            #TODO use the allclose method to map more complex movements

            # elif recorded_movement and numpy.allclose(position_stacks, recorded_movement[:1+frame_lag]):
            #     #Play sound if we do the recored move correcly
            #     print("HERE!")
            #     simpleaudio.WaveObject.from_wave_file(base_path + "sound_6.wav").play()

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