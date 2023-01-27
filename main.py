import cv2
import mediapipe as mp
import os
import simpleaudio
import keyboard
import yaml
import numpy
import time
from utils.file_io import yaml_load
from scipy.spatial import distance

'''
'LEFT_ANKLE', 'LEFT_EAR', 'LEFT_ELBOW', 'LEFT_EYE', 'LEFT_EYE_INNER', 'LEFT_EYE_OUTER', 
'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_HIP', 'LEFT_INDEX', 'LEFT_KNEE', 'LEFT_PINKY', 'LEFT_SHOULDER', 'LEFT_THUMB',
 'LEFT_WRIST', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'NOSE', 'RIGHT_ANKLE', 'RIGHT_EAR', 'RIGHT_ELBOW', 'RIGHT_EYE', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER', 
'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_HIP', 'RIGHT_INDEX', 'RIGHT_KNEE', 'RIGHT_PINKY', 'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_WRIST'
'''
def load_sound_mappings(parts_tracked):
    base_path = os.path.abspath(os.path.dirname(os.curdir)) + "/sounds/"
    sound_objects = {}
    for body_part, info in parts_tracked.items():
        sound_objects[body_part] = {}
        sound_objects[body_part]['vector'] = info['vector']
        sound_objects[body_part]['wav_object'] = simpleaudio.WaveObject.from_wave_file(base_path + info['sound_file'])
        sound_objects[body_part]['play_object'] = sound_objects[body_part]['wav_object'].play()
    return sound_objects

def get_stacks(frame_lag, parts_tracked):
    position_stacks = {}

    motion_vectors = {}

    for body_part, info in parts_tracked.items():
        position_stacks[body_part] = [[-1, -1] for i in range(frame_lag)]
        motion_vectors[body_part] = [0, 0]
    return position_stacks, motion_vectors


def play_sounds(landmark, position_stacks, motion_vectors, parts_tracked, frame_lag):
    #TODO edit this to work with the new tracking

    # This code plays if body part appears:
    # for body_idx in list(sound_objects.keys()):
    #     if body_idx in list(human.body_parts.keys()) and not sound_objects[body_idx]['play_object'].is_playing():
    #         sound_objects[body_idx]['play_object'] = sound_objects[body_idx]['wav_object'].play()

    #Check if the item is in the frame
    tracked_in_frame = [body_part for body_part, info in parts_tracked.items() if landmark[mp_pose.PoseLandmark[body_part]].visibility > .5]

    for key in tracked_in_frame:
        position_stacks[key] = [[landmark[mp_pose.PoseLandmark[key]].x, landmark[mp_pose.PoseLandmark[key]].y]] + position_stacks[key]
        position_stacks[key].pop()

        # Make sure that the body part has been in the frame for long enough
        if position_stacks[key][-1][0] != -1:
            motion_vectors[key] = [position_stacks[key][-1][0] - position_stacks[key][0][0],
                                   position_stacks[key][-1][1] - position_stacks[key][0][1]]
        else:
            motion_vectors[key] = [0, 0]

    tracked_not_in_frame = list(set(parts_tracked).difference(set(tracked_in_frame)))

    for key in tracked_not_in_frame:
        position_stacks[key] = [[-1, -1] for i in range(frame_lag)]
        motion_vectors[key] = [0, 0]

    for body_part in tracked_in_frame:
        if not sound_objects[body_part]['play_object'].is_playing() \
                and abs(motion_vectors[body_part][0]) + abs(motion_vectors[body_part][1]) > 0 \
                and distance.cosine(sound_objects[body_part]['vector'], motion_vectors[body_part]) < .05:
            sound_objects[body_part]['play_object'] = sound_objects[body_part]['wav_object'].play()
            # # Maybe we want to clear the position and motion vectors here
            for body_part, info in parts_tracked.items():
                position_stacks[body_part] = [[-1, -1] for i in range(frame_lag)]
                motion_vectors[body_part] = [0, 0]

    return position_stacks, motion_vectors

def draw_landmakrs():
    #TODO Draw drums or other cues on the screen
    pass

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Load from config and setup
config = yaml_load(os.path.abspath(os.path.dirname(os.curdir)) + '/configs/sound_mapping.yaml')
frame_lag = config.frame_lag
parts_tracked = config.parts_tracked
sound_objects = load_sound_mappings(parts_tracked)
position_stacks, motion_vectors = get_stacks(frame_lag, parts_tracked)

if "background_music" in config:
    base_path = os.path.abspath(os.path.dirname(os.curdir)) + "/sounds/"
    background_sound_object = simpleaudio.WaveObject.from_wave_file(base_path + config.background_music)
    background_sound_object.play()

record_mode = False
cap = cv2.VideoCapture(0)
recorded_movement = []

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
        print("Entering Record Mode...")
        recorded_movement = []
        record_mode = True
    if keyboard.is_pressed('space'):
        print("End Record Mode...")
        record_mode = False
        print(recorded_movement)


    if results and results.pose_landmarks and results.pose_landmarks.landmark:
        landmark = results.pose_landmarks.landmark

    if landmark:
        # This method plays sounds based on movements defined in the config file
        #TODO seperate updating position stacks from the sounds
        position_stacks, motion_vectors = play_sounds(landmark, position_stacks, motion_vectors, parts_tracked, frame_lag)
        if record_mode:
            if landmark[mp_pose.PoseLandmark["RIGHT_WRIST"]].visibility > .5:
                recorded_movement = [[landmark[mp_pose.PoseLandmark["RIGHT_WRIST"]].x,
                                      landmark[mp_pose.PoseLandmark["RIGHT_WRIST"]].y]] + recorded_movement
            else:
                recorded_movement = []

            cv2.circle(image, (10, 20), 10, (0, 0, 255), -1)
        elif recorded_movement and numpy.allclose(position_stacks, recorded_movement[:1+frame_lag]):
            #Play sound if we got the recorded move
            print("HERE!")
            simpleaudio.WaveObject.from_wave_file(base_path + "sound_6.wav").play()

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