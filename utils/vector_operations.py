import numpy
import mediapipe as mp
from scipy.spatial import distance

def compare_vectors(a, b):
    return numpy.allclose(a, b)


def get_stacks(frame_lag, parts_tracked):
    position_stacks = {}
    motion_vectors = {}
    if parts_tracked:
        for body_part, info in parts_tracked.items():
            position_stacks[body_part] = [[-1, -1] for i in range(frame_lag)]
            motion_vectors[body_part] = [0, 0]
    return position_stacks, motion_vectors


def play_sounds(landmark, sound_objects, position_stacks, motion_vectors, parts_tracked, frame_lag):
    #Check if the item is in the frame
    tracked_in_frame = [body_part for body_part, info in parts_tracked.items() if landmark[mp.solutions.pose.PoseLandmark[body_part]].visibility > .5]

    for key in tracked_in_frame:
        position_stacks[key] = [[landmark[mp.solutions.pose.PoseLandmark[key]].x, landmark[mp.solutions.pose.PoseLandmark[key]].y]] + position_stacks[key]
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
            # Maybe we want to clear the position and motion vectors here (Helps with repeat plays)
            position_stacks[body_part] = [[-1, -1] for i in range(frame_lag)]
            motion_vectors[body_part] = [0, 0]

    return position_stacks, motion_vectors