import numpy
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

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

def update_positions_and_play_sounds(landmark, positions, all_parts, frame_lag, moves):
    '''
    Args:
        positions: dictionary k: body part name, v: x, y coordinates
        landmark: media pipe object containing the locations of all the parts in the frame
        all_parts: all tracked body parts
        frame_lag: the length of the pos array
        sound_objects: play sounds while updating positions
    Returns:
        positions: Updated positions dictionary
    '''
    parts_in_frame = [body_part for body_part in all_parts if landmark[mp.solutions.pose.PoseLandmark[body_part]].visibility > .5]

    tracked_not_in_frame = list(set(all_parts).difference(set(parts_in_frame)))

    for body_part in tracked_not_in_frame:
        positions[body_part] = np.zeros((frame_lag, 2), dtype=float) - np.ones((frame_lag, 2), dtype=float)

    print(positions['RIGHT_WRIST'].shape)
    for body_part in parts_in_frame:
        print(np.delete(positions[body_part], -1, axis=0))
        positions[body_part] = np.concatenate(([[landmark[mp.solutions.pose.PoseLandmark[body_part]].x,
                                                landmark[mp.solutions.pose.PoseLandmark[body_part]].y]],
                                              np.delete(positions[body_part], -1, axis=0)), axis=0)

    to_reset = []
    for move in moves:
        parts = move['tracked_parts']
        stored_motion_array = move['motion_array']
        sound_object = move['sound_object']
        captured_motion_array = np.array([positions[body_part] for body_part in move['tracked_parts']], dtype=float)

        if not sound_object['play_object'].is_playing() \
                and np.allclose(captured_motion_array, stored_motion_array) < .05:
            sound_object['play_object'] = sound_object['wav_object'].play()
            # Maybe we want to clear the position and motion vectors here (Helps with repeat plays)
        to_reset += parts

    to_reset = set(to_reset)
    for body_part in to_reset:
        positions[body_part] = np.zeros((frame_lag, 2), dtype=float) - np.ones((frame_lag, 2), dtype=float)

    return positions



def play_sounds(landmark, sound_objects, position_stacks, motion_vectors, parts_tracked, frame_lag):
    '''
    Args:
        landmark:
        sound_objects:
        position_stacks:
        motion_vectors:
        parts_tracked:
        frame_lag:

    Returns:
        updated position_stacks and motion_vectors
    '''
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