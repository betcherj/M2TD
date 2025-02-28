import yaml
import simpleaudio
import os
from easydict import EasyDict as edict

def yaml_load(fileName):
    fc = None
    with open(fileName, 'r') as f:
        fc = edict(yaml.load(f, Loader=yaml.FullLoader))

    return fc

def load_sound_mappings(parts_tracked,sound_objects={}):
    # If this gives unknown format 3 error the wav file is unreadable
    base_path = os.path.abspath(os.path.dirname(os.curdir)) + "/sounds/"
    if parts_tracked:
        for body_part, info in parts_tracked.items():
            sound_objects[body_part] = {}
            sound_objects[body_part]['vector'] = info['vector']
            sound_objects[body_part]['wav_object'] = simpleaudio.WaveObject.from_wave_file(base_path + info['sound_file'])
            sound_objects[body_part]['play_object'] = sound_objects[body_part]['wav_object'].play()
    return sound_objects


def get_sound_object(file_name):
    try:
        base_path = os.path.abspath(os.path.dirname(os.curdir)) + "/sounds/"
        sound_object = {}
        sound_object['wav_object'] = simpleaudio.WaveObject.from_wave_file(os.path.join(base_path, file_name))
        sound_object['play_object'] = sound_object['wav_object'].play()
    except Exception as e:
        print("Unable to load sound object for " + file_name)
        raise e
    return sound_object



