import argparse
import logging
import os
import time
import yaml

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from scipy.spatial import distance
import simpleaudio

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


def load_sound_mappings(config):
    base_path = os.path.abspath(os.path.dirname(os.curdir)) + "/sounds/"
    sound_objects = {}
    for key, item in config.items():
        body_idx = item['body_idx']
        sound_objects[body_idx] = {}
        sound_objects[body_idx]['vector'] = item['vector']
        sound_objects[body_idx]['wav_object'] = simpleaudio.WaveObject.from_wave_file(base_path + item['sound_file'])
        sound_objects[body_idx]['play_object'] = sound_objects[body_idx]['wav_object'].play()
    return sound_objects

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='mobilenet_thin_432x368', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.model)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    #TODO read in more sounds here
    #TODO add config file that mapps sounds to body parts or locations

    with open(os.path.abspath(os.path.dirname(os.curdir)) + '/configs/sound_mapping.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        logger.debug("Read in sound mapping file")
    logger.debug(config)
    sound_objects = load_sound_mappings(config)

    parts_tracked = [2,3,4,5,7,9,10,12,13]
    frames_tracked = 24
    position_stacks = {}

    motion_vectors = {}

    for key in parts_tracked:
        position_stacks[key] = [[-1,-1] for i in range(frames_tracked)]
        motion_vectors[key] = [0,0]
    logger.debug(position_stacks)
    while True:
        ret_val, image = cam.read()

        # logger.debug('image preprocess+')
        if args.zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        # logger.debug('image process+')
        humans = e.inference(image)

        #TODO put this in a method
        #TODO figure out how to do motion rather than position

        if humans:
            #TODO make this work for multiple humans
            human = humans[0]

            #This code plays if body part appears:
            # for body_idx in list(sound_objects.keys()):
            #     if body_idx in list(human.body_parts.keys()) and not sound_objects[body_idx]['play_object'].is_playing():
            #         sound_objects[body_idx]['play_object'] = sound_objects[body_idx]['wav_object'].play()

            tracked_in_frame = list(set(parts_tracked).intersection(set(human.body_parts.keys())))
            for key in tracked_in_frame:
                position_stacks[key] = [[human.body_parts[key].x, human.body_parts[key].y]] + position_stacks[key]
                position_stacks[key].pop()

                # Make sure that the body part has been in the frame for long enough
                if position_stacks[key][-1][0] != -1:
                    motion_vectors[key] = [position_stacks[key][-1][0] - position_stacks[key][0][0], position_stacks[key][-1][1] - position_stacks[key][0][1]]

            tracked_not_in_frame = list(set(parts_tracked).difference(set(human.body_parts.keys())))
            for key in tracked_not_in_frame:
                position_stacks[key] = [[-1, -1] for i in range(frames_tracked)]
                motion_vectors[key] = [0,0]

            for body_idx in list(sound_objects.keys()):
                if body_idx in list(human.body_parts.keys()) and not sound_objects[body_idx]['play_object'].is_playing() \
                        and distance.cosine(sound_objects[body_idx]['vector'], motion_vectors[body_idx]) < .1:
                    sound_objects[body_idx]['play_object'] = sound_objects[body_idx]['wav_object'].play()

        else:
            for key in parts_tracked:
                position_stacks[key] = [[-1,-1] for i in range(frames_tracked)]
                motion_vectors[key] = [0,0]

        # logger.debug('postprocess+')
        #TODO can remove this for speed up (helpful for debug of poses)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        # logger.debug('finished+')
        logger.debug("++")
        logger.debug(position_stacks)
        logger.debug('------------------')
        logger.debug(motion_vectors)
        logger.debug("++")


    cv2.destroyAllWindows()
